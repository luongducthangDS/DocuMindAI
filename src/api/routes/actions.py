"""
Human approval for agent-proposed actions (admin: X-Admin-Key).
  GET  /api/v1/actions?status=pending
  POST /api/v1/actions/{id}/approve   — executes once
  POST /api/v1/actions/{id}/reject
  GET  /api/v1/actions/stats          — tool-call success rate, outcomes
"""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from src.agent import actions
from src.api.principal import require_admin

router = APIRouter(prefix="/api/v1/actions", tags=["actions"], dependencies=[Depends(require_admin)])


def _approver(request: Request) -> str:
    # Audit trail: who decided. The admin key authenticates; this header names the person.
    name = re.sub(r"[^\w.@ -]", "", request.headers.get("x-approver", ""))[:64].strip()
    return name or "admin"


@router.get("")
def list_actions(status_filter: str | None = Query(None, alias="status"), limit: int = 100):
    return actions.list_actions(status_filter, limit)


@router.get("/stats")
def action_stats():
    return actions.stats()


@router.post("/{action_id}/approve")
def approve(action_id: str, request: Request):
    row = actions.approve(action_id, _approver(request))
    if row is None:
        current = actions.get(action_id)
        if current is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown action")
        raise HTTPException(status.HTTP_409_CONFLICT, f"Action is already {current['status']}")
    return row


@router.post("/{action_id}/reject")
def reject(action_id: str, request: Request):
    row = actions.reject(action_id, _approver(request))
    if row is None:
        current = actions.get(action_id)
        if current is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown action")
        raise HTTPException(status.HTTP_409_CONFLICT, f"Action is already {current['status']}")
    return row
