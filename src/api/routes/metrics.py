"""
Operations dashboard data (admin: X-Admin-Key).
  GET /api/v1/metrics?hours=24 — requests, p50/p95 latency, error rate, LLM calls, tokens, cost,
                                 per-intent breakdown, and agent action outcomes (tool success rate).
Langfuse keeps the per-trace detail; this is the rollup that answers "how much, how fast, how often".
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src import usage
from src.agent import actions
from src.api.principal import require_admin

router = APIRouter(prefix="/api/v1", tags=["metrics"], dependencies=[Depends(require_admin)])


@router.get("/metrics")
def metrics(hours: int = Query(24, ge=1, le=24 * 90)):
    return usage.summary(hours) | {"actions": actions.stats()}
