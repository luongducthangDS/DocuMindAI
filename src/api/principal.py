"""
Resolve the caller into a RetrievalContext — the API's trust boundary.

The rule this file exists to enforce: a client states *who it is* (an API key),
never *what it may read*. ACL labels and tenant id are looked up server-side.
If the request body could carry `acl_labels`, any caller could award itself
`confidential` and the whole filter chain downstream would faithfully honour it.

Anonymous callers are not rejected — the labour-law corpus is public
legislation and answering it needs no identity. They get PUBLIC_CONTEXT, which
sees tenant `public` and label `public` only.
"""

from __future__ import annotations

import hmac
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException, Request, status
from loguru import logger

from src.rag.context import DEFAULT_ACL_LABEL, PUBLIC_TENANT, RetrievalContext

TENANTS_PATH = Path(__file__).resolve().parents[2] / "data" / "tenants" / "tenants.json"

API_KEY_HEADER = "x-api-key"


@dataclass(frozen=True)
class Tenant:
    tenant_id: str
    name: str
    api_key: str
    acl_labels: frozenset[str]


@lru_cache(maxsize=1)
def load_tenants() -> dict[str, Tenant]:
    """Seed tenants, keyed by api_key. Missing file = public-only deployment."""
    if not TENANTS_PATH.exists():
        logger.info("No tenants file at {} — running public-only", TENANTS_PATH)
        return {}
    try:
        raw = json.loads(TENANTS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        # Fail closed: an unreadable tenant file must not silently downgrade
        # every request to anonymous, because anonymous still gets answers and
        # nobody would notice the tenants had vanished.
        raise RuntimeError(f"Cannot read {TENANTS_PATH}: {exc}") from exc

    out: dict[str, Tenant] = {}
    for t in raw.get("tenants", []):
        key = str(t.get("api_key") or "").strip()
        if not key:
            continue
        out[key] = Tenant(
            tenant_id=str(t["tenant_id"]),
            name=str(t.get("name", t["tenant_id"])),
            api_key=key,
            acl_labels=frozenset(t.get("acl_labels") or [DEFAULT_ACL_LABEL]),
        )
    logger.info("Loaded {} tenant(s)", len(out))
    return out


class InvalidApiKey(Exception):
    """Raised by the header-level resolver; each transport renders its own error.

    HTTP turns this into 401, WebSocket into a close frame — a WS handshake has
    already completed by the time we read headers, so HTTPException is not a
    thing that can be raised there.
    """


def context_from_headers(headers, as_of_date: str | None = None) -> RetrievalContext:
    """Transport-agnostic resolution, shared by the REST and WebSocket paths.

    The WebSocket handler used to run its own retrieval with no context at all —
    which, since streaming is what the UI actually uses, meant the default path
    was the unfiltered one. One resolver for both transports is what stops that
    class of gap reappearing.
    """
    presented = (headers.get(API_KEY_HEADER) or "").strip()
    if not presented:
        return RetrievalContext(
            tenant_id=PUBLIC_TENANT,
            acl_labels=frozenset({DEFAULT_ACL_LABEL}),
            as_of_date=as_of_date,
        )

    tenants = load_tenants()
    # compare_digest against every known key rather than a dict lookup: a dict
    # hit/miss is measurably faster than a miss, which leaks whether a guessed
    # key prefix was right. Constant work, constant time, tiny tenant list.
    matched = None
    for key, tenant in tenants.items():
        if hmac.compare_digest(presented, key):
            matched = tenant

    if matched is None:
        logger.warning("Rejected unknown API key (len={})", len(presented))
        raise InvalidApiKey("API key không hợp lệ.")

    return RetrievalContext(
        tenant_id=matched.tenant_id,
        acl_labels=matched.acl_labels,
        as_of_date=as_of_date,
    )


def resolve_context(request: Request, as_of_date: str | None = None) -> RetrievalContext:
    """HTTP wrapper: same resolution, rendered as a 401 on failure."""
    try:
        return context_from_headers(request.headers, as_of_date)
    except InvalidApiKey as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
