"""
Retrieval context: who is asking, on behalf of which tenant, as of when.

This is the one object that turns "retrieve(query)" into an access-controlled,
tenant-scoped, point-in-time query. Three concerns that look separate — ACL,
multi-tenancy and effective dates — are all the same thing at the storage
layer: a predicate over chunk metadata, applied *before* the search runs.

Why pre-filter matters differently for each:
  - temporal: post-filtering costs recall. Measured on the 30-question gold set,
    filtering after a top-8 retrieve left avg_chunks=2.87 (reports/temporal_eval.json)
    — the reranker spent 5 of its 8 slots on clauses that were never in force.
  - ACL/tenant: post-filtering is a *leak*. A forbidden chunk that reaches
    `state["retrieved_chunks"]` has already passed through the reranker, the
    Langfuse span and the fallback path in graph.py before anything drops it.

Storage note (verified against chromadb 0.6.3, not assumed): `$lte`/`$gt`
reject string operands — "Expected operand value to be an int or a float".
That is why effective dates are mirrored into `effective_from_i` /
`effective_to_i` as YYYYMMDD integers. The ISO string fields stay as they are;
`src/rag/temporal.py` still reads those, and still runs as a second layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from src.ingestion.manifest import EFFECTIVE_TO_OPEN
from src.rag.temporal import _EXCLUDED_STATUS, today_iso

# Corpus-wide documents (Vietnamese labour law) belong to no single tenant:
# every tenant retrieves them, alongside whatever that tenant uploaded itself.
PUBLIC_TENANT = "public"

# Sentinel ints mirroring the ISO sentinels. A chunk with no effective_from is
# "in force since forever" rather than invisible — matching temporal.is_in_force,
# which keeps undatable chunks (a preamble, an annex) instead of dropping them.
EFFECTIVE_FROM_UNKNOWN = 0
EFFECTIVE_TO_OPEN_INT = 99991231

DEFAULT_ACL_LABEL = "public"


def to_date_int(iso: str | None) -> int | None:
    """'2021-01-01' -> 20210101. None/'' -> None."""
    s = (iso or "").strip()
    if not s:
        return None
    try:
        d = date.fromisoformat(s)
    except ValueError:
        return None
    return d.year * 10000 + d.month * 100 + d.day


def date_ints_for(meta: dict) -> dict[str, int]:
    """The two integer mirrors for one chunk's metadata.

    Used by the ingest path (new chunks) and the backfill script (existing
    ones) so both produce identical values — a drift between them would show
    up as chunks silently missing from every filtered query.
    """
    eff_to_raw = str(meta.get("effective_to", "") or "").strip()
    eff_to = to_date_int(eff_to_raw)
    if eff_to is None or eff_to_raw == EFFECTIVE_TO_OPEN:
        eff_to = EFFECTIVE_TO_OPEN_INT

    eff_from = to_date_int(str(meta.get("effective_from", "") or "").strip())
    if eff_from is None:
        eff_from = EFFECTIVE_FROM_UNKNOWN

    return {"effective_from_i": eff_from, "effective_to_i": eff_to}


@dataclass(frozen=True)
class RetrievalContext:
    """Identity + time envelope carried from the API boundary to the vector store.

    Defaults are the current public behaviour: anonymous reader, public corpus,
    today. An existing call that passes no context keeps working unchanged.
    """

    tenant_id: str = PUBLIC_TENANT
    acl_labels: frozenset[str] = field(default_factory=lambda: frozenset({DEFAULT_ACL_LABEL}))
    as_of_date: str | None = None

    def __post_init__(self) -> None:
        # frozenset() so a caller passing a list still gets value semantics
        # (this object is a cache key in retriever.py — a list would be unhashable).
        if not isinstance(self.acl_labels, frozenset):
            object.__setattr__(self, "acl_labels", frozenset(self.acl_labels or {DEFAULT_ACL_LABEL}))
        if not self.acl_labels:
            object.__setattr__(self, "acl_labels", frozenset({DEFAULT_ACL_LABEL}))

    @property
    def effective_as_of(self) -> str:
        return self.as_of_date or today_iso()

    @property
    def visible_tenants(self) -> list[str]:
        """Shared corpus plus this tenant's own documents."""
        if self.tenant_id == PUBLIC_TENANT:
            return [PUBLIC_TENANT]
        return [PUBLIC_TENANT, self.tenant_id]

    def to_where(self) -> dict:
        """Chroma `where` clause. Qdrant goes through the same dict in vector_backend."""
        as_of_i = to_date_int(self.effective_as_of) or to_date_int(today_iso())
        return {
            "$and": [
                {"tenant_id": {"$in": self.visible_tenants}},
                {"acl_label": {"$in": sorted(self.acl_labels)}},
                {"effective_from_i": {"$lte": as_of_i}},
                {"effective_to_i": {"$gt": as_of_i}},
                {"status": {"$nin": sorted(_EXCLUDED_STATUS)}},
            ]
        }

    def to_llama_filters(self):
        """Same predicate as `to_where`, expressed for LlamaIndex retrievers.

        The vector store adapter translates this into the provider's own filter
        language, which is what lets the dense leg pre-filter on Qdrant too —
        `to_where()` alone would be Chroma-specific.
        """
        from llama_index.core.vector_stores import (
            FilterCondition,
            FilterOperator,
            MetadataFilter,
            MetadataFilters,
        )

        as_of_i = to_date_int(self.effective_as_of) or to_date_int(today_iso())
        return MetadataFilters(
            filters=[
                MetadataFilter(key="tenant_id", value=self.visible_tenants,
                               operator=FilterOperator.IN),
                MetadataFilter(key="acl_label", value=sorted(self.acl_labels),
                               operator=FilterOperator.IN),
                MetadataFilter(key="effective_from_i", value=as_of_i,
                               operator=FilterOperator.LTE),
                MetadataFilter(key="effective_to_i", value=as_of_i,
                               operator=FilterOperator.GT),
                MetadataFilter(key="status", value=sorted(_EXCLUDED_STATUS),
                               operator=FilterOperator.NIN),
            ],
            condition=FilterCondition.AND,
        )

    def allows(self, meta: dict) -> bool:
        """Same predicate in Python, for retrieval paths that cannot pre-filter.

        BM25 holds one in-process index over the whole corpus, so its hits are
        screened here before they leave the retriever — never after.
        """
        if str(meta.get("tenant_id", PUBLIC_TENANT)) not in self.visible_tenants:
            return False
        if str(meta.get("acl_label", DEFAULT_ACL_LABEL)) not in self.acl_labels:
            return False

        from src.rag.temporal import is_in_force

        return is_in_force(meta, self.effective_as_of)


PUBLIC_CONTEXT = RetrievalContext()


def stamp_access_meta(
    meta: dict,
    *,
    tenant_id: str = PUBLIC_TENANT,
    acl_label: str = DEFAULT_ACL_LABEL,
) -> dict:
    """Add the four filterable fields to one chunk's metadata, at write time.

    Called at the ingest boundary rather than inside the chunker on purpose:
    `chunker._version_meta` builds `effective_to` and callers then *overwrite*
    it for superseded clause versions (chunker.py, the `version["effective_to"]
    = amendment.effective_from` branch). Deriving the integer mirror any earlier
    would freeze it at the pre-mutation value, and a superseded clause carrying
    effective_to_i=99991231 would sail straight through the point-in-time
    filter — reintroducing exactly the distractors temporal filtering removes.

    Tenancy is also only knowable here: the chunker does not know who uploaded.
    """
    out = dict(meta)
    out["tenant_id"] = str(out.get("tenant_id") or tenant_id)
    out["acl_label"] = str(out.get("acl_label") or acl_label)
    out.update(date_ints_for(out))
    return out


# ── Ambient context ───────────────────────────────────────────────────────────
# Some retrieval does not happen on a call path we control: `search_legal_docs`
# in src/agent/tools.py is bound into the LLM, so the *model* decides when to
# call it and with what arguments. There is no parameter to thread a context
# through. A ContextVar set once per request is the way those paths still get
# filtered — it propagates across await points and across asyncio.to_thread,
# which is how the graph runs its sync nodes.
#
# The default is PUBLIC_CONTEXT, so a path that forgets to set it retrieves
# public data only. Defaulting to "unrestricted" would invert that: every new
# entry point would be a leak until someone remembered to close it.

from contextvars import ContextVar  # noqa: E402

_current_context: ContextVar[RetrievalContext] = ContextVar(
    "documind_retrieval_context", default=PUBLIC_CONTEXT
)


def current_context() -> RetrievalContext:
    return _current_context.get()


def set_current_context(ctx: RetrievalContext):
    """Returns the token to reset with — callers must reset in a finally block."""
    return _current_context.set(ctx or PUBLIC_CONTEXT)


def reset_current_context(token) -> None:
    _current_context.reset(token)
