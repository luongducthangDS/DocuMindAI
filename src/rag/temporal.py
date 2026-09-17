"""
Temporal-aware retrieval: keep only the clause versions in force at a given date.

This is the layer *above* retrieval — `src/rag/retriever.py` (fusion, rerank)
is never touched. The graph calls `versions_in_force()` on the chunks that came
back from retrieval, so a query with `as_of_date=2025-03-01` sees the minimum
wage of Nghị định 74/2024 while the same query at 2026-02-01 sees Nghị định
293/2025.

Naming: this is *temporal-aware retrieval with clause-level effective dates*,
not "point-in-time reconstruction" — only 4 clauses of the corpus were ever
amended in place (see corpus_manifest.yaml, `sua_doi_in_place`).

Rule for "the version that applies at T" (CORPUS_SPEC §2.3):

    effective_from <= T
    AND (effective_to empty OR effective_to > T)
    AND status != 'repealed' at T
"""

from __future__ import annotations

from datetime import date
from typing import Any

from loguru import logger

from src.ingestion.manifest import EFFECTIVE_TO_OPEN

# Clause lifecycle values that must never be shown as current law.
_EXCLUDED_STATUS = frozenset({"repealed"})


def today_iso() -> str:
    return date.today().isoformat()


def _meta(chunk: Any) -> dict:
    """Metadata of a RetrievedChunk, a LlamaIndex node, or a plain dict."""
    meta = getattr(chunk, "metadata", None)
    if meta is None and isinstance(chunk, dict):
        meta = chunk.get("metadata", chunk)
    return meta or {}


def is_in_force(meta: dict, as_of: str) -> bool:
    """Whether one clause version applies at `as_of` (ISO date)."""
    if str(meta.get("status", "")).strip().lower() in _EXCLUDED_STATUS:
        return False

    eff_from = str(meta.get("effective_from", "") or "").strip()
    eff_to = str(meta.get("effective_to", "") or "").strip()

    # A chunk with no effective date is not datable — keep it rather than
    # silently dropping content (e.g. a preamble or an annex without dates).
    if eff_from and as_of < eff_from:
        return False
    if eff_to and eff_to != EFFECTIVE_TO_OPEN and as_of >= eff_to:
        return False
    return True


def versions_in_force(chunks: list, as_of: str) -> list:
    """Filter retrieved chunks down to the versions in force at `as_of`.

    Order is preserved (retrieval ranking still decides what comes first). When
    several versions of the *same* clause survive the date filter — which means
    the corpus has overlapping ranges and is wrong somewhere — the one with the
    latest `effective_from` wins and the overlap is logged.
    """
    if not as_of:
        as_of = today_iso()

    kept = [c for c in chunks if is_in_force(_meta(c), as_of)]

    # Tie-break per clause_uid. Chunks without a clause_uid (older documents,
    # fallback chunks) are never deduplicated against each other.
    best_by_clause: dict[str, Any] = {}
    for chunk in kept:
        uid = str(_meta(chunk).get("clause_uid", "") or "")
        if not uid:
            continue
        current = best_by_clause.get(uid)
        if current is None:
            best_by_clause[uid] = chunk
            continue
        new_from = str(_meta(chunk).get("effective_from", "") or "")
        cur_from = str(_meta(current).get("effective_from", "") or "")
        # Same version retrieved twice is just a duplicate chunk; two *different*
        # versions in force at the same date means the corpus dates overlap.
        if _meta(chunk).get("version_id") != _meta(current).get("version_id"):
            logger.warning(
                "Overlapping versions in force at {} for clause {} ({} vs {}) — keeping later",
                as_of,
                uid,
                cur_from or "?",
                new_from or "?",
            )
        if new_from > cur_from:
            best_by_clause[uid] = chunk

    dropped_uids = set()
    out = []
    for chunk in kept:
        uid = str(_meta(chunk).get("clause_uid", "") or "")
        if uid and best_by_clause.get(uid) is not chunk:
            dropped_uids.add(uid)
            continue
        out.append(chunk)

    if len(out) != len(chunks):
        logger.debug(
            "Temporal filter @{}: {} → {} chunks ({} out of force, {} superseded duplicates)",
            as_of,
            len(chunks),
            len(out),
            len(chunks) - len(kept),
            len(dropped_uids),
        )
    return out


def is_out_of_range(as_of: str, earliest: str) -> bool:
    """True when `as_of` predates what the corpus can answer for.

    The answer must then say the corpus only covers from `earliest` instead of
    guessing at law that was never indexed.
    """
    if not as_of or not earliest:
        return False
    return as_of < earliest
