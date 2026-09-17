"""
eval/metrics.py — local evaluation metrics for DocuMind AI.

All metrics here run WITHOUT an external LLM judge:
  • recall_at_k            — fraction where top-k holds the gold clause, right version
  • mrr_at_k               — Mean Reciprocal Rank of the first gold clause
  • citation_validity      — every [N] in an answer points at a source that exists
  • citation_groundedness  — at least one [N] points at a gold clause
  • answer_correctness     — cosine similarity(answer, ground_truth) + lexical overlap
  • ooc_refusal_rate       — for out-of-corpus questions, % system correctly declines
  • citation_rate          — % of answers that cite a source at all

Retrieval metrics match on `clause_uid` AND reject superseded `version_id`s — see
docs/decisions/DEC-0004-metric-theo-clause-uid.md. Same rule as
eval/temporal_eval.py::score_context, which it deliberately mirrors rather than
re-invents.

These complement RAGAS (which requires an LLM judge) and are cheaper to run.
Separation of concerns:
  Retrieval quality  → recall_at_k, mrr_at_k, context_recall (RAGAS)
  Generation quality → faithfulness (RAGAS), answer_correctness (local)
  Grounding          → citation_validity, citation_groundedness
  Domain robustness  → ooc_refusal_rate, citation_rate
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.guardrails import cited_indices  # noqa: E402, I001


# ── Token-overlap helpers ─────────────────────────────────────────────────────

def _token_set(text: str) -> set[str]:
    """Vietnamese-friendly tokenisation: lowercase, split on whitespace/punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return set(text.split())


def lexical_overlap(a: str, b: str) -> float:
    """Token-level F1 between two strings (standard SQuAD-style).

    This is a PROXY, not a correctness measure: it counts shared words and knows
    nothing about meaning, negation, or which version of a clause a text states.
    Two versions of the same article that differ only in the figure they name
    score near 1.0 against each other. Use it to compare an answer against a
    ground truth as a rough signal — never to decide whether a retrieved chunk
    is the right one (that is what recall_at_k is for), and never as a headline
    number in README or a DEC.
    """
    a_toks, b_toks = _token_set(a), _token_set(b)
    if not a_toks or not b_toks:
        return 0.0
    common = len(a_toks & b_toks)
    if common == 0:
        return 0.0
    precision = common / len(a_toks)
    recall = common / len(b_toks)
    return 2 * precision * recall / (precision + recall)


# ── answer_correctness ────────────────────────────────────────────────────────

def answer_correctness(
    answers: list[str],
    ground_truths: list[str],
    embedder=None,
) -> dict[str, float]:
    """
    Semantic similarity between generated answer and ground truth.
    Uses the same MiniLM embedder as the retriever — no extra API cost.
    Falls back to token-F1 if embedder unavailable.

    Returns:
        {"semantic": float, "token_f1": float}
        Both are mean over the non-empty (answer, ground_truth) pairs.
    """
    pairs = [
        (a, g) for a, g in zip(answers, ground_truths)
        if a and g and not _is_ooc_question_answer(g)
    ]
    if not pairs:
        return {"semantic": 0.0, "token_f1": 0.0}

    ans_list, gt_list = zip(*pairs)

    # Lexical overlap (always computed)
    f1_scores = [lexical_overlap(a, g) for a, g in zip(ans_list, gt_list)]
    token_f1 = round(sum(f1_scores) / len(f1_scores), 4)

    # Semantic cosine similarity via embedder
    semantic = 0.0
    if embedder is not None:
        try:
            import numpy as np

            ans_embs = [embedder.get_text_embedding(a) for a in ans_list]
            gt_embs = [embedder.get_text_embedding(g) for g in gt_list]

            sims = []
            for ae, ge in zip(ans_embs, gt_embs):
                ae, ge = np.array(ae), np.array(ge)
                norm = (np.linalg.norm(ae) * np.linalg.norm(ge))
                if norm > 0:
                    sims.append(float(np.dot(ae, ge) / norm))
            semantic = round(sum(sims) / len(sims), 4) if sims else 0.0
        except Exception:
            pass  # fall back to token_f1 only

    return {"semantic": semantic, "token_f1": token_f1}


# ── Gold lookup ───────────────────────────────────────────────────────────────

Record = dict[str, Any]  # {"clause_uid", "version_id", "doc_id", "text"}


def gold_clause_uids(item: dict) -> list[str]:
    """Clause ids a question must retrieve, or [] when the question has none.

    Reads the v2 gold field first, then falls back to the single `source_clause`
    of the temporal gold set. A question with neither is not scoreable here and
    the caller must skip it — never silently fall back to a text-similarity
    proxy, which would change the measuring instrument mid-run.
    """
    uids = item.get("gold_clause_uids")
    if uids:
        return [str(u) for u in uids if u]
    single = item.get("source_clause")
    return [str(single)] if single else []


def _superseded_versions(item: dict) -> set[str]:
    return {str(v) for v in (item.get("distractor_versions") or []) if v}


def _is_gold(rec: Record, gold: set[str], bad_versions: set[str]) -> bool:
    """The right clause, in a version that is still the right one."""
    return (
        str(rec.get("clause_uid", "")) in gold
        and str(rec.get("version_id", "")) not in bad_versions
    )


def _first_gold_rank(records: list[Record], item: dict, k: int) -> Optional[int]:
    """1-based rank of the first record that is the gold clause in a version
    that is still the right one, or None. Both halves matter: the right article
    in a superseded version is a miss, because that is exactly the failure the
    temporal filter exists to prevent."""
    gold = set(gold_clause_uids(item))
    if not gold:
        return None
    bad_versions = _superseded_versions(item)
    for rank, rec in enumerate(records[:k], start=1):
        if _is_gold(rec, gold, bad_versions):
            return rank
    return None


def _scoreable(items: list[dict]) -> list[int]:
    """Indices of questions that carry a gold clause id."""
    return [i for i, item in enumerate(items) if gold_clause_uids(item)]


# ── recall@k & MRR@k ──────────────────────────────────────────────────────────

def recall_at_k(
    items: list[dict],
    retrieved_list: list[list[Record]],
    k: int = 8,
) -> Optional[float]:
    """Fraction of scoreable questions whose top-k holds the gold clause.

    Returns None when no question in the set carries a gold clause id.
    """
    idxs = _scoreable(items)
    if not idxs:
        return None
    hits = sum(
        1 for i in idxs
        if _first_gold_rank(retrieved_list[i], items[i], k) is not None
    )
    return round(hits / len(idxs), 4)


def mrr_at_k(
    items: list[dict],
    retrieved_list: list[list[Record]],
    k: int = 8,
) -> Optional[float]:
    """Mean Reciprocal Rank of the gold clause within top-k (0 when absent)."""
    idxs = _scoreable(items)
    if not idxs:
        return None
    total = 0.0
    for i in idxs:
        rank = _first_gold_rank(retrieved_list[i], items[i], k)
        if rank is not None:
            total += 1.0 / rank
    return round(total / len(idxs), 4)


# ── Citation grounding ────────────────────────────────────────────────────────

def citation_validity(
    answers: list[str],
    retrieved_list: list[list[Record]],
) -> Optional[float]:
    """Fraction of citing answers where every [N] points at a source that exists.

    Only answers that cite something are counted — an answer with no citation
    is not invalid, it is a different case (see citation_rate). Returns None
    when nothing in the set cites.
    """
    scored = 0
    valid = 0
    for answer, records in zip(answers, retrieved_list):
        indices = cited_indices(answer)
        if not indices:
            continue
        scored += 1
        if all(1 <= n <= len(records) for n in indices):
            valid += 1
    return round(valid / scored, 4) if scored else None


def citation_groundedness(
    items: list[dict],
    answers: list[str],
    retrieved_list: list[list[Record]],
) -> Optional[float]:
    """Fraction of citing answers where at least one [N] points at a gold clause.

    Deliberately "at least one", not "all": a correct answer may also cite a
    supporting chunk outside the gold set, and penalising that would measure
    style rather than grounding.
    """
    scored = 0
    grounded = 0
    for item, answer, records in zip(items, answers, retrieved_list):
        gold = set(gold_clause_uids(item))
        indices = cited_indices(answer)
        if not gold or not indices:
            continue
        scored += 1
        bad_versions = _superseded_versions(item)
        for n in indices:
            if not 1 <= n <= len(records):
                continue
            if _is_gold(records[n - 1], gold, bad_versions):
                grounded += 1
                break
    return round(grounded / scored, 4) if scored else None


# ── OOC refusal rate ──────────────────────────────────────────────────────────

_REFUSAL_PATTERNS = [
    r"không\s+(?:tìm\s+thấy|có\s+thông\s+tin|có\s+dữ\s+liệu)",
    r"ngoài\s+phạm\s+vi",
    r"chưa\s+có\s+trong\s+(?:cơ\s+sở|corpus|dữ\s+liệu)",
    r"không\s+thể\s+(?:trả\s+lời|cung\s+cấp)",
    r"thông\s+tin\s+(?:này\s+)?chưa\s+có",
    r"không\s+(?:tìm|tra)\s+cứu\s+được",
    r"vượt\s+quá\s+phạm\s+vi",
    r"tài\s+liệu\s+(?:này\s+)?chưa\s+được\s+(?:cập\s+nhật|tích\s+hợp|nạp)",
    r"i\s+don.t\s+have",               # English fallback from LLM
    r"not\s+(?:found|available)\s+in",
]

_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE | re.UNICODE)


def _is_ooc_question_answer(ground_truth: str) -> bool:
    """Ground truth for OOC questions starts with 'Câu hỏi' or 'Câu hỏi này'."""
    return ground_truth.lower().startswith("câu hỏi")


def ooc_refusal_rate(
    test_items: list[dict],
    answers: list[str],
) -> Optional[float]:
    """
    For out-of-corpus questions (category='out_of_corpus' or tier-3 OOC ground truth),
    fraction where the system correctly signals it doesn't know.
    Returns None if there are no OOC questions in the test set.
    """
    ooc_pairs = [
        (item, ans)
        for item, ans in zip(test_items, answers)
        if item.get("category") == "out_of_corpus"
        or _is_ooc_question_answer(item.get("ground_truth", ""))
    ]
    if not ooc_pairs:
        return None

    refused = sum(
        1 for _, ans in ooc_pairs
        if ans and _REFUSAL_RE.search(ans)
    )
    return round(refused / len(ooc_pairs), 4)


# ── Citation rate ─────────────────────────────────────────────────────────────

_CITATION_RE = re.compile(
    r"\["
    r"(?:"
    r"\d+"                          # [1], [2]
    r"|Điều\s+\d+"                  # [Điều 48]
    r"|Khoản\s+\d+"                 # [Khoản 2]
    r"|[A-ZĐÀÁẢÃẠĂẮẶẴẲÂẤẦẨẪẬ][^\]]{2,60}"  # [Luật Doanh nghiệp 2020]
    r")"
    r"\]",
    re.UNICODE,
)


def citation_rate(answers: list[str]) -> float:
    """
    Fraction of non-empty answers that contain at least one citation marker.
    Markers: [1], [Điều 48], [Khoản 2 Điều 10], [Luật DN 2020], etc.
    """
    non_empty = [a for a in answers if a and a.strip()]
    if not non_empty:
        return 0.0
    cited = sum(1 for a in non_empty if _CITATION_RE.search(a))
    return round(cited / len(non_empty), 4)


# ── Aggregate helper ──────────────────────────────────────────────────────────

def compute_all(
    test_items: list[dict],
    answers: list[str],
    retrieved_list: list[list[Record]],
    ground_truths: list[str],
    embedder=None,
    ks: tuple[int, ...] = (1, 5, 8, 20),
) -> dict:
    """
    Compute all local metrics in one call.

    `retrieved_list` holds one list of records per question, in rank order —
    see Record. Questions without a gold clause id are skipped by the retrieval
    metrics and counted in "coverage", so a run can never look complete while
    silently scoring nothing.

    Returns a nested dict grouped by layer:

    {
      "retrieval":  {"recall@1": ..., "recall@8": ..., "mrr@8": ...},
      "coverage":   {"n_questions": ..., "n_scored": ..., "n_skipped": ...},
      "grounding":  {"citation_validity": ..., "citation_groundedness": ...},
      "generation": {"answer_correctness_semantic": ..., "answer_correctness_lexical": ...},
      "domain":     {"citation_rate": ..., "ooc_refusal_rate": ...},
    }
    """
    ac = answer_correctness(answers, ground_truths, embedder)
    ooc = ooc_refusal_rate(test_items, answers)
    n_scored = len(_scoreable(test_items))

    retrieval: dict[str, Optional[float]] = {
        f"recall@{k}": recall_at_k(test_items, retrieved_list, k) for k in ks
    }
    retrieval["mrr@8"] = mrr_at_k(test_items, retrieved_list, 8)

    return {
        "retrieval": retrieval,
        "coverage": {
            "n_questions": len(test_items),
            "n_scored": n_scored,
            "n_skipped": len(test_items) - n_scored,
        },
        "grounding": {
            "citation_validity": citation_validity(answers, retrieved_list),
            "citation_groundedness": citation_groundedness(test_items, answers, retrieved_list),
        },
        "generation": {
            "answer_correctness_semantic": ac["semantic"],
            "answer_correctness_lexical": ac["token_f1"],
        },
        "domain": {
            "citation_rate": citation_rate(answers),
            **({"ooc_refusal_rate": ooc} if ooc is not None else {}),
        },
    }
