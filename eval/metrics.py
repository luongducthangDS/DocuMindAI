"""
eval/metrics.py — local evaluation metrics for DocuMind AI.

All metrics here run WITHOUT an external LLM judge:
  • answer_correctness  — cosine similarity(answer, ground_truth) via MiniLM
  • hit_rate            — fraction where top-K contains a relevant chunk
  • mrr                 — Mean Reciprocal Rank of first relevant chunk
  • ooc_refusal_rate    — for out-of-corpus questions, % system correctly declines
  • citation_rate       — % of answers that cite a source

These complement RAGAS (which requires an LLM judge) and are cheaper to run.
Separation of concerns:
  Retrieval quality  → hit_rate, mrr, context_recall (RAGAS)
  Generation quality → faithfulness (RAGAS), answer_correctness (local)
  Domain robustness  → ooc_refusal_rate, citation_rate
"""

from __future__ import annotations

import re
from typing import Optional


# ── Token-overlap helpers ─────────────────────────────────────────────────────

def _token_set(text: str) -> set[str]:
    """Vietnamese-friendly tokenisation: lowercase, split on whitespace/punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return set(text.split())


def _f1_overlap(a: str, b: str) -> float:
    """Token-level F1 between two strings (standard SQuAD-style)."""
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

    # Token-F1 (always computed)
    f1_scores = [_f1_overlap(a, g) for a, g in zip(ans_list, gt_list)]
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


# ── hit_rate & MRR ────────────────────────────────────────────────────────────

def _chunk_relevant(chunk_text: str, ground_truth: str, threshold: float = 0.15) -> bool:
    """
    A retrieved chunk is 'relevant' if its token-F1 overlap with the ground
    truth exceeds threshold.  0.15 is deliberately low — we're checking
    whether the retriever at least retrieved a chunk from the right ballpark,
    not whether it found the exact sentence.
    """
    return _f1_overlap(chunk_text, ground_truth) >= threshold


def hit_rate(
    contexts_list: list[list[str]],
    ground_truths: list[str],
    threshold: float = 0.15,
) -> float:
    """
    Fraction of questions where at least one retrieved chunk is relevant.
    Measures retriever coverage independently of the LLM.
    """
    if not contexts_list:
        return 0.0

    hits = 0
    valid = 0
    for contexts, gt in zip(contexts_list, ground_truths):
        if _is_ooc_question_answer(gt):
            continue
        valid += 1
        if any(_chunk_relevant(c, gt, threshold) for c in contexts):
            hits += 1

    return round(hits / valid, 4) if valid else 0.0


def mrr(
    contexts_list: list[list[str]],
    ground_truths: list[str],
    threshold: float = 0.15,
) -> float:
    """
    Mean Reciprocal Rank — reciprocal of the rank of the first relevant chunk.
    MRR = 1.0 means the first chunk is always relevant; 0.5 means it's usually rank 2.
    """
    if not contexts_list:
        return 0.0

    rr_scores = []
    for contexts, gt in zip(contexts_list, ground_truths):
        if _is_ooc_question_answer(gt):
            continue
        rr = 0.0
        for rank, chunk in enumerate(contexts, start=1):
            if _chunk_relevant(chunk, gt, threshold):
                rr = 1.0 / rank
                break
        rr_scores.append(rr)

    return round(sum(rr_scores) / len(rr_scores), 4) if rr_scores else 0.0


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
    contexts_list: list[list[str]],
    ground_truths: list[str],
    embedder=None,
) -> dict:
    """
    Compute all local metrics in one call.
    Returns a nested dict grouped by layer:

    {
      "retrieval": {"hit_rate": ..., "mrr": ...},
      "generation": {"answer_correctness_semantic": ..., "answer_correctness_f1": ...},
      "domain":     {"citation_rate": ..., "ooc_refusal_rate": ...},
    }
    """
    ac = answer_correctness(answers, ground_truths, embedder)
    ooc = ooc_refusal_rate(test_items, answers)

    return {
        "retrieval": {
            "hit_rate": hit_rate(contexts_list, ground_truths),
            "mrr": mrr(contexts_list, ground_truths),
        },
        "generation": {
            "answer_correctness_semantic": ac["semantic"],
            "answer_correctness_f1": ac["token_f1"],
        },
        "domain": {
            "citation_rate": citation_rate(answers),
            **({"ooc_refusal_rate": ooc} if ooc is not None else {}),
        },
    }
