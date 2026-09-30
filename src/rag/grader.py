"""
Relevance grading for retrieved chunks — feeds the self-correction retry loop
in src/agent/graph.py.

Two-tier grading, cheapest check first:
1. Heuristic: best chunk score well above the generator's abstain threshold
   -> short-circuit "relevant", no LLM call.
2. LLM-as-judge: score is ambiguous (near threshold) or reranker disabled
   -> ask Gemini for a Yes/No + reason. Fails open (relevant=True) on
   any LLM error so a grader outage never blocks answering.
"""

from __future__ import annotations

import json
import re

from loguru import logger

from src.config import get_settings
from src.rag.generator import _effective_min_score
from src.rag.retriever import RetrievedChunk

# Above this multiple of the abstain threshold, skip the LLM judge entirely —
# the reranker/heuristic score is already unambiguous.
_CONFIDENT_SCORE_MULTIPLIER = 2.0

# Không có reranker (Render, và máy dev từ 09-23) thì điểm chunk là RRF ~0.01–0.03,
# không nói được gì tuyệt đối — trước đây vì thế MỌI câu đều hỏi LLM judge. Cosine
# của nhánh dense thì có thang tuyệt đối. Ngưỡng đo trên legal_qa_200, gemini-embedding-001
# (reports/agent_budget_*.json): cosine của model này dồn trong khoảng hẹp — gold lọt
# top-8 có p10 0.755 / p50 0.790 / min 0.688, câu ngoài phạm vi p50 0.737 — nên 0.82/0.45
# kiểu "trông hợp lý" gần như không bao giờ khớp (0.82: 13/199 câu).
# CONFIDENT = ngay dưới p10 của nhóm có gold. Baseline cho thấy judge LLM chấm
# "irrelevant" 11/14 câu mà gold đứng top-8 (đa số hạng 1) — tức ở vùng cao nó chỉ
# thêm reformulate + grade + embed, không cứu được câu nào.
# HOPELESS = dưới mọi câu có gold; thực tế chỉ còn các lượt truy hồi rỗng/lệch hẳn.
# Đổi EMBEDDING_MODEL là phải đo lại (eval/agent_budget_eval.py).
CONFIDENT_COSINE = 0.75
HOPELESS_COSINE = 0.60

_JUDGE_PROMPT = """Câu hỏi: {query}

Các đoạn văn bản tìm được:
{excerpts}

Các đoạn trên có đủ thông tin để trả lời câu hỏi không? Trả lời CHÍNH XÁC theo định dạng JSON \
một dòng, không thêm chữ nào khác: {{"relevant": true hoặc false, "reason": "lý do ngắn gọn 1 câu"}}"""


_MAX_JUDGE_CHUNKS = 12
_MAX_JUDGE_CHARS_PER_CHUNK = 250
_MAX_JUDGE_TOTAL_CHARS = 2500


def _format_excerpts(
    chunks: list[RetrievedChunk],
    max_chunks: int = _MAX_JUDGE_CHUNKS,
    max_chars: int = _MAX_JUDGE_CHARS_PER_CHUNK,
) -> str:
    """Sample across a total character budget rather than a fixed chunk count.
    When the reranker is off (or fails to load), the retrieved pool isn't
    sorted by true relevance — the answer-bearing chunk can rank outside a
    small fixed window, making the judge blind to it. Mirrors the budget
    style of generator._build_context so the judge sees roughly what the
    generator would."""
    parts = []
    total_chars = 0
    for i, c in enumerate(chunks[:max_chunks], 1):
        text = (c.text or "")[:max_chars]
        if total_chars + len(text) > _MAX_JUDGE_TOTAL_CHARS:
            break
        parts.append(f"[{i}] {text}")
        total_chars += len(text)
    return "\n\n".join(parts)


def _parse_judge_response(raw: str) -> dict | None:
    if not raw:
        return None
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return {"relevant": bool(data.get("relevant")), "reason": str(data.get("reason", ""))[:200]}
    except (json.JSONDecodeError, TypeError):
        return None


def _call_judge_llm(query: str, chunks: list[RetrievedChunk]) -> dict | None:
    """Ask an LLM whether the chunks are relevant. Returns None on any failure
    (caller fails open)."""
    excerpts = _format_excerpts(chunks)
    prompt = _JUDGE_PROMPT.format(query=query, excerpts=excerpts)
    settings = get_settings()

    try:
        from src.rag.generator import gemini_generate

        models = [m.strip() for m in settings.gemini_judge_models.split(",") if m.strip()]
        parsed = _parse_judge_response(gemini_generate(prompt, models=models, name="grade-chunks"))
        if parsed is not None:
            return parsed
    except Exception as exc:
        logger.warning("Grader Gemini judge unavailable: {}", exc)

    return None


def grade_chunks(query: str, chunks: list[RetrievedChunk]) -> dict:
    """Returns {"relevant": bool, "reason": str}."""
    if not chunks:
        return {"relevant": False, "reason": "Không tìm thấy đoạn văn bản nào"}

    threshold = _effective_min_score()
    best_score = max(c.score for c in chunks)

    # threshold == 0 means the reranker is disabled (e.g. Render free tier) and
    # the score scale is unreliable — see generator._effective_min_score's own
    # calibration warning. `best_score >= 0 * multiplier` would then be true for
    # ANY positive score, short-circuiting every query as "relevant" and making
    # this whole grading step a no-op. Only take the cheap shortcut when the
    # threshold is actually meaningful; otherwise always defer to the LLM judge.
    if threshold > 0 and best_score >= threshold * _CONFIDENT_SCORE_MULTIPLIER:
        return {"relevant": True, "reason": f"Điểm liên quan cao ({best_score:.3f})"}

    if threshold == 0:
        dense = [c.dense_score for c in chunks if c.dense_score is not None]
        best_cos = max(dense) if dense else None
        if best_cos is not None and best_cos >= CONFIDENT_COSINE:
            return {"relevant": True, "reason": f"Độ tương đồng cao (cos {best_cos:.3f})"}
        if best_cos is not None and best_cos < HOPELESS_COSINE:
            return {"relevant": False, "reason": f"Độ tương đồng quá thấp (cos {best_cos:.3f})"}

    judged = _call_judge_llm(query, chunks)
    if judged is None:
        logger.info("Grader LLM unavailable — failing open (relevant=True)")
        return {"relevant": True, "reason": "Không thể đánh giá — mặc định coi là liên quan"}

    return judged
