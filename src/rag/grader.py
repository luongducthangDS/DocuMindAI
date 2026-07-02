"""
Relevance grading for retrieved chunks — feeds the self-correction retry loop
in src/agent/graph.py.

Two-tier grading, cheapest check first:
1. Heuristic: best chunk score well above the generator's abstain threshold
   -> short-circuit "relevant", no LLM call.
2. LLM-as-judge: score is ambiguous (near threshold) or reranker disabled
   -> ask Groq/Gemini for a Yes/No + reason. Fails open (relevant=True) on
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

_JUDGE_PROMPT = """Câu hỏi: {query}

Các đoạn văn bản tìm được:
{excerpts}

Các đoạn trên có đủ thông tin để trả lời câu hỏi không? Trả lời CHÍNH XÁC theo định dạng JSON \
một dòng, không thêm chữ nào khác: {{"relevant": true hoặc false, "reason": "lý do ngắn gọn 1 câu"}}"""


def _format_excerpts(chunks: list[RetrievedChunk], max_chunks: int = 5, max_chars: int = 300) -> str:
    parts = []
    for i, c in enumerate(chunks[:max_chunks], 1):
        text = (c.text or "")[:max_chars]
        parts.append(f"[{i}] {text}")
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

    if settings.groq_api_key:
        try:
            from src.rag.generator import _get_groq_client

            client = _get_groq_client()
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=120,
            )
            parsed = _parse_judge_response(resp.choices[0].message.content)
            if parsed is not None:
                return parsed
        except Exception as exc:
            logger.warning("Grader Groq judge failed: {}", exc)

    try:
        import google.generativeai as genai

        from src.rag.generator import _gemini_keys

        keys = _gemini_keys()
        if not keys:
            return None
        models = [m.strip() for m in settings.gemini_judge_models.split(",") if m.strip()]
        for model_name in models:
            for api_key in keys:
                try:
                    genai.configure(api_key=api_key)
                    response = genai.GenerativeModel(model_name).generate_content(prompt)
                    parsed = _parse_judge_response(response.text)
                    if parsed is not None:
                        return parsed
                except Exception as exc:
                    logger.debug("Grader Gemini judge {} failed: {}", model_name, str(exc)[:100])
                    continue
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

    judged = _call_judge_llm(query, chunks)
    if judged is None:
        logger.info("Grader LLM unavailable — failing open (relevant=True)")
        return {"relevant": True, "reason": "Không thể đánh giá — mặc định coi là liên quan"}

    return judged
