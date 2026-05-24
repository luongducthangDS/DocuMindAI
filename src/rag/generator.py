"""
LLM generation with mandatory citations.
Primary: Groq Llama-3.3-70B | Fallback: Gemini 2.0 Flash Lite
Retry logic via tenacity; fallback logic on timeout/rate-limit.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_settings
from src.rag.retriever import RetrievedChunk

# LangSmith tracing — optional
try:
    from langsmith import traceable as _traceable  # type: ignore
except ImportError:
    def _traceable(**kwargs):
        def decorator(fn):
            return fn
        return decorator

_SYSTEM_PROMPT = """Bạn là trợ lý pháp lý chuyên về luật Việt Nam.
Chỉ trả lời dựa trên các đoạn văn bản pháp luật được cung cấp.
Quy tắc bắt buộc:
1. Mỗi câu trả lời PHẢI trích dẫn inline [số thứ tự nguồn] khi dùng thông tin từ đoạn đó.
2. Nếu không có thông tin trong văn bản, nói rõ: "Tôi không tìm thấy quy định này trong tài liệu hiện có."
3. KHÔNG bịa đặt hoặc suy luận ngoài văn bản được cung cấp.
4. Ngôn ngữ: tiếng Việt, rõ ràng, chính xác.
5. KHÔNG liệt kê lại danh sách nguồn ở cuối — hệ thống sẽ tự động thêm."""

_CITATION_SUFFIX = "\n\n**Nguồn trích dẫn:**\n{citations}"

_GROQ_ERRORS = (
    "groq.RateLimitError",
    "groq.APITimeoutError",
    "groq.APIConnectionError",
)


_MAX_CHUNK_CHARS = 3_000   # ~750 tokens per chunk
_MAX_TOTAL_CHARS = 12_000  # ~3000 tokens total context
_EXTRACTIVE_CHARS_PER_SOURCE = 700


def _build_context(chunks: list[RetrievedChunk]) -> tuple[str, str]:
    """Returns (context_block, citation_list). Truncates to stay within LLM limits."""
    context_parts = []
    citations = []
    total_chars = 0

    for i, chunk in enumerate(chunks, 1):
        text = chunk.text
        if len(text) > _MAX_CHUNK_CHARS:
            text = text[:_MAX_CHUNK_CHARS] + "…"
        if total_chars + len(text) > _MAX_TOTAL_CHARS:
            break
        context_parts.append(f"[{i}] {text}")
        citations.append(f"[{i}] {chunk.citation_label}")
        total_chars += len(text)

    return "\n\n---\n\n".join(context_parts), "\n".join(citations)


def _build_extractive_answer(query: str, chunks: list[RetrievedChunk]) -> str:
    """Return a useful answer from retrieved sources when LLM providers fail."""
    if not chunks:
        return "Tôi không tìm thấy văn bản pháp luật liên quan đến câu hỏi này."

    lines = [
        "Tôi đã tìm thấy các quy định liên quan trong dữ liệu hiện có, nhưng dịch vụ LLM đang tạm thời không phản hồi. Dưới đây là phần trích xuất trực tiếp từ nguồn để bạn vẫn có thể tham khảo:",
        "",
    ]

    for i, chunk in enumerate(chunks[:5], 1):
        title = chunk.metadata.get("title") or "Văn bản pháp luật"
        dieu = chunk.metadata.get("dieu_header") or ""
        text = " ".join((chunk.text or "").split())
        if len(text) > _EXTRACTIVE_CHARS_PER_SOURCE:
            text = text[:_EXTRACTIVE_CHARS_PER_SOURCE].rstrip() + "..."

        heading = f"**[{i}] {title}**"
        if dieu:
            heading += f" - {dieu}"
        lines.append(heading)
        lines.append(text or "Không có nội dung trích xuất.")
        lines.append("")

    return "\n".join(lines).strip()


def _get_groq_client():
    from groq import Groq

    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    return Groq(api_key=settings.groq_api_key)


def _get_gemini_client():
    import google.generativeai as genai

    settings = get_settings()
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    genai.configure(api_key=settings.google_api_key)
    return genai.GenerativeModel("gemini-2.5-flash-lite")


def _call_groq(prompt: str, context: str) -> str | None:
    client = _get_groq_client()
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"**Văn bản tham chiếu:**\n{context}\n\n**Câu hỏi:** {prompt}"},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as exc:
        logger.error("Groq API error (type={}, detail={})", type(exc).__name__, str(exc)[:300])
        raise


def _call_gemini(prompt: str, context: str) -> str | None:
    model = _get_gemini_client()
    full_prompt = (
        f"{_SYSTEM_PROMPT}\n\n**Văn bản tham chiếu:**\n{context}\n\n**Câu hỏi:** {prompt}"
    )
    response = model.generate_content(full_prompt)
    return response.text


@_traceable(
    name="rag-generate-answer",
    run_type="llm",
    tags=["groq", "gemini", "legal-qa", "citations"],
)
def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    use_fallback: bool = False,
) -> dict:
    """
    Generate answer with citations.
    Returns: {answer, sources, used_llm, chunk_count}

    Decorated with @traceable: each call appears in LangSmith as a child span
    of the parent 'documind-agent' run, showing the prompt, LLM response, and
    which provider was used (Groq primary / Gemini fallback).
    """
    if not chunks:
        return {
            "answer": "Tôi không tìm thấy văn bản pháp luật liên quan đến câu hỏi này.",
            "sources": [],
            "used_llm": "none",
            "chunk_count": 0,
        }

    context, citation_list = _build_context(chunks)
    used_llm = "groq"
    answer = None

    if not use_fallback:
        try:
            answer = _call_groq(query, context)
            logger.info("Groq answered query ({} chars)", len(answer or ""))
        except Exception as exc:
            logger.warning("Groq failed, switching to Gemini: {}", exc)
            used_llm = "gemini_fallback"

    if answer is None:
        try:
            answer = _call_gemini(query, context)
            used_llm = "gemini"
            logger.info("Gemini answered query ({} chars)", len(answer or ""))
        except Exception as exc:
            logger.error("Both LLMs failed: {}", exc)
            answer = _build_extractive_answer(query, chunks)
            used_llm = "extractive_fallback"

    # Append citation list to answer
    full_answer = answer + _CITATION_SUFFIX.format(citations=citation_list)

    return {
        "answer": full_answer,
        "sources": [{"index": i + 1, **c.metadata, "score": c.score} for i, c in enumerate(chunks)],
        "used_llm": used_llm,
        "chunk_count": len(chunks),
    }


async def stream_answer(
    query: str,
    chunks: list[RetrievedChunk],
) -> AsyncIterator[str]:
    """
    Stream tokens from Groq (primary) with Gemini fallback.
    Yields text chunks for WebSocket/SSE streaming.
    """
    if not chunks:
        yield "Tôi không tìm thấy văn bản pháp luật liên quan đến câu hỏi này."
        return

    context, citation_list = _build_context(chunks)
    settings = get_settings()

    if not settings.groq_api_key:
        yield _build_extractive_answer(query, chunks)
        yield _CITATION_SUFFIX.format(citations=citation_list)
        return

    try:
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"**Văn bản tham chiếu:**\n{context}\n\n**Câu hỏi:** {query}"},
            ],
            temperature=0.1,
            max_tokens=1024,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
                await asyncio.sleep(0)  # yield control to event loop

        yield _CITATION_SUFFIX.format(citations=citation_list)

    except Exception as exc:
        logger.error("Streaming failed: {}", exc)
        # Fallback to non-streaming Gemini
        result = generate_answer(query, chunks, use_fallback=True)
        yield result["answer"]
