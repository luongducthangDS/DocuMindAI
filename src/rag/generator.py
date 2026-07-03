"""
LLM generation with mandatory citations.
Primary: Groq Llama-3.3-70B | Fallback: Gemini 2.0 Flash Lite
Retry logic via tenacity; fallback logic on timeout/rate-limit.
"""

from __future__ import annotations

import asyncio
import re
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

_SYSTEM_PROMPT = """Bạn là trợ lý tra cứu quy định nội bộ của Trường Đại học Kinh tế - Kỹ thuật Công nghiệp (UNETI). Bạn CHỈ trả lời về quy định của UNETI, và CHỈ dựa trên các đoạn văn bản được cung cấp.

Quy tắc bắt buộc:
1. Mỗi câu trả lời PHẢI trích dẫn inline [số thứ tự nguồn] khi dùng thông tin từ đoạn đó.
2. Nếu câu hỏi nhắc tới một văn bản của UNETI theo số/tên (ví dụ "QĐ-853", "QĐ-740"), hãy coi các đoạn được cung cấp là nội dung của văn bản đó và trả lời theo NỘI DUNG — KHÔNG từ chối chỉ vì số/tên văn bản không lặp lại nguyên văn trong đoạn.
3. Nếu chỉ có một phần thông tin trong các đoạn, hãy trả lời phần có (kèm trích dẫn) và nêu rõ phần nào chưa có —
   đây LÀ một câu trả lời hợp lệ, KHÔNG phải trường hợp từ chối.
4. TỪ CHỐI — dùng ĐÚNG NGUYÊN VĂN VÀ CHỈ DUY NHẤT câu: "Tôi không tìm thấy quy định này trong tài liệu hiện có."
   (không thêm bất kỳ chữ nào khác trước hay sau câu này) — CHỈ trong trường hợp các đoạn được cung cấp
   HOÀN TOÀN không có nội dung liên quan:
   - Câu hỏi về một trường/tổ chức khác, KHÔNG phải UNETI.
   - Nội dung được hỏi (học phí, lịch thi, điểm chuẩn, tuyển sinh, ký túc xá... hoặc bất kỳ thông tin nào) KHÔNG xuất hiện trong các đoạn được cung cấp.
   - Câu hỏi nằm ngoài phạm vi các đoạn văn bản được cung cấp.
   Tuyệt đối KHÔNG suy đoán hay lấp bằng kiến thức bên ngoài đoạn văn bản.
   QUAN TRỌNG: nếu bạn sắp trích dẫn [N] bất kỳ nội dung nào từ các đoạn — dù chỉ một phần (áp dụng rule 3) —
   thì KHÔNG được dùng câu từ chối này ở bất kỳ đâu trong câu trả lời. Hai rule 3 và 4 loại trừ lẫn nhau:
   chọn MỘT trong hai, không ghép cả hai vào cùng một câu trả lời.
5. KHÔNG bịa đặt số liệu, điều kiện hay quy định không có trong các đoạn được cung cấp.
6. Ngôn ngữ: tiếng Việt, rõ ràng, chính xác.
7. KHÔNG liệt kê lại danh sách nguồn ở cuối — hệ thống sẽ tự động thêm.
8. BLUF (Bottom Line Up Front): câu ĐẦU TIÊN phải là câu trả lời trực tiếp (con số, Có/Không, điều kiện cốt lõi) —
   KHÔNG mở đầu bằng "Dựa trên tài liệu...", "Theo quy định...", "Dưới đây là..." hay các câu dẫn dắt không mang thông tin.
   Giải thích/điều kiện chi tiết đưa vào SAU câu trả lời trực tiếp.
9. Định dạng để dễ đọc quét (scannable):
   - In đậm (**...**) các con số, mốc thời gian, điều kiện, tên loại/mức quan trọng.
   - Dùng gạch đầu dòng (mỗi dòng bắt đầu bằng "- ") khi liệt kê từ 3 ý trở lên.
   - Khi câu hỏi yêu cầu SO SÁNH từ 2 đối tượng trở lên (ví dụ 2 loại học bổng, 2 mức điểm rèn luyện),
     trình bày bằng bảng markdown (dùng cú pháp "| Cột 1 | Cột 2 |" với dòng phân cách "|---|---|")
     thay vì viết thành đoạn văn dài."""

_CITATION_SUFFIX = "\n\n**Nguồn trích dẫn:**\n{citations}"

_GROQ_ERRORS = (
    "groq.RateLimitError",
    "groq.APITimeoutError",
    "groq.APIConnectionError",
)


_MAX_CHUNK_CHARS = 3_000   # ~750 tokens per chunk
_MAX_TOTAL_CHARS = 15_000  # ~3750 tokens — expanded to support top_k=20 candidate set
_EXTRACTIVE_CHARS_PER_SOURCE = 700

# Abstain gate calibrated for the CROSS-ENCODER reranker score (production path):
# relevant chunks score well above 0.05, OOC chunks below.
# WARNING: this scale does NOT match other retrievers. Raw RRF fusion scores are
# ~1/(60+rank) ≈ 0.016 (always < 0.05 → would abstain on everything); raw BM25
# scores are on yet another scale. Callers using a non-reranked retriever MUST
# pass an appropriate `min_score` (e.g. 0.0 to disable the gate).
_MIN_RELEVANCE_SCORE = 0.05


def _effective_min_score() -> float:
    """0.05 when the cross-encoder reranker is active (scores are on that scale);
    0.0 when it's disabled (e.g. Render free tier) or unavailable, per the
    calibration warning above — raw RRF/BM25 scores never clear 0.05, which
    would abstain on every query.

    Checks the retriever's actual runtime state (src.rag.retriever._reranker_active),
    not just the settings.enable_reranker config flag — the reranker can be
    *requested* but still fail to load (missing model cache, OOM, etc.), in
    which case chunk scores stay on the raw RRF scale even though the config
    says reranking is on. Trusting the config alone caused every chunk to be
    filtered out and the generator to abstain even when retrieval and grading
    both found relevant content.
    """
    if not get_settings().enable_reranker:
        return 0.0
    import src.rag.retriever as r_module
    return _MIN_RELEVANCE_SCORE if getattr(r_module, "_reranker_active", False) else 0.0


# Matches both "[1]" and combined "[1, 2]" / "[1,2,3]" citation styles —
# the LLM isn't consistent about which format it uses.
_CITATION_BRACKET_RE = re.compile(r"\[([\d,\s]+)\]")


def _cited_sources(answer: str, chunks: list[RetrievedChunk]) -> list[dict]:
    """Only return sources the answer actually cites via [N] markers.

    Without this, an LLM that correctly declines ("không tìm thấy quy định
    này...") in its own words — rather than via the hardcoded abstain
    message — still had all retrieved chunks attached as "sources", making
    an uncited refusal look like a grounded, cited answer.
    """
    cited_indices: set[int] = set()
    for bracket in _CITATION_BRACKET_RE.findall(answer or ""):
        for piece in bracket.split(","):
            piece = piece.strip()
            if piece.isdigit():
                cited_indices.add(int(piece))
    return [
        {"index": i + 1, **c.metadata, "score": c.score}
        for i, c in enumerate(chunks)
        if (i + 1) in cited_indices
    ]


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


def _gemini_keys() -> list[str]:
    s = get_settings()
    return [k for k in (s.google_api_key, s.google_api_key_2, s.google_api_key_3) if k]


def _gemini_pairs() -> list[tuple[str, str]]:
    """(api_key, model) pairs, model-major: try all 3 keys for a model before
    moving on. Spreads generation load across keys + models so we don't exhaust
    one key's daily RPD (the old bug — generation only ever hit key #1)."""
    s = get_settings()
    models = [m.strip() for m in s.gemini_generation_models.split(",") if m.strip()]
    keys = _gemini_keys()
    return [(k, m) for m in models for k in keys]


# Round-robin cursor so consecutive generations start at DIFFERENT (key, model)
# pairs — spreads requests across pairs to stay under each pair's RPM limit,
# instead of hammering pair[0] every call and tripping 429s.
_GEMINI_PAIR_CURSOR = 0


def _call_groq(prompt: str, context: str, history: list[dict] | None = None) -> str | None:
    client = _get_groq_client()
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"**Văn bản tham chiếu:**\n{context}\n\n**Câu hỏi:** {prompt}"})
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.0,
            max_tokens=1536,
        )
        return response.choices[0].message.content
    except Exception as exc:
        logger.error("Groq API error (type={}, detail={})", type(exc).__name__, str(exc)[:300])
        raise


def _call_gemini(prompt: str, context: str, history: list[dict] | None = None) -> str | None:
    """Rotate across (key, model) pairs until one succeeds — spreads load over all
    3 API keys and the configured models to dodge per-key/per-model daily limits."""
    import google.generativeai as genai

    history_block = ""
    if history:
        lines = [f"{'Người dùng' if m['role'] == 'user' else 'Trợ lý'}: {m['content'][:500]}"
                 for m in history]
        history_block = "\n**Lịch sử hội thoại:**\n" + "\n".join(lines) + "\n\n"
    full_prompt = (
        f"{_SYSTEM_PROMPT}\n\n{history_block}**Văn bản tham chiếu:**\n{context}\n\n**Câu hỏi:** {prompt}"
    )

    pairs = _gemini_pairs()
    if not pairs:
        raise RuntimeError("No GOOGLE_API_KEY configured for Gemini generation")

    global _GEMINI_PAIR_CURSOR
    n = len(pairs)
    start = _GEMINI_PAIR_CURSOR
    last_exc: Exception | None = None
    for offset in range(n):
        api_key, model_name = pairs[(start + offset) % n]
        try:
            genai.configure(api_key=api_key)
            response = genai.GenerativeModel(model_name).generate_content(full_prompt)
            if response.text:
                # Advance cursor so the NEXT call starts at the following pair —
                # round-robin keeps any single (key, model) under its RPM limit.
                _GEMINI_PAIR_CURSOR = (start + offset + 1) % n
                return response.text
        except Exception as exc:
            err = str(exc).lower()
            if any(t in err for t in ("quota", "429", "not found", "exhaust", "rate")):
                logger.debug("Gemini {} (key…{}) unavailable, rotating: {}",
                             model_name, api_key[-4:], str(exc)[:100])
                last_exc = exc
                continue
            raise  # non-quota error — propagate immediately
    raise last_exc  # all pairs exhausted


@_traceable(
    name="rag-generate-answer",
    run_type="llm",
    tags=["groq", "gemini", "legal-qa", "citations"],
)
def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    use_fallback: bool = False,
    history: list[dict] | None = None,
    min_score: float | None = None,
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

    # Filter out chunks the reranker scored as irrelevant before calling LLM.
    # Without this, a "hoàn thuế GTGT" query returns forest/labour law chunks
    # (score≈0.01) and the LLM correctly abstains — but we still showed 8 wrong sources.
    if min_score is None:
        min_score = _effective_min_score()
    relevant_chunks = [c for c in chunks if c.score >= min_score]
    if not relevant_chunks:
        logger.info(
            "All {} chunks below relevance threshold ({:.3f}) — abstaining without LLM call",
            len(chunks), min_score,
        )
        return {
            "answer": (
                "Tôi không tìm thấy văn bản pháp luật liên quan đến câu hỏi này trong cơ sở dữ liệu hiện có.\n\n"
                "Gợi ý: câu hỏi của bạn có thể thuộc lĩnh vực chưa được tích hợp vào hệ thống. "
                "Bạn có thể tải thêm văn bản pháp luật liên quan qua tab **Tải lên văn bản**."
            ),
            "sources": [],
            "used_llm": "none",
            "chunk_count": 0,
        }
    chunks = relevant_chunks

    context, citation_list = _build_context(chunks)
    prefer_gemini = get_settings().generator_provider.lower() == "gemini"
    used_llm = "gemini" if prefer_gemini else "groq"
    answer = None

    # Skip Groq entirely when provider=gemini (e.g. Groq daily TPD exhausted).
    if not use_fallback and not prefer_gemini:
        try:
            answer = _call_groq(query, context, history=history)
            logger.info("Groq answered query ({} chars)", len(answer or ""))
        except Exception as exc:
            logger.warning("Groq failed, switching to Gemini: {}", exc)
            used_llm = "gemini_fallback"

    if answer is None:
        try:
            answer = _call_gemini(query, context, history=history)
            used_llm = "gemini"
            logger.info("Gemini answered query ({} chars)", len(answer or ""))
        except Exception as exc:
            logger.error("Both LLMs failed: {}", exc)
            answer = _build_extractive_answer(query, chunks)
            used_llm = "extractive_fallback"

    return {
        "answer": answer,
        "sources": _cited_sources(answer, chunks),
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

    # Mirror generate_answer: filter irrelevant chunks before calling LLM
    min_score = _effective_min_score()
    relevant_chunks = [c for c in chunks if c.score >= min_score]
    if not relevant_chunks:
        logger.info(
            "stream_answer: all {} chunks below threshold ({:.2f}) — abstaining",
            len(chunks), min_score,
        )
        yield (
            "Tôi không tìm thấy văn bản pháp luật liên quan đến câu hỏi này trong cơ sở dữ liệu hiện có.\n\n"
            "Gợi ý: câu hỏi của bạn có thể thuộc lĩnh vực chưa được tích hợp vào hệ thống. "
            "Bạn có thể tải thêm văn bản pháp luật liên quan qua tab **Tải lên văn bản**."
        )
        return
    chunks = relevant_chunks

    context, citation_list = _build_context(chunks)
    settings = get_settings()

    if not settings.groq_api_key:
        yield _build_extractive_answer(query, chunks)
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
            temperature=0.0,  # match non-streaming path for consistent output
            max_tokens=1536,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
                await asyncio.sleep(0)  # yield control to event loop

    except Exception as exc:
        logger.error("Streaming failed: {}", exc)
        # Fallback to non-streaming Gemini
        result = generate_answer(query, chunks, use_fallback=True)
        yield result["answer"]
