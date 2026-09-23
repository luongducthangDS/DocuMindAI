"""
LLM generation with mandatory citations.
Nhà cung cấp duy nhất: Gemini (xoay vòng key × model, xem _call_gemini).
"""

from __future__ import annotations

import asyncio
import queue
import re
import threading
import time
from datetime import datetime, timezone
from typing import AsyncIterator

from loguru import logger

from src.config import DOMAIN_NAME, DOMAIN_SCOPE, get_settings
from src.langfuse_otel import record_generation
from src.rag.retriever import RetrievedChunk

# LangSmith tracing — optional
try:
    from langsmith import traceable as _traceable  # type: ignore
except ImportError:
    def _traceable(**kwargs):
        def decorator(fn):
            return fn
        return decorator

_SYSTEM_PROMPT = f"""Bạn là trợ lý tra cứu {DOMAIN_NAME} ({DOMAIN_SCOPE}). Bạn CHỈ trả lời dựa trên các đoạn văn bản được cung cấp.

Quy tắc bắt buộc:
1. Mỗi câu trả lời PHẢI trích dẫn inline [số thứ tự nguồn] khi dùng thông tin từ đoạn đó.
2. Nếu câu hỏi nhắc tới một văn bản theo số/tên (ví dụ "45/2019/QH14", "NĐ 145/2020"), hãy coi các đoạn được cung cấp là nội dung của văn bản đó và trả lời theo NỘI DUNG — KHÔNG từ chối chỉ vì số/tên văn bản không lặp lại nguyên văn trong đoạn.
3. Nếu chỉ có một phần thông tin trong các đoạn, hãy trả lời phần có (kèm trích dẫn) và nêu rõ phần nào chưa có —
   đây LÀ một câu trả lời hợp lệ, KHÔNG phải trường hợp từ chối.
4. TỪ CHỐI — dùng ĐÚNG NGUYÊN VĂN VÀ CHỈ DUY NHẤT câu: "Tôi không tìm thấy quy định này trong tài liệu hiện có."
   (không thêm bất kỳ chữ nào khác trước hay sau câu này) — CHỈ trong trường hợp các đoạn được cung cấp
   HOÀN TOÀN không có nội dung liên quan:
   - Nội dung được hỏi (mức lương, số ngày nghỉ, thời gian đóng BHXH, điều kiện hưởng chế độ... hoặc bất kỳ thông tin nào) KHÔNG xuất hiện trong các đoạn được cung cấp.
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
   - Khi câu hỏi yêu cầu SO SÁNH từ 2 đối tượng trở lên (ví dụ 2 loại hợp đồng, 2 mức trợ cấp),
     trình bày bằng bảng markdown (dùng cú pháp "| Cột 1 | Cột 2 |" với dòng phân cách "|---|---|")
     thay vì viết thành đoạn văn dài."""

_CITATION_SUFFIX = "\n\n**Nguồn trích dẫn:**\n{citations}"


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

# --- Routing theo độ khó câu hỏi -------------------------------------------
# ponytail: heuristic rẻ tiền, không phải classifier ML — nâng cấp lên model/
# nhãn thật khi có đủ traffic để đo lệch. Dùng CHUNG cho cả việc chọn model
# tier (RPD thấp hơn nhưng mạnh hơn flash-lite) và số chunk đưa vào context:
# câu "khó" giữ nguyên top_n=8 của retriever để không mất nguồn cần đối chiếu,
# câu thường cắt bớt để giảm token input.
_COMPLEX_MIN_QUERY_CHARS = 150
_COMPLEX_KEYWORDS = ("so sánh", "khác nhau")
_COMPLEX_MIN_HISTORY_TURNS = 4
_DOC_REF_RE = re.compile(r"\d{1,3}[-/]\d{4}")
_SIMPLE_QUERY_MAX_CHUNKS = 5


def _is_complex_query(query: str, history: list[dict] | None = None) -> bool:
    """Câu hỏi dài, nhắc >=2 văn bản, có từ so sánh, hoặc hội thoại đã đi vài
    lượt => coi là "khó"."""
    if history and len(history) >= _COMPLEX_MIN_HISTORY_TURNS:
        return True
    if len(query) >= _COMPLEX_MIN_QUERY_CHARS:
        return True
    if len(set(_DOC_REF_RE.findall(query))) >= 2:
        return True
    ql = query.lower()
    return any(kw in ql for kw in _COMPLEX_KEYWORDS)


def _select_models(complex_query: bool) -> list[str] | None:
    """None = dùng danh sách mặc định (gemini_generation_models) của
    _gemini_pairs. Chỉ chuyển sang tier "khó" khi câu hỏi bị coi là phức tạp
    VÀ tier đó thực sự có cấu hình (không rỗng)."""
    if not complex_query:
        return None
    s = get_settings()
    models = [m.strip() for m in s.gemini_generation_models_complex.split(",") if m.strip()]
    return models or None


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


def _as_of_block(as_of_date: str | None) -> str:
    """Prompt preamble stating which date the answer must speak for.

    The chunks were already filtered by `versions_in_force`, so the model is not
    asked to reason about dates — only to name the date it is answering for and
    to use the figures of that version (minimum wage, contribution rates...).
    """
    if not as_of_date:
        return ""
    return (
        f"**Thời điểm tra cứu:** {as_of_date}\n"
        "Các đoạn dưới đây đã được lọc theo hiệu lực tại ngày này. Khi nêu số liệu "
        "(mức lương tối thiểu, tỷ lệ đóng, thời gian nghỉ...), dùng đúng con số của bản có "
        f"hiệu lực tại {as_of_date} và nêu rõ mốc thời điểm này trong câu trả lời.\n\n"
    )


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


def _gemini_keys() -> list[str]:
    s = get_settings()
    return [k for k in (s.google_api_key, s.google_api_key_2, s.google_api_key_3) if k]


def _gemini_pairs(models: list[str] | None = None) -> list[tuple[str, str]]:
    """(api_key, model) pairs, model-major: try all 3 keys for a model before
    moving on. Spreads generation load across keys + models so we don't exhaust
    one key's daily RPD (the old bug — generation only ever hit key #1).
    Bỏ qua cặp đang trong cooldown (xem _cool_down)."""
    s = get_settings()
    if models is None:
        models = [m.strip() for m in s.gemini_generation_models.split(",") if m.strip()]
    keys = _gemini_keys()
    return [(k, m) for m in models for k in keys if not _is_down(k, m)]


# SDK google-generativeai mặc định TỰ retry 503 tới 600s (và timeout 600s) trên
# CÙNG một cặp — Gemini quá tải là 1 lời gọi treo 1–6 phút, vòng xoay (key, model)
# không bao giờ được chạy. Đo thật trên Render 2026-09-23: query >3 phút không về.
# Tắt retry ngầm. Call nhỏ (thêm dấu, router, grade: 2–7s khi khoẻ) timeout ngắn;
# riêng câu trả lời chính cần đủ cho văn bản dài (stream: deadline cả luồng).
_GEMINI_TIMEOUT_S = 20
_GEMINI_ANSWER_TIMEOUT_S = 45
# Cặp vừa lỗi tạm thời bị bỏ qua một lúc: 1 câu hỏi gọi Gemini 3–5 lần (dấu,
# router, grade, generate...), không thì lần nào cũng chờ lại đúng cặp đang treo.
# ponytail: cooldown cố định, không phân biệt 429 phút/ngày — đủ để chặn treo.
_PAIR_COOLDOWN_S = 120
_PAIR_DOWN_UNTIL: dict[tuple[str, str], float] = {}
_MODEL_WIDE_MARKERS = ("503", "504", "unavailable", "deadline", "timed out", "timeout")
_TRANSIENT_MARKERS = ("quota", "429", "not found", "exhaust", "rate") + _MODEL_WIDE_MARKERS


def _is_transient(exc: Exception) -> bool:
    err = str(exc).lower()
    return any(t in err for t in _TRANSIENT_MARKERS)


def _is_down(api_key: str, model_name: str) -> bool:
    return _PAIR_DOWN_UNTIL.get((api_key, model_name), 0) > time.monotonic()


def _cool_down(api_key: str, model_name: str, exc: Exception) -> None:
    """429 = quota của riêng key đó. 503/504/timeout = model quá tải, key nào cũng
    như nhau (đo 2026-09-23) → cooldown cả model, khỏi chờ timeout lần lượt 3 key.
    Không gia hạn cặp đang cooldown — traffic liên tục sẽ giữ nó down mãi."""
    err = str(exc).lower()
    keys = _gemini_keys() if any(t in err for t in _MODEL_WIDE_MARKERS) else [api_key]
    until = time.monotonic() + _PAIR_COOLDOWN_S
    for k in keys:
        if not _is_down(k, model_name):
            _PAIR_DOWN_UNTIL[(k, model_name)] = until


def gemini_call(api_key: str, model_name: str, prompt: str,
                timeout: float = _GEMINI_TIMEOUT_S, **kwargs):
    """Lời gọi generate_content DUY NHẤT tới Gemini: timeout cứng, không retry
    ngầm, lỗi tạm thời thì cho cặp vào cooldown rồi ném lại cho caller xoay tiếp.
    (Lỗi xảy ra lúc lặp stream nằm ngoài hàm này — stream_answer tự _cool_down.)"""
    import google.generativeai as genai

    if _is_down(api_key, model_name):
        raise RuntimeError(f"Gemini {model_name}: unavailable (cooldown)")
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name).generate_content(
            prompt, request_options={"retry": None, "timeout": timeout}, **kwargs
        )
    except Exception as exc:
        if _is_transient(exc):
            _cool_down(api_key, model_name, exc)
        raise


# Round-robin cursor so consecutive generations start at DIFFERENT (key, model)
# pairs — spreads requests across pairs to stay under each pair's RPM limit,
# instead of hammering pair[0] every call and tripping 429s.
_GEMINI_PAIR_CURSOR = 0


def gemini_generate(
    prompt: str, models: list[str] | None = None, log_input: str | None = None,
    timeout: float = _GEMINI_TIMEOUT_S,
) -> str:
    """Sinh văn bản qua Gemini, xoay vòng (key, model) cho tới khi một cặp trả lời.

    Điểm vào DUY NHẤT cho mọi nơi cần LLM (generator, grader, compliance, agent)
    kể từ khi Gemini là nhà cung cấp duy nhất — một cursor chung giữ cho các lần
    gọi liên tiếp không dồn hết vào cặp đầu tiên.

    `log_input`: phần được GHI LÊN LANGFUSE thay cho `prompt` đầy đủ, khi caller
    có prompt template tĩnh (rules, hướng dẫn định dạng...) ghép với phần biến
    đổi thực sự (câu hỏi, lịch sử). Không có nó, mọi lần gọi đều lộ nguyên văn
    template tĩnh trong "input" — làm trace không đọc được (phần thay đổi chìm
    giữa hàng chục dòng rules không đổi). `prompt` gửi cho Gemini KHÔNG đổi."""
    pairs = _gemini_pairs(models)
    if not pairs:
        raise RuntimeError("Gemini: không còn cặp (key, model) nào dùng được (thiếu key hoặc đều đang cooldown)")

    global _GEMINI_PAIR_CURSOR
    n = len(pairs)
    start = _GEMINI_PAIR_CURSOR
    last_exc: Exception | None = None
    for offset in range(n):
        api_key, model_name = pairs[(start + offset) % n]
        try:
            t0 = datetime.now(timezone.utc)
            response = gemini_call(api_key, model_name, prompt, timeout=timeout)
            t1 = datetime.now(timezone.utc)
            if response.text:
                usage = getattr(response, "usage_metadata", None)
                record_generation(
                    "gemini-generate",
                    model_name,
                    log_input if log_input is not None else prompt,
                    response.text,
                    t0,
                    t1,
                    prompt_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
                    completion_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
                )
                # Advance cursor so the NEXT call starts at the following pair —
                # round-robin keeps any single (key, model) under its RPM limit.
                _GEMINI_PAIR_CURSOR = (start + offset + 1) % n
                return response.text
        except Exception as exc:
            if _is_transient(exc):
                logger.debug("Gemini {} (key…{}) unavailable, rotating: {}",
                             model_name, api_key[-4:], str(exc)[:100])
                last_exc = exc
                continue
            raise  # non-quota error — propagate immediately
    # Mọi cặp im lặng (response.text rỗng) thì last_exc vẫn None — `raise None`
    # sẽ ném TypeError che mất nguyên nhân thật.
    raise last_exc or RuntimeError(
        f"Gemini: cả {n} cặp (key, model) đều không trả về nội dung"
    )


def _call_gemini(
    prompt: str,
    context: str,
    history: list[dict] | None = None,
    models: list[str] | None = None,
) -> str | None:
    """Câu trả lời có trích dẫn cho một câu hỏi, qua vòng xoay của gemini_generate."""
    history_block = ""
    if history:
        lines = [f"{'Người dùng' if m['role'] == 'user' else 'Trợ lý'}: {m['content'][:500]}"
                 for m in history]
        history_block = "\n**Lịch sử hội thoại:**\n" + "\n".join(lines) + "\n\n"
    # Phần biến đổi thực sự mỗi lần gọi — KHÔNG gồm _SYSTEM_PROMPT (~50 dòng rules
    # cố định, luôn giống hệt nhau) để log lên Langfuse còn đọc được.
    variable_part = f"{history_block}**Văn bản tham chiếu:**\n{context}\n\n**Câu hỏi:** {prompt}"
    return gemini_generate(
        f"{_SYSTEM_PROMPT}\n\n{variable_part}",
        models=models,
        log_input=variable_part,
        timeout=_GEMINI_ANSWER_TIMEOUT_S,
    )


@_traceable(
    name="rag-generate-answer",
    run_type="llm",
    tags=["gemini", "legal-qa", "citations"],
)
def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    use_fallback: bool = False,   # giữ cho call site cũ; chỉ còn một nhà cung cấp
    history: list[dict] | None = None,
    min_score: float | None = None,
    as_of_date: str | None = None,
    time_out_of_range: bool = False,
    earliest_covered: str = "",
) -> dict:
    """
    Generate answer with citations.
    Returns: {answer, sources, used_llm, chunk_count}

    Decorated with @traceable: each call appears in LangSmith as a child span
    of the parent 'documind-agent' run, showing the prompt, LLM response, and
    which (key, model) pair of Gemini answered.
    """
    if time_out_of_range:
        # Answering a date the corpus never covered would mean presenting later
        # law as if it applied then — say what the corpus covers instead.
        moc = f" ({as_of_date})" if as_of_date else ""
        tu_ngay = earliest_covered or "mốc sớm nhất của corpus"
        return {
            "answer": (
                f"Câu hỏi về thời điểm{moc} nằm ngoài khoảng thời gian hệ thống phủ. "
                f"Corpus hiện chỉ phủ từ {tu_ngay} trở đi, nên tôi không có căn cứ để trả lời "
                "cho mốc thời gian này.\n\n"
                f"Bạn có thể hỏi lại với một mốc thời điểm từ {tu_ngay} trở đi."
            ),
            "sources": [],
            "used_llm": "none",
            "chunk_count": 0,
        }

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

    complex_query = _is_complex_query(query, history)
    if not complex_query and len(chunks) > _SIMPLE_QUERY_MAX_CHUNKS:
        chunks = chunks[:_SIMPLE_QUERY_MAX_CHUNKS]

    context, citation_list = _build_context(chunks)
    context = _as_of_block(as_of_date) + context

    # Gemini là nhà cung cấp DUY NHẤT (2026-09-19). Chịu lỗi nằm ở vòng xoay
    # (key × model) bên trong _call_gemini; hết mọi cặp thì rơi về trích dẫn
    # nguyên văn, không gọi nhà cung cấp nào khác.
    try:
        answer = _call_gemini(query, context, history=history, models=_select_models(complex_query))
        used_llm = "gemini"
        logger.info("Gemini answered query ({} chars)", len(answer or ""))
    except Exception as exc:
        logger.error("Gemini failed on every (key, model) pair: {}", exc)
        answer = _build_extractive_answer(query, chunks)
        used_llm = "extractive_fallback"

    return {
        "answer": answer,
        "sources": _cited_sources(answer, chunks),
        "used_llm": used_llm,
        "chunk_count": len(chunks),
    }


def _stream_chunks_in_thread(model_name: str, api_key: str, prompt: str, q: queue.Queue) -> None:
    """Chạy trong thread riêng, KHÔNG phải trong event loop asyncio.

    genai `generate_content(..., stream=True)` trả về iterator ĐỒNG BỘ — lặp
    `for chunk in stream` trực tiếp bên trong 1 hàm async thì mỗi lần chờ
    chunk kế tiếp từ mạng sẽ đứng hình TOÀN BỘ event loop (mọi WS/REST khác
    trên cùng process, không chỉ kết nối đang stream) cho tới khi chunk đó về
    — đo được thật: 1 request WS treo là kéo theo 1 request REST hoàn toàn
    không liên quan chờ tới hơn 1 phút. Đẩy từng chunk qua queue để phía async
    chỉ `await asyncio.to_thread(q.get)` — nhường lại event loop trong lúc chờ."""
    try:
        stream = gemini_call(api_key, model_name, prompt, timeout=_GEMINI_ANSWER_TIMEOUT_S, stream=True)
        for chunk in stream:
            if chunk.text:
                q.put(("token", chunk.text))
        q.put(("usage", getattr(stream, "usage_metadata", None)))
    except Exception as exc:
        q.put(("error", exc))
    finally:
        q.put(("end", None))


async def stream_answer(
    query: str,
    chunks: list[RetrievedChunk],
) -> AsyncIterator[str]:
    """
    Stream tokens from Gemini, rotating over (key, model) pairs like the
    non-streaming path. Yields text chunks for WebSocket/SSE streaming.
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

    complex_query = _is_complex_query(query)
    if not complex_query and len(chunks) > _SIMPLE_QUERY_MAX_CHUNKS:
        chunks = chunks[:_SIMPLE_QUERY_MAX_CHUNKS]

    context, citation_list = _build_context(chunks)

    pairs = _gemini_pairs(_select_models(complex_query))
    if not pairs:
        yield _build_extractive_answer(query, chunks)
        return

    variable_part = f"**Văn bản tham chiếu:**\n{context}\n\n**Câu hỏi:** {query}"
    full_prompt = f"{_SYSTEM_PROMPT}\n\n{variable_part}"
    # Cùng vòng xoay (key, model) như đường không streaming; chỉ đổi cặp khi lỗi
    # xảy ra TRƯỚC token đầu tiên — đổi giữa chừng thì client đã nhận nửa câu trả
    # lời của cặp trước, nối tiếp bằng cặp khác sẽ ra văn bản chắp vá.
    for api_key, model_name in pairs:
        streamed = False
        answer_parts: list[str] = []
        usage = None
        stream_exc: Exception | None = None
        t0 = datetime.now(timezone.utc)

        q: queue.Queue = queue.Queue()
        threading.Thread(
            target=_stream_chunks_in_thread,
            args=(model_name, api_key, full_prompt, q),
            daemon=True,
        ).start()

        while True:
            kind, payload = await asyncio.to_thread(q.get)
            if kind == "token":
                streamed = True
                answer_parts.append(payload)
                yield payload
            elif kind == "usage":
                usage = payload
            elif kind == "error":
                stream_exc = payload
            elif kind == "end":
                break

        if stream_exc is not None:
            logger.warning("Gemini stream {} (key…{}) failed: {}",
                           model_name, api_key[-4:], str(stream_exc)[:150])
            if _is_transient(stream_exc):
                _cool_down(api_key, model_name, stream_exc)
            if streamed:
                return  # nửa câu trả lời đã ra — không nối thêm từ cặp khác
            continue  # chưa có chữ nào ra — thử cặp (key, model) kế tiếp

        if streamed:
            record_generation(
                "gemini-generate-stream",
                model_name,
                variable_part,
                "".join(answer_parts),
                t0,
                datetime.now(timezone.utc),
                prompt_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
                completion_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            )
            return

    logger.error("Gemini streaming failed on every (key, model) pair")
    yield _build_extractive_answer(query, chunks)
