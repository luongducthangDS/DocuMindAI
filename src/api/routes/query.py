"""
Query routes: POST /api/v1/query (JSON) + WebSocket /api/v1/ws/{session_id}
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from starlette.requests import HTTPConnection
from loguru import logger

from src.agent.graph import run_agent
from src.agent.memory import ShortTermMemory, get_long_term_memory
from src.api.principal import InvalidApiKey, context_from_headers, resolve_context
from src.api.schemas import ComplianceVerdict, QueryRequest, QueryResponse, SourceItem, ThinkingStep
from src.config import DOMAIN_NAME, get_settings
from src.guardrails import check_prompt_injection, redact_pii, truncate_ip, validate_citations
from src.langfuse_otel import end_trace, start_trace
from src.rag.context import (
    PUBLIC_TENANT,
    degraded_flags,
    mark_degraded,
    reset_current_context,
    reset_degraded,
    set_current_context,
    start_degraded,
)
from src.rag.generator import _cited_sources, stream_answer, trace_answer
from src.rag.temporal import today_iso

_CHAT_LOG: Path | None = None


def _get_chat_log() -> Path:
    global _CHAT_LOG
    if _CHAT_LOG is None:
        from src.config import get_settings

        log_dir = get_settings().logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        _CHAT_LOG = log_dir / "chat_history.jsonl"
    return _CHAT_LOG


_PII_LOG_FIELDS = ("query", "answer", "error", "guard_reason")


def _log_chat(entry: dict) -> None:
    """Một chỗ ghi duy nhất cho chat_history.jsonl — che PII + cắt IP ở đây (NĐ 13/2023)."""
    entry = dict(entry)
    for key in _PII_LOG_FIELDS:
        if isinstance(entry.get(key), str):
            entry[key] = redact_pii(entry[key])
    if "ip" in entry:
        entry["ip"] = truncate_ip(entry["ip"])
    try:
        with _get_chat_log().open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Failed to write chat log: {}", exc)

router = APIRouter(prefix="/api/v1", tags=["query"])


def _get_client_ip(request: HTTPConnection) -> str:
    """Proxy-aware IP extraction — the LAST X-Forwarded-For entry, not the first.

    Proxy nối địa chỉ nó thấy vào CUỐI header; mọi thứ bên trái là client tự
    khai. Lấy phần tử đầu = để client chọn bucket rate limit và IP trong audit
    log (đổi header mỗi request là vượt RATE_LIMIT_PER_MINUTE).

    ponytail: giả định đúng MỘT proxy tin cậy (Render) phía trước. Thêm một tầng
    nữa (Cloudflare → Render) thì phần tử cuối là IP của tầng đó và cả site chung
    một bucket — khi ấy lấy phần tử thứ N từ phải, N = số proxy tin cậy.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    last_hop = forwarded.split(",")[-1].strip()
    if last_hop:
        return last_hop[:45]
    return (request.client.host if request.client else "unknown")[:45]


# ── WebSocket abuse limits (R3) ───────────────────────────────────────────────
# SlowAPI chỉ bọc HTTP. Một socket đã mở gửi bao nhiêu câu cũng được, mỗi câu
# 2–4 lời gọi Gemini (thêm dấu, contextualize, embed, generate) → cạn quota của
# mọi người dùng thật.
# ponytail: đếm trong RAM — đúng với 1 uvicorn worker (Render). Nhiều worker hoặc
# nhiều instance thì mỗi process đếm riêng; khi đó chuyển bộ đếm sang Redis.
WS_MAX_CONNECTIONS_PER_IP = 3
WS_MESSAGES_PER_MINUTE = 6
_ws_open_by_ip: dict[str, int] = {}


class _TokenBucket:
    """Đầy `per_minute` token, hồi đều theo thời gian — cho phép dồn `per_minute` câu rồi chờ."""

    def __init__(self, per_minute: int, clock=time.monotonic) -> None:
        self.capacity = float(per_minute)
        self.tokens = float(per_minute)
        self.rate = per_minute / 60.0
        self.clock = clock
        self.updated = clock()

    def take(self) -> bool:
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


def _ws_origin_allowed(origin: str | None, host: str | None = None) -> bool:
    """Chặn Cross-Site WebSocket Hijacking: trình duyệt LUÔN gửi Origin.

    Không có Origin = client không phải trình duyệt (curl, SDK) — không mang
    credential ngầm nào của nạn nhân, nên không phải CSWSH; vẫn chịu token bucket.
    Origin trùng Host = chính UI do server này phục vụ (Render một service, không
    Vercel) — cũng không phải CSWSH. Thiếu nhánh này, UI same-origin bị 403 ở mọi
    lần mở WS và lặng lẽ lùi về REST, mất stream (đo 2026-10-02 trên pd0r).
    So netloc, không so scheme: Render kết thúc TLS ở proxy.
    """
    if not origin or origin in get_settings().cors_origins:
        return True
    return bool(host) and urlsplit(origin).netloc == host


def _release_ws_slot(ip: str) -> None:
    n = _ws_open_by_ip.get(ip, 0) - 1
    if n > 0:
        _ws_open_by_ip[ip] = n
    else:
        _ws_open_by_ip.pop(ip, None)

# In-process session cache — not shared across workers/instances. LRU có trần:
# session bị đẩy ra chỉ mất cache, lượt sau hydrate lại từ SQLite.
MAX_SESSIONS = 1000
_sessions: OrderedDict[tuple[str, str], ShortTermMemory] = OrderedDict()


def _get_session(tenant_id: str, session_id: str) -> ShortTermMemory:
    """Lịch sử khoá theo (tenant, session_id). session_id do client tự khai —
    khoá trần thì tenant này đọc được lịch sử tenant kia qua bước contextualize
    (mặc định cũ "default" còn cho mọi client không gửi id chung một lịch sử)."""
    key = (tenant_id, session_id)
    if key in _sessions:
        _sessions.move_to_end(key)
        return _sessions[key]
    # public giữ khoá SQLite cũ để lịch sử web hiện có không mất; tenant khác có
    # tiền tố. session_id chỉ gồm [A-Za-z0-9_-] nên "acme:x" không thể trùng khoá public.
    store_id = session_id if tenant_id == PUBLIC_TENANT else f"{tenant_id}:{session_id}"
    # Hydrate từ SQLite nếu worker này chưa thấy session (sau restart/redeploy).
    _sessions[key] = ShortTermMemory(max_turns=10, session_id=store_id)
    while len(_sessions) > MAX_SESSIONS:
        _sessions.popitem(last=False)
    return _sessions[key]


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_endpoint(request: Request, body: QueryRequest) -> QueryResponse:
    """
    Main Q&A endpoint. Routes through LangGraph agent.
    Returns structured answer with mandatory citations.
    """
    t0 = time.time()
    ip = _get_client_ip(request)
    ua = request.headers.get("user-agent", "")[:200]
    retrieval_ctx = resolve_context(request, body.as_of_date)
    session = _get_session(retrieval_ctx.tenant_id, body.session_id)

    # Guardrail: prompt injection check
    guard = check_prompt_injection(body.query)
    if guard.blocked:
        get_long_term_memory().log_error("/api/v1/query", "PromptInjectionBlocked", body.session_id)
        _log_chat({
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": body.session_id,
            "query": body.query,
            "answer": None,
            "error": "guard_blocked",
            "guard_triggered": True,
            "guard_reason": guard.reason,
            "ip": ip,
            "user_agent": ua,
            "latency_ms": 0,
        })
        logger.warning("Guard blocked query session={} reason={}", body.session_id, guard.reason)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query không hợp lệ. Vui lòng đặt câu hỏi về {DOMAIN_NAME}.",
        )

    error_detail: str | None = None
    result: dict = {}
    try:
        from src.api.main import ensure_rag_initialized

        await ensure_rag_initialized()
        result = await run_agent(
            query=body.query,
            session_id=body.session_id,
            history=session.as_messages(),
            as_of_date=body.as_of_date,
            retrieval_ctx=retrieval_ctx,
        )
    except HTTPException:
        # ensure_rag_initialized() đã trả sẵn 503 kèm loại lỗi — giữ nguyên,
        # đừng bọc lại thành "Agent temporarily unavailable (HTTPException)".
        raise
    except Exception as exc:
        error_detail = str(exc)
        logger.error("Agent failed for query '{}': {}", body.query[:60], exc)
        get_long_term_memory().log_error("/api/v1/query", type(exc).__name__, body.session_id)
        _log_chat({
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": body.session_id,
            "query": body.query,
            "answer": None,
            "error": error_detail,
            "guard_triggered": False,
            "ip": ip,
            "user_agent": ua,
            "latency_ms": int((time.time() - t0) * 1000),
        })
        # Chỉ tên lớp lỗi, không kèm nội dung: đủ để định vị khi chẩn đoán từ xa
        # mà không đẩy đường dẫn/cấu hình nội bộ ra ngoài. Chi tiết nằm ở log.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Agent temporarily unavailable ({type(exc).__name__}). Please retry.",
        ) from exc

    # Guardrail: citation hallucination validation
    answer_text = result.get("answer", "")
    chunk_count = len(result.get("retrieved_chunks", []))
    cleaned_answer, invalid_citations = validate_citations(answer_text, chunk_count)
    if invalid_citations:
        logger.warning(
            "Citation hallucination: session={} invalid_indices={}", body.session_id, invalid_citations
        )

    # Update short-term memory
    session.add("user", body.query)
    session.add("assistant", cleaned_answer)

    latency = int((time.time() - t0) * 1000)

    sources = [
        SourceItem(
            index=s.get("index", i + 1),
            title=s.get("title", ""),
            dieu_header=s.get("dieu_header", ""),
            source_url=s.get("source_url", ""),
            score=float(s.get("score", 0)),
            source=str(s.get("source", "")),
        )
        for i, s in enumerate(result.get("sources", []))
    ]

    _log_chat({
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": body.session_id,
        "query": body.query,
        "answer": cleaned_answer,
        "used_llm": result.get("used_llm", "unknown"),
        "chunk_count": chunk_count,
        "source_titles": [s.get("title", "") for s in result.get("sources", [])[:3]],
        "latency_ms": latency,
        "ip": ip,
        "user_agent": ua,
        "guard_triggered": False,
        "invalid_citations": invalid_citations,
        "degraded": result.get("degraded", []),
    })

    steps = [
        ThinkingStep(label=s.get("label", ""), detail=s.get("detail", ""), ms=s.get("ms", 0))
        for s in (result.get("steps") or [])
    ]

    compliance = None
    compliance_result = result.get("compliance_result")
    if compliance_result:
        compliance = ComplianceVerdict(
            matched=compliance_result.get("matched", False),
            verdict=compliance_result.get("verdict", "no_match"),
            criterion_id=compliance_result.get("criterion_id", ""),
            extracted_value=compliance_result.get("extracted_value"),
        )

    return QueryResponse(
        answer=cleaned_answer,
        sources=sources,
        used_llm=result.get("used_llm", "unknown"),
        chunk_count=chunk_count,
        latency_ms=result.get("latency_ms", latency),
        session_id=body.session_id,
        steps=steps,
        retry_count=result.get("retry_count", 0),
        grade_reason=result.get("grade_reason", ""),
        compliance=compliance,
        degraded=result.get("degraded", []),
    )


@router.get("/whoami")
async def whoami(request: Request) -> dict:
    """Which tenant and ACL labels this request resolves to.

    Exists so a client can confirm a key before asking anything — otherwise a
    wrong key shows up only as a 401 on the first question, and a *valid* key
    for the wrong tenant shows up as nothing at all: just different answers.
    """
    rctx = resolve_context(request)
    return {
        "tenant_id": rctx.tenant_id,
        "acl_labels": sorted(rctx.acl_labels),
        "authenticated": rctx.tenant_id != PUBLIC_TENANT,
    }


async def _send_step(websocket: WebSocket, enabled: bool, label: str, detail: str, started: float) -> None:
    """Một bước tiến trình, cùng dạng ThinkingStep của REST ({label, detail, ms})."""
    if enabled:
        ms = int((time.perf_counter() - started) * 1000)
        await websocket.send_json({"step": {"label": label, "detail": detail, "ms": ms}})


@router.websocket("/ws/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str) -> None:
    """
    WebSocket endpoint for streaming token-by-token responses.
    Client sends: {"query": "...", "progress": true}
    Server sends: [{"step": {...}} ...] token chunks, then {"done": true, "sources": [...]}

    "progress" là opt-in: client cũ coi mọi JSON không có done/error là token
    và sẽ in thẳng {"step": ...} ra khung chat.
    """
    # Validate session_id before accepting
    import re

    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", session_id):
        await websocket.close(code=1008, reason="Invalid session_id")
        return

    origin = websocket.headers.get("origin")
    if not _ws_origin_allowed(origin, websocket.headers.get("host")):
        logger.warning("WS rejected: origin {!r} not in ALLOWED_ORIGINS", origin)
        await websocket.close(code=1008, reason="Origin not allowed")
        return

    await websocket.accept()
    try:
        ws_ctx = context_from_headers(websocket.headers)
    except InvalidApiKey:
        # 4401: application-level "unauthorized" in the private close-code range.
        logger.warning("WS rejected: bad API key, session={}", session_id)
        await websocket.close(code=4401, reason="API key không hợp lệ.")
        return
    client_ip = _get_client_ip(websocket)
    if _ws_open_by_ip.get(client_ip, 0) >= WS_MAX_CONNECTIONS_PER_IP:
        logger.warning("WS rejected: {} already has {} open sockets", client_ip, WS_MAX_CONNECTIONS_PER_IP)
        await websocket.close(code=1008, reason="Too many connections")
        return
    # Kiểm và tăng liền nhau, không await xen giữa → nguyên tử trong asyncio.
    # Trả slot ở `finally` cuối hàm.
    _ws_open_by_ip[client_ip] = _ws_open_by_ip.get(client_ip, 0) + 1
    bucket = _TokenBucket(WS_MESSAGES_PER_MINUTE)
    logger.info("WebSocket connected: session={} tenant={}", session_id, ws_ctx.tenant_id)

    try:
        while True:
            data = await websocket.receive_json()
            if not bucket.take():
                await websocket.send_json({
                    "error": f"Quá nhiều câu hỏi — tối đa {WS_MESSAGES_PER_MINUTE} câu/phút, vui lòng chờ.",
                })
                continue
            raw_query = data.get("query", "").strip()
            query = raw_query
            progress = bool(data.get("progress"))

            # min 2 — không phải 3: khớp đúng /api/v1/query (QueryRequest.query)
            # và với chính danh sách lời chào ngắn nhất của _SMALLTALK_RE ("hi",
            # "ok") — trước đây 3 ký tự chặn cả "hi" trước khi kịp tới nhánh
            # smalltalk, người dùng nhận lỗi validate khó hiểu ngay câu đầu tiên.
            if not query or len(query) < 2:
                await websocket.send_json({"error": "Query too short (min 2 chars)"})
                continue
            if len(query) > 1000:
                await websocket.send_json({"error": "Query too long (max 1000 chars)"})
                continue

            # Guardrail: prompt injection check
            guard = check_prompt_injection(query)
            if guard.blocked:
                logger.warning("WS guard blocked session={} reason={}", session_id, guard.reason)
                await websocket.send_json({
                    "error": f"Query không hợp lệ. Vui lòng đặt câu hỏi về {DOMAIN_NAME}.",
                    "guard_triggered": True,
                })
                continue

            # Resolve context-dependent follow-ups before retrieval — same
            # contextualization the REST /query path gets via run_agent's
            # do_contextualize node, otherwise WS follow-ups reproduce the
            # "90 điểm thì sao" false-negative bug.
            # Kiểm tra RAG trước khi viết lại câu hỏi: hai bước rewrite đều gọi
            # Gemini, không có lý do đốt chúng khi retrieval chắc chắn hỏng.
            from src.api.main import ensure_rag_initialized

            try:
                await ensure_rag_initialized()
            except HTTPException as exc:
                # RAG hỏng: nói thẳng thay vì stream ra "không tìm thấy văn bản"
                # — client không phân biệt được hỏng với ngoài phạm vi.
                await websocket.send_json({"error": exc.detail, "done": True})
                continue

            # Trace này scope theo 1 lượt hỏi-đáp WS (không try/finally quanh cả
            # khối bên dưới để tránh re-indent lớn — an toàn vì mỗi kết nối WS
            # chạy trong 1 asyncio Task riêng, contextvar không rò sang request
            # khác kể cả khi có exception thoát khỏi vòng lặp này).
            # lf_token (không phải "token") — vòng lặp stream bên dưới dùng
            # đúng tên "token" cho từng chunk văn bản; trùng tên sẽ ghi đè mất
            # contextvars.Token thật, làm end_trace() nhận nhầm 1 chuỗi text
            # và raise TypeError ("expected an instance of Token") — đã xảy ra
            # thật, WS trả "Server error" dù câu trả lời stream ra đúng hết.
            ctx, lf_token = start_trace(
                "documind-ws-query", session_id=session_id,
                tags=["legal-qa", "feature:websocket-chat", f"env:{get_settings().environment}"],
            )

            # Browsers cannot set headers on a WebSocket handshake (the WebSocket
            # constructor takes a URL and subprotocols, nothing else), so the UI
            # sends its key inside the message. A header, when present, still
            # works for non-browser clients. Resolved per turn: the key in the
            # message is what this question is asked under — and whose history
            # it reads, so it is resolved BEFORE the session is looked up.
            msg_key = str(data.get("api_key") or "").strip()
            try:
                base_ctx = context_from_headers({"x-api-key": msg_key}) if msg_key else ws_ctx
            except InvalidApiKey as exc:
                await websocket.send_json({"error": str(exc)})
                end_trace(ctx, lf_token, "documind-ws-query", raw_query, "")
                continue
            session = _get_session(base_ctx.tenant_id, session_id)

            history = session.as_messages()
            t_step = time.perf_counter()
            from src.agent.graph import _expand_legal_terms, _normalize_query

            # Thêm dấu + diễn giải theo ngữ cảnh: tối đa 1 lời gọi Gemini đồng bộ (tới
            # 20s × vài cặp key/model) — trong thread, không thì đứng cả event loop.
            query = await asyncio.to_thread(_normalize_query, query, history[-6:])
            await _send_step(
                websocket, progress, "Phân tích câu hỏi",
                query if query != raw_query else "Giữ nguyên câu hỏi", t_step,
            )

            # Import retriever to get chunks
            # Per-message as_of so a client can ask the same question at two
            # points in time on one socket; falls back to the connection default.
            turn_ctx = replace(base_ctx, as_of_date=(data.get("as_of_date") or "").strip() or None)
            # The WS path streams without going through run_agent, so it sets the
            # ambient context itself — otherwise anything this turn reaches would
            # fall back to the public default.
            ws_ctx_token = set_current_context(turn_ctx)
            degraded_token = start_degraded()
            try:
                t_step = time.perf_counter()
                t_retrieve = datetime.now(timezone.utc)
                # Câu mở rộng thuật ngữ chỉ dùng để truy hồi; LLM vẫn nhận câu gốc.
                search_query = _expand_legal_terms(query)
                from src.rag.retriever import retrieve_direct_chroma, retrieve_with_context, trace_retrieval

                retrieval_path = "hybrid"
                try:
                    chunks = await asyncio.to_thread(retrieve_with_context, search_query, turn_ctx)
                    if not chunks:
                        chunks = await asyncio.to_thread(retrieve_direct_chroma, search_query, ctx=turn_ctx)
                        retrieval_path = "direct"
                except Exception as exc:
                    logger.warning("Retrieval failed in WS handler: {}", exc)
                    mark_degraded("retriever_error")
                    chunks = await asyncio.to_thread(retrieve_direct_chroma, search_query, ctx=turn_ctx)
                    retrieval_path = f"error_fallback:{type(exc).__name__}"
                trace_retrieval(search_query, chunks, retrieval_path, turn_ctx, t_retrieve)
                await _send_step(websocket, progress, "Tìm kiếm tài liệu", f"{len(chunks)} đoạn liên quan", t_step)
                # Lọc hiệu lực đã đẩy xuống vector store cùng lượt truy hồi ở trên
                # (RetrievalContext) — bước này chỉ báo mốc đã dùng, không tốn thêm thời gian.
                as_of = turn_ctx.as_of_date
                await _send_step(
                    websocket, progress, "Lọc theo hiệu lực",
                    f"Áp dụng tại {'/'.join(reversed(as_of.split('-')))}" if as_of else "Quy định hiện hành",
                    time.perf_counter(),
                )

                t_step = time.perf_counter()
                answer_parts = []
                async for token in stream_answer(query, chunks):
                    await websocket.send_text(token)
                    answer_parts.append(token)
            finally:
                # In a finally: one socket serves many turns, and a turn that
                # raised mid-stream would otherwise leave its context set for
                # whatever runs next on this connection.
                degraded = degraded_flags()
                reset_degraded(degraded_token)
                reset_current_context(ws_ctx_token)

            full_answer = "".join(answer_parts)
            sources = _cited_sources(full_answer, chunks)
            await _send_step(websocket, progress, "Tổng hợp câu trả lời", f"{len(sources)} trích dẫn", t_step)
            audit_flags = trace_answer(full_answer, sources, {"chunks": len(chunks)}) or []
            end_trace(ctx, lf_token, "documind-ws-query", raw_query, full_answer,
                      extra_tags=[f"degraded:{f}" for f in degraded]
                      + [f"as_of:{turn_ctx.as_of_date or today_iso()}",
                         f"as_of_src:{'request' if turn_ctx.as_of_date else 'today'}"]
                      + audit_flags)
            if degraded:
                logger.warning("DEGRADED WS answer session={} flags={}", session_id, degraded)
            _, invalid_citations = validate_citations(full_answer, len(chunks))
            if invalid_citations:
                logger.warning(
                    "WS citation hallucination: session={} invalid={}", session_id, invalid_citations
                )

            # Log the raw text the user actually typed, not the LLM-rewritten
            # retrieval query — otherwise next turn's contextualize call reads
            # its own prior rewrite as if it were the user's words, and the
            # drift compounds turn over turn.
            session.add("user", raw_query)
            session.add("assistant", full_answer)
            # _cited_sources (hàm dùng chung với REST, generator.py) — không tự
            # ghép chunks[:5]: (a) tên field khớp đúng Source phía frontend
            # (index/title/dieu_header/source_url), chunks[:5] cũ thiếu "index"
            # nên frontend render literal "[]"; (b) chỉ trả chunk THẬT SỰ được
            # trích dẫn [N] trong câu trả lời — chunks[:5] cũ hiện đủ 5 nguồn dù
            # câu trả lời là "không tìm thấy quy định này" (0 trích dẫn), gây
            # hiển thị mâu thuẫn: trả lời "không tìm thấy" nhưng vẫn liệt kê nguồn.
            await websocket.send_json({
                "done": True,
                "sources": sources,
                "degraded": degraded,
            })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session={}", session_id)
    except Exception as exc:
        logger.error("WebSocket error: {}", exc)
        try:
            await websocket.send_json({"error": "Server error. Please reconnect."})
        except Exception:
            pass
    finally:
        _release_ws_slot(client_ip)
