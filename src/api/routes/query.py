"""
Query routes: POST /api/v1/query (JSON) + WebSocket /api/v1/ws/{session_id}
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from loguru import logger

from src.agent.graph import run_agent
from src.agent.memory import ShortTermMemory, get_long_term_memory
from src.api.principal import InvalidApiKey, context_from_headers, resolve_context
from src.api.schemas import ComplianceVerdict, QueryRequest, QueryResponse, SourceItem, ThinkingStep
from src.config import DOMAIN_NAME, get_settings
from src.guardrails import check_prompt_injection, validate_citations
from src.langfuse_otel import end_trace, start_trace
from src.rag.context import PUBLIC_TENANT, reset_current_context, set_current_context
from src.rag.generator import _cited_sources, stream_answer

_CHAT_LOG: Path | None = None


def _get_chat_log() -> Path:
    global _CHAT_LOG
    if _CHAT_LOG is None:
        from src.config import get_settings

        log_dir = get_settings().logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        _CHAT_LOG = log_dir / "chat_history.jsonl"
    return _CHAT_LOG


def _log_chat(entry: dict) -> None:
    try:
        with _get_chat_log().open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Failed to write chat log: {}", exc)

router = APIRouter(prefix="/api/v1", tags=["query"])


def _get_client_ip(request: Request) -> str:
    """Proxy-aware IP extraction — respects X-Forwarded-For (Render/Cloudflare)."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()[:45]
    return (request.client.host if request.client else "unknown")[:45]

# In-process session store — not shared across workers/instances.
_sessions: dict[str, ShortTermMemory] = {}


def _get_session(session_id: str) -> ShortTermMemory:
    if session_id not in _sessions:
        # session_id=... => hydrate từ SQLite nếu worker này chưa từng thấy
        # session này trong RAM (vd. sau restart/redeploy, hoặc worker khác
        # trong cluster nhiều process) — không còn mất ngữ cảnh câm lặng.
        _sessions[session_id] = ShortTermMemory(max_turns=10, session_id=session_id)
    return _sessions[session_id]


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_endpoint(request: Request, body: QueryRequest) -> QueryResponse:
    """
    Main Q&A endpoint. Routes through LangGraph agent.
    Returns structured answer with mandatory citations.
    """
    t0 = time.time()
    ip = _get_client_ip(request)
    ua = request.headers.get("user-agent", "")[:200]
    session = _get_session(body.session_id)

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
            retrieval_ctx=resolve_context(request, body.as_of_date),
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

    await websocket.accept()
    try:
        ws_ctx = context_from_headers(websocket.headers)
    except InvalidApiKey:
        # 4401: application-level "unauthorized" in the private close-code range.
        logger.warning("WS rejected: bad API key, session={}", session_id)
        await websocket.close(code=4401, reason="API key không hợp lệ.")
        return
    session = _get_session(session_id)
    logger.info("WebSocket connected: session={} tenant={}", session_id, ws_ctx.tenant_id)

    try:
        while True:
            data = await websocket.receive_json()
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

            history = session.as_messages()
            t_step = time.perf_counter()
            from src.agent.graph import (
                _contextualize_query,
                _needs_diacritics,
                _restore_diacritics,
            )

            if _needs_diacritics(query):
                query = _restore_diacritics(query)
            if history:
                query = _contextualize_query(query, history[-6:])
            await _send_step(
                websocket, progress, "Phân tích câu hỏi",
                query if query != raw_query else "Giữ nguyên câu hỏi", t_step,
            )

            # Import retriever to get chunks
            # Per-message as_of so a client can ask the same question at two
            # points in time on one socket; falls back to the connection default.
            # Browsers cannot set headers on a WebSocket handshake (the WebSocket
            # constructor takes a URL and subprotocols, nothing else), so the UI
            # sends its key inside the message. A header, when present, still
            # works for non-browser clients. Resolved per turn: the key in the
            # message is what this question is asked under.
            msg_key = str(data.get("api_key") or "").strip()
            try:
                base_ctx = context_from_headers({"x-api-key": msg_key}) if msg_key else ws_ctx
            except InvalidApiKey as exc:
                await websocket.send_json({"error": str(exc)})
                end_trace(ctx, lf_token, "documind-ws-query", raw_query, "")
                continue
            turn_ctx = replace(base_ctx, as_of_date=(data.get("as_of_date") or "").strip() or None)
            # The WS path streams without going through run_agent, so it sets the
            # ambient context itself — otherwise anything this turn reaches would
            # fall back to the public default.
            ws_ctx_token = set_current_context(turn_ctx)
            try:
                t_step = time.perf_counter()
                try:
                    from src.rag.retriever import retrieve_with_context

                    chunks = await asyncio.to_thread(retrieve_with_context, query, turn_ctx)
                    if not chunks:
                        from src.rag.retriever import retrieve_direct_chroma

                        chunks = retrieve_direct_chroma(query, ctx=turn_ctx)
                except Exception as exc:
                    logger.warning("Retrieval failed in WS handler: {}", exc)
                    from src.rag.retriever import retrieve_direct_chroma

                    chunks = retrieve_direct_chroma(query, ctx=turn_ctx)
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
                reset_current_context(ws_ctx_token)

            full_answer = "".join(answer_parts)
            sources = _cited_sources(full_answer, chunks)
            await _send_step(websocket, progress, "Tổng hợp câu trả lời", f"{len(sources)} trích dẫn", t_step)
            end_trace(ctx, lf_token, "documind-ws-query", raw_query, full_answer)
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
            })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session={}", session_id)
    except Exception as exc:
        logger.error("WebSocket error: {}", exc)
        try:
            await websocket.send_json({"error": "Server error. Please reconnect."})
        except Exception:
            pass
