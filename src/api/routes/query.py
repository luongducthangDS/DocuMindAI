"""
Query routes: POST /api/v1/query (JSON) + WebSocket /api/v1/ws/{session_id}
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from loguru import logger

from src.agent.graph import run_agent
from src.agent.memory import ShortTermMemory, get_long_term_memory
from src.api.schemas import ComplianceVerdict, QueryRequest, QueryResponse, SourceItem, ThinkingStep
from src.guardrails import check_prompt_injection, validate_citations
from src.rag.generator import stream_answer

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
    """Proxy-aware IP extraction — respects X-Forwarded-For (Railway/Cloudflare)."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()[:45]
    return (request.client.host if request.client else "unknown")[:45]

# In-process session store — not shared across workers/instances.
_sessions: dict[str, ShortTermMemory] = {}


def _get_session(session_id: str) -> ShortTermMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ShortTermMemory(max_turns=10)
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
            detail="Query không hợp lệ. Vui lòng đặt câu hỏi về quy định UNETI.",
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
        )
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
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent temporarily unavailable. Please retry.",
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


@router.websocket("/ws/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str) -> None:
    """
    WebSocket endpoint for streaming token-by-token responses.
    Client sends: {"query": "..."}
    Server sends: token chunks, then {"done": true, "sources": [...]}
    """
    # Validate session_id before accepting
    import re

    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", session_id):
        await websocket.close(code=1008, reason="Invalid session_id")
        return

    await websocket.accept()
    session = _get_session(session_id)
    logger.info("WebSocket connected: session={}", session_id)

    try:
        while True:
            data = await websocket.receive_json()
            raw_query = data.get("query", "").strip()
            query = raw_query

            if not query or len(query) < 3:
                await websocket.send_json({"error": "Query too short (min 3 chars)"})
                continue
            if len(query) > 1000:
                await websocket.send_json({"error": "Query too long (max 1000 chars)"})
                continue

            # Guardrail: prompt injection check
            guard = check_prompt_injection(query)
            if guard.blocked:
                logger.warning("WS guard blocked session={} reason={}", session_id, guard.reason)
                await websocket.send_json({
                    "error": "Query không hợp lệ. Vui lòng đặt câu hỏi về quy định UNETI.",
                    "guard_triggered": True,
                })
                continue

            # Resolve context-dependent follow-ups before retrieval — same
            # contextualization the REST /query path gets via run_agent's
            # do_contextualize node, otherwise WS follow-ups reproduce the
            # "90 điểm thì sao" false-negative bug.
            history = session.as_messages()
            if history:
                from src.agent.graph import _contextualize_query

                query = _contextualize_query(query, history[-6:])

            # Import retriever to get chunks
            try:
                from src.api.main import ensure_rag_initialized

                await ensure_rag_initialized()
                import src.rag.retriever as r_module

                retriever = getattr(r_module, "_active_retriever", None)
                if retriever:
                    if hasattr(retriever, "aretrieve"):
                        nodes = await retriever.aretrieve(query)
                    else:
                        nodes = retriever.retrieve(query)
                    from src.rag.retriever import nodes_to_chunks

                    chunks = nodes_to_chunks(nodes)
                else:
                    chunks = []
                if not chunks:
                    from src.rag.retriever import retrieve_direct_chroma

                    chunks = retrieve_direct_chroma(query)
            except Exception as exc:
                logger.warning("Retrieval failed in WS handler: {}", exc)
                from src.rag.retriever import retrieve_direct_chroma

                chunks = retrieve_direct_chroma(query)

            answer_parts = []
            async for token in stream_answer(query, chunks):
                await websocket.send_text(token)
                answer_parts.append(token)

            full_answer = "".join(answer_parts)
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
            await websocket.send_json({
                "done": True,
                "sources": [
                    {
                        "title": c.metadata.get("title", ""),
                        "dieu": c.metadata.get("dieu_header", ""),
                        "url": c.metadata.get("source_url", ""),
                    }
                    for c in chunks[:5]
                ],
            })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session={}", session_id)
    except Exception as exc:
        logger.error("WebSocket error: {}", exc)
        try:
            await websocket.send_json({"error": "Server error. Please reconnect."})
        except Exception:
            pass
