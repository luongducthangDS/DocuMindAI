"""
Query routes: POST /api/v1/query (JSON) + WebSocket /api/v1/ws/{session_id}
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from loguru import logger

from src.agent.graph import run_agent
from src.agent.memory import ShortTermMemory
from src.api.schemas import QueryRequest, QueryResponse, SourceItem
from src.rag.generator import stream_answer

router = APIRouter(prefix="/api/v1", tags=["query"])

# In-process session stores — replaced by Redis in production
_sessions: dict[str, ShortTermMemory] = {}


def _get_session(session_id: str) -> ShortTermMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ShortTermMemory(max_turns=10)
    return _sessions[session_id]


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_endpoint(body: QueryRequest) -> QueryResponse:
    """
    Main Q&A endpoint. Routes through LangGraph agent.
    Returns structured answer with mandatory citations.
    """
    t0 = time.time()
    session = _get_session(body.session_id)

    try:
        from src.api.main import ensure_rag_initialized

        await ensure_rag_initialized()
        result = await run_agent(
            query=body.query,
            session_id=body.session_id,
            history=session.as_messages(),
        )
    except Exception as exc:
        logger.error("Agent failed for query '{}': {}", body.query[:60], exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent temporarily unavailable. Please retry.",
        ) from exc

    # Update short-term memory
    session.add("user", body.query)
    session.add("assistant", result.get("answer", ""))

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

    return QueryResponse(
        answer=result.get("answer", ""),
        sources=sources,
        used_llm=result.get("used_llm", "unknown"),
        chunk_count=len(result.get("retrieved_chunks", [])),
        latency_ms=result.get("latency_ms", latency),
        session_id=body.session_id,
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
            query = data.get("query", "").strip()

            if not query or len(query) < 3:
                await websocket.send_json({"error": "Query too short (min 3 chars)"})
                continue
            if len(query) > 1000:
                await websocket.send_json({"error": "Query too long (max 1000 chars)"})
                continue

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
            except Exception as exc:
                logger.warning("Retrieval failed in WS handler: {}", exc)
                chunks = []

            async for token in stream_answer(query, chunks):
                await websocket.send_text(token)

            session.add("user", query)
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
