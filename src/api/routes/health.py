"""
Health check and metrics endpoints.
GET /api/v1/health   — service dependency status
GET /api/v1/metrics  — query stats
"""

from __future__ import annotations

import sqlite3
import time

from fastapi import APIRouter
from loguru import logger

from src.api.schemas import HealthResponse, MetricsResponse, ServiceStatus
from src.config import get_settings

router = APIRouter(prefix="/api/v1", tags=["observability"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check status of all downstream services — runs checks in parallel."""
    import asyncio

    chroma, redis = await asyncio.gather(
        _check_chroma(),
        _check_redis(),
    )
    services = [chroma, redis, _check_llm(), _check_sqlite()]

    all_healthy = all(s.healthy for s in services)
    any_healthy = any(s.healthy for s in services)

    if all_healthy:
        overall = "ok"
    elif any_healthy:
        overall = "degraded"
    else:
        overall = "error"

    return HealthResponse(status=overall, services=services)


async def _check_chroma() -> ServiceStatus:
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        settings = get_settings()
        client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        client.heartbeat()
        return ServiceStatus(name="chromadb", healthy=True, detail="HTTP connection OK")
    except Exception as exc:
        # Fallback: check if local persistent exists
        local_path = get_settings().data_dir / "chroma_db"
        if local_path.exists():
            return ServiceStatus(name="chromadb", healthy=True, detail="Local persistent mode")
        return ServiceStatus(name="chromadb", healthy=False, detail=str(exc)[:100])


async def _check_redis() -> ServiceStatus:
    try:
        import redis.asyncio as aioredis

        settings = get_settings()
        client = aioredis.from_url(settings.redis_url, socket_connect_timeout=1)
        await client.ping()
        await client.aclose()
        return ServiceStatus(name="redis", healthy=True, detail="PONG received")
    except Exception as exc:
        return ServiceStatus(name="redis", healthy=False, detail=str(exc)[:80])


def _check_llm() -> ServiceStatus:
    settings = get_settings()
    if settings.groq_api_key:
        return ServiceStatus(name="llm_groq", healthy=True, detail="API key configured")
    if settings.google_api_key:
        return ServiceStatus(name="llm_gemini", healthy=True, detail="API key configured")
    return ServiceStatus(name="llm", healthy=False, detail="No LLM API key configured")


def _check_sqlite() -> ServiceStatus:
    settings = get_settings()
    try:
        conn = sqlite3.connect(str(settings.sqlite_db))
        conn.execute("SELECT 1").fetchone()
        conn.close()
        return ServiceStatus(name="sqlite", healthy=True, detail=str(settings.sqlite_db))
    except Exception as exc:
        return ServiceStatus(name="sqlite", healthy=False, detail=str(exc)[:80])


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """Return query statistics from SQLite."""
    settings = get_settings()
    try:
        conn = sqlite3.connect(str(settings.sqlite_db))
        conn.row_factory = sqlite3.Row

        total = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
        avg_lat = conn.execute(
            "SELECT AVG(latency_ms) FROM query_log WHERE latency_ms > 0"
        ).fetchone()[0] or 0

        sessions = conn.execute("SELECT COUNT(DISTINCT session_id) FROM query_log").fetchone()[0]
        conn.close()

        # Corpus chunk count from ChromaDB (best effort)
        corpus_chunks = await _count_corpus_chunks()

        return MetricsResponse(
            total_queries=total,
            avg_latency_ms=round(avg_lat, 1),
            error_rate=0.0,  # TODO: track error count in DB
            active_sessions=sessions,
            corpus_chunks=corpus_chunks,
        )
    except Exception as exc:
        logger.warning("Metrics query failed: {}", exc)
        return MetricsResponse(
            total_queries=0,
            avg_latency_ms=0,
            error_rate=0,
            active_sessions=0,
            corpus_chunks=0,
        )


async def _count_corpus_chunks() -> int:
    import asyncio

    def _count_sync() -> int:
        try:
            from src.rag.embedder import get_chroma_collection
            _, col = get_chroma_collection()
            return col.count()
        except Exception:
            return 0

    return await asyncio.to_thread(_count_sync)
