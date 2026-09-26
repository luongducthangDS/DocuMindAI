"""
Health check and metrics endpoints.
GET /api/v1/health   — service dependency status
GET /api/v1/metrics  — query stats
"""

from __future__ import annotations

import asyncio
import sqlite3
import time

from fastapi import APIRouter
from loguru import logger

from src.agent.memory import get_long_term_memory
from src.api.schemas import HealthResponse, MetricsResponse, ServiceStatus
from src.config import get_settings

router = APIRouter(prefix="/api/v1", tags=["observability"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check status of all downstream services — runs checks in parallel."""
    vector_store = await _check_vector_store()
    llm = _check_llm()
    sqlite = _check_sqlite()
    services = [vector_store, llm, sqlite]

    # Core serving path needs corpus retrieval + an LLM provider.
    # SQLite is optional on Render: it backs metrics/history best-effort persistence.
    required = [vector_store, llm]
    all_required_healthy = all(s.healthy for s in required)
    any_required_healthy = any(s.healthy for s in required)

    if all_required_healthy:
        overall = "ok"
    elif any_required_healthy:
        overall = "degraded"
    else:
        overall = "error"

    return HealthResponse(status=overall, services=services)


async def _check_vector_store() -> ServiceStatus:
    """Mở collection thật rồi đếm chunk — không suy ra sức khoẻ từ sự tồn tại của file.

    Bản cũ chỉ kiểm `chroma.sqlite3` có tồn tại và lớn hơn 1KB. Ngày 2026-09-21
    collection `documind_legal` hỏng metadata (KeyError '_type' ngay lúc mở), mọi
    query trả 0 chunk và trả lời "không tìm thấy văn bản", nhưng health vẫn báo
    xanh — kiểu hỏng tệ nhất, vì bảng điều khiển nói bình thường trong khi không
    câu hỏi nào được phục vụ.

    Một hàm cho cả hai provider: `vector_backend` sinh ra chính để chỗ gọi không
    phải phân nhánh chroma/qdrant, health check không nên là ngoại lệ.
    """
    name = "qdrant" if _provider() == "qdrant" else "chromadb"
    count, error = await asyncio.to_thread(_probe_vector_store)
    if error:
        return ServiceStatus(name=name, healthy=False, detail=error)
    if count <= 0:
        return ServiceStatus(
            name=name, healthy=False, detail="Collection mở được nhưng rỗng — chưa ingest corpus"
        )
    return ServiceStatus(name=name, healthy=True, detail=f"{count} chunks")


def _provider() -> str:
    return (get_settings().vector_store_provider or "chroma").lower()


# Đo trên máy dev 2026-09-21: một lần probe mất 4.1s vì CHROMA_HOST trỏ tới server
# không chạy — get_chroma_collection() chờ hết timeout HTTP rồi mới fallback local.
# HEALTHCHECK trong Dockerfile bỏ cuộc sau 5s, nên probe mỗi lần poll là hẹn giờ cho
# một health check chập chờn. Nhớ kết quả trong TTL ngắn: tình trạng corpus không
# đổi theo từng giây, và cái giá là sau khi ingest xong health còn báo hỏng thêm
# tối đa ngần này giây.
_PROBE_TTL_SECONDS = 15.0
_probe_cache: tuple[float, tuple[int, str]] | None = None


def _reset_probe_cache() -> None:
    """Dùng trong test — mỗi ca phải thấy tầng vector store mà chính nó dựng."""
    global _probe_cache
    _probe_cache = None


def _count_qdrant_readonly() -> int:
    """Đếm điểm mà KHÔNG đi qua get_backend().

    get_backend() gọi get_qdrant_client_and_collection(), và hàm đó tạo collection
    khi thiếu — kèm get_embedding_dim(), vốn embed một chuỗi rỗng để hỏi chiều
    vector. Nghĩa là một cú GET /health (endpoint public, ai gọi cũng được) sẽ dựng
    collection trên cluster và đốt một lệnh Gemini. Health check chỉ được đọc.
    """
    from qdrant_client import QdrantClient

    settings = get_settings()
    if not settings.qdrant_url:
        raise RuntimeError("VECTOR_STORE_PROVIDER=qdrant nhưng QDRANT_URL chưa được set")

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    return client.count(collection_name=settings.qdrant_collection, exact=True).count


def _probe_vector_store() -> tuple[int, str]:
    """(chunk_count, detail_lỗi). Chặn luồng — luôn gọi qua asyncio.to_thread.

    Chi tiết lỗi chỉ giữ tên exception, theo đúng lối của a664ea4: /health là
    endpoint public, nguyên văn lỗi chromadb/qdrant lộ đường dẫn máy chủ. Bản đầy
    đủ nằm ở log server.
    """
    global _probe_cache
    now = time.monotonic()
    if _probe_cache is not None and now - _probe_cache[0] < _PROBE_TTL_SECONDS:
        return _probe_cache[1]

    try:
        if _provider() == "qdrant":
            result = (_count_qdrant_readonly(), "")
        else:
            # Chroma: get_or_create dựng collection rỗng nếu thiếu — với file local
            # thì vô hại, và kết quả vẫn là đỏ ("rỗng"), đúng sự thật phục vụ.
            from src.rag.vector_backend import count_chunks, get_backend

            result = (count_chunks(get_backend()), "")
    except Exception as exc:
        logger.error("Vector store health probe failed: {}", exc)
        result = (0, f"Không mở được collection ({type(exc).__name__}) — xem log server")

    _probe_cache = (now, result)
    return result


def _check_llm() -> ServiceStatus:
    settings = get_settings()
    if settings.google_api_key:
        return ServiceStatus(name="llm_gemini", healthy=True, detail="API key configured")
    return ServiceStatus(name="llm", healthy=False, detail="GOOGLE_API_KEY chưa cấu hình")


def _check_sqlite() -> ServiceStatus:
    settings = get_settings()
    try:
        conn = sqlite3.connect(str(settings.sqlite_db))
        conn.execute("SELECT 1").fetchone()
        conn.close()
        return ServiceStatus(name="sqlite", healthy=True, detail="connected")
    except Exception as exc:
        # Cùng lối với a664ea4 và _probe_vector_store: /health là endpoint public,
        # nguyên văn lỗi sqlite kèm luôn đường dẫn file trên máy chủ.
        logger.error("SQLite health check failed: {}", exc)
        return ServiceStatus(
            name="sqlite", healthy=False, detail=f"{type(exc).__name__} — xem log server"
        )


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

        error_rate = get_long_term_memory().get_error_rate(window_minutes=60)

        return MetricsResponse(
            total_queries=total,
            avg_latency_ms=round(avg_lat, 1),
            error_rate=error_rate,
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
    count, _ = await asyncio.to_thread(_probe_vector_store)
    return count
