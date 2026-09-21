"""
Health check phải chạm thật vào collection.

Ca gốc (2026-09-21): `documind_legal` hỏng metadata, chromadb ném KeyError '_type'
ngay lúc mở, mọi query trả 0 chunk — nhưng `/health` vẫn báo `chromadb: healthy`
vì nó chỉ kiểm `chroma.sqlite3` có tồn tại và lớn hơn 1KB. 59MB dữ liệu nằm đó,
không đọc được đoạn nào, bảng điều khiển vẫn xanh.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import health as health_module


@pytest.fixture(autouse=True)
def clear_probe_cache():
    """Probe nhớ kết quả 15s — không dọn thì ca sau đọc trúng kết quả của ca trước."""
    health_module._reset_probe_cache()
    yield
    health_module._reset_probe_cache()


@pytest.fixture
def client() -> TestClient:
    """App tối thiểu — chỉ router health, không chạy lifespan/RAG init của main."""
    app = FastAPI()
    app.include_router(health_module.router)
    return TestClient(app)


def _probe(monkeypatch, *, count: int = 0, exc: Exception | None = None):
    """Giả lập tầng vector store ở đúng chỗ health import nó."""
    def fake_get_backend():
        if exc is not None:
            raise exc
        return object()

    monkeypatch.setattr("src.rag.vector_backend.get_backend", fake_get_backend)
    monkeypatch.setattr("src.rag.vector_backend.count_chunks", lambda _b: count)


class TestVectorStoreProbe:
    def test_healthy_when_collection_has_chunks(self, monkeypatch):
        _probe(monkeypatch, count=1146)
        status = asyncio.run(health_module._check_vector_store())
        assert status.healthy
        assert "1146" in status.detail

    def test_unhealthy_when_collection_cannot_be_opened(self, monkeypatch):
        """Chính xác lỗi đã xảy ra: chromadb ném KeyError('_type') lúc mở collection."""
        _probe(monkeypatch, exc=KeyError("_type"))
        status = asyncio.run(health_module._check_vector_store())
        assert not status.healthy
        assert "KeyError" in status.detail

    def test_unhealthy_when_collection_is_empty(self, monkeypatch):
        """Mở được nhưng rỗng cũng là chết — chưa ingest thì không phục vụ nổi câu nào."""
        _probe(monkeypatch, count=0)
        status = asyncio.run(health_module._check_vector_store())
        assert not status.healthy
        assert "rỗng" in status.detail

    def test_error_detail_hides_internal_paths(self, monkeypatch):
        """/health là endpoint public — chỉ tên exception, theo lối a664ea4."""
        _probe(monkeypatch, exc=RuntimeError(r"failed at D:\github\DocuMindAI\data\chroma_db"))
        status = asyncio.run(health_module._check_vector_store())
        assert "chroma_db" not in status.detail
        assert "D:" not in status.detail
        assert "RuntimeError" in status.detail

    def test_names_service_after_active_provider(self, monkeypatch):
        _probe(monkeypatch, count=5)
        monkeypatch.setattr(health_module, "_provider", lambda: "qdrant")
        assert asyncio.run(health_module._check_vector_store()).name == "qdrant"


class TestHealthEndpoint:
    def test_broken_collection_is_not_reported_ok(self, client, monkeypatch):
        """Bánh cóc chống tái phát: corpus chết thì status tuyệt đối không được 'ok'."""
        _probe(monkeypatch, exc=KeyError("_type"))
        body = client.get("/api/v1/health").json()
        assert body["status"] != "ok"
        vector = next(s for s in body["services"] if s["name"] in {"chromadb", "qdrant"})
        assert not vector["healthy"]

    def test_ok_when_corpus_serves(self, client, monkeypatch):
        _probe(monkeypatch, count=1146)
        body = client.get("/api/v1/health").json()
        assert body["status"] == "ok"

    def test_endpoint_stays_200_so_orchestrator_does_not_restart_loop(self, client, monkeypatch):
        """Render/Docker HEALTHCHECK đọc HTTP status. Corpus hỏng là việc phải sửa
        bằng ingest, không phải bằng restart — giữ 200, nói sự thật trong body."""
        _probe(monkeypatch, exc=KeyError("_type"))
        assert client.get("/api/v1/health").status_code == 200


class TestProbeCost:
    def test_repeat_polls_do_not_reopen_the_collection(self, client, monkeypatch):
        """Mỗi lần mở tốn ~4s khi CHROMA_HOST trỏ vào server không chạy; Dockerfile
        HEALTHCHECK bỏ cuộc sau 5s. Poll dồn dập phải đọc lại kết quả đã nhớ."""
        calls = []

        def counting_backend():
            calls.append(1)
            return object()

        monkeypatch.setattr("src.rag.vector_backend.get_backend", counting_backend)
        monkeypatch.setattr("src.rag.vector_backend.count_chunks", lambda _b: 1146)

        for _ in range(3):
            assert client.get("/api/v1/health").json()["status"] == "ok"
        assert len(calls) == 1, f"mở lại collection {len(calls)} lần cho 3 lần poll"

    def test_cache_expires_so_recovery_is_visible(self, monkeypatch):
        """Cache không được biến thành trí nhớ vĩnh viễn: ingest xong phải xanh lại."""
        _probe(monkeypatch, exc=KeyError("_type"))
        assert asyncio.run(health_module._check_vector_store()).healthy is False

        monkeypatch.setattr(health_module, "_PROBE_TTL_SECONDS", 0.0)
        _probe(monkeypatch, count=1146)
        assert asyncio.run(health_module._check_vector_store()).healthy is True


class TestQdrantProbeIsReadOnly:
    """get_backend() cho Qdrant gọi get_or_create + get_embedding_dim (một lệnh embed
    thật). Nếu health đi đường đó thì mỗi cú GET /health vào cluster trắng sẽ tạo
    collection và đốt quota Gemini — endpoint public không được phép có tác dụng phụ."""

    def test_counts_without_creating_collection(self, monkeypatch):
        monkeypatch.setattr(health_module, "_provider", lambda: "qdrant")
        monkeypatch.setenv("QDRANT_URL", "https://cluster.example")
        from src.config import get_settings

        get_settings.cache_clear()

        def explode():
            raise AssertionError("health không được gọi get_backend() trên Qdrant")

        monkeypatch.setattr("src.rag.vector_backend.get_backend", lambda: explode())

        class FakeClient:
            def __init__(self, **kw): pass
            def count(self, collection_name, exact): return type("R", (), {"count": 1146})()

        monkeypatch.setattr("qdrant_client.QdrantClient", FakeClient)
        status = asyncio.run(health_module._check_vector_store())
        assert status.healthy and status.name == "qdrant"
        get_settings.cache_clear()

    def test_missing_url_is_reported_unhealthy(self, monkeypatch):
        monkeypatch.setattr(health_module, "_provider", lambda: "qdrant")
        monkeypatch.setenv("QDRANT_URL", "")
        from src.config import get_settings

        get_settings.cache_clear()
        status = asyncio.run(health_module._check_vector_store())
        assert not status.healthy
        assert "RuntimeError" in status.detail
        get_settings.cache_clear()


class TestSqliteCheck:
    """_check_sqlite cũng là một phần của /health public — không lộ đường dẫn file."""

    def test_healthy_detail_has_no_path(self, monkeypatch):
        status = health_module._check_sqlite()
        assert status.healthy
        assert "documind.db" not in status.detail
        assert "\\" not in status.detail and "/" not in status.detail

    def test_error_detail_hides_internal_paths(self, monkeypatch):
        def boom(*_args, **_kwargs):
            raise sqlite3.OperationalError(r"unable to open D:\github\DocuMindAI\data\documind.db")

        monkeypatch.setattr(health_module.sqlite3, "connect", boom)
        status = health_module._check_sqlite()
        assert not status.healthy
        assert "documind.db" not in status.detail
        assert "D:" not in status.detail
        assert "OperationalError" in status.detail
