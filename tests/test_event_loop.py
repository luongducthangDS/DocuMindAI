"""P0-3: việc đồng bộ nặng không được chạy trên event loop.

Chỉ có 1 uvicorn worker: một lời gọi đồng bộ (Gemini, parse PDF, embed khi index)
chạy thẳng trong handler async là đứng hình cả process — kể cả /health, và Render
restart container khi health timeout.

Cách đo: cho hàm đồng bộ ngủ BLOCK_S giây, trong lúc đó gọi /health. Nếu hàm chạy
trên event loop, /health phải chờ hết BLOCK_S; chạy trong thread thì về ngay.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import documents, health, query

BLOCK_S = 1.5
HEALTH_BUDGET_S = 0.2
ADMIN_KEY = "test-admin-key"


def _slow(started: threading.Event, result: object = None) -> Callable[..., object]:
    def fn(*_args: object, **_kwargs: object) -> object:
        started.set()
        time.sleep(BLOCK_S)
        return result
    return fn


@pytest.fixture
def client(monkeypatch) -> Iterator[TestClient]:
    monkeypatch.setenv("API_SECRET_KEY", ADMIN_KEY)
    from src.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setattr("src.rag.vector_backend.get_backend", lambda: object())
    monkeypatch.setattr("src.rag.vector_backend.count_chunks", lambda _b: 1151)
    health._reset_probe_cache()

    async def noop() -> None:
        return None

    app = FastAPI()
    for r in (documents.router, query.router, health.router):
        app.include_router(r)
    with patch("src.api.main.ensure_rag_initialized", noop), TestClient(app) as c:
        yield c
    health._reset_probe_cache()


def _health_latency_while(client: TestClient, started: threading.Event,
                          fire: Callable[[], None]) -> float:
    worker = threading.Thread(target=fire, daemon=True)
    worker.start()
    assert started.wait(5), "hàm chậm không được gọi tới — test dựng sai"
    t0 = time.perf_counter()
    r = client.get("/api/v1/health")
    elapsed = time.perf_counter() - t0
    worker.join(10)
    assert r.status_code == 200
    return elapsed


def _upload(client: TestClient) -> None:
    client.post("/api/v1/upload", headers={"x-admin-key": ADMIN_KEY},
                files={"file": ("a.pdf", b"%PDF-1.4 x", "application/pdf")})


class TestUploadDoesNotBlock:
    def test_load_pdf_runs_off_loop(self, client):
        started = threading.Event()
        with patch.object(documents, "load_pdf", _slow(started, None)):
            elapsed = _health_latency_while(client, started, lambda: _upload(client))
        assert elapsed < HEALTH_BUDGET_S

    def test_insert_nodes_runs_off_loop(self, client):
        started = threading.Event()
        doc = {"title": "t", "content": "Điều 1. x", "doc_type": "uploaded_pdf"}
        chunk = SimpleNamespace(text="Điều 1. x", metadata={}, is_valid=True)
        fake_index = SimpleNamespace(insert_nodes=_slow(started))
        with patch.object(documents, "load_pdf", return_value=doc), \
             patch.object(documents, "chunk_by_dieu", return_value=[chunk]), \
             patch.object(documents, "_rebuild_retriever", lambda _m: None), \
             patch("src.rag.retriever._active_index", fake_index):
            elapsed = _health_latency_while(client, started, lambda: _upload(client))
        assert elapsed < HEALTH_BUDGET_S


def _ws_turn(client: TestClient, payload: dict) -> Callable[[], None]:
    def fire() -> None:
        with client.websocket_connect("/api/v1/ws/loop-test") as ws:
            ws.send_json(payload)
            while '"done"' not in (ws.receive().get("text") or ""):
                pass
    return fire


async def _fake_stream(_query, _chunks):
    yield "ok"


@pytest.mark.parametrize("slow_target, payload, retrieved", [
    # Câu không dấu, ≥3 từ → đi qua _restore_diacritics.
    ("src.agent.graph._restore_diacritics",
     {"query": "muc luong toi thieu vung mot"}, ["c"]),
    # Truy hồi có ngữ cảnh rỗng → rơi xuống retrieve_direct_chroma.
    ("src.rag.retriever.retrieve_direct_chroma",
     {"query": "Mức lương tối thiểu vùng I?"}, []),
])
def test_ws_sync_calls_run_off_loop(client, slow_target, payload, retrieved):
    started = threading.Event()
    with patch(slow_target, _slow(started, "câu hỏi" if "diacritics" in slow_target else [])), \
         patch("src.rag.retriever.retrieve_with_context", return_value=retrieved), \
         patch.object(query, "stream_answer", _fake_stream), \
         patch.object(query, "_cited_sources", return_value=[]):
        elapsed = _health_latency_while(client, started, _ws_turn(client, payload))
    assert elapsed < HEALTH_BUDGET_S


def test_ws_contextualize_runs_off_loop(client):
    """Lượt thứ hai trên cùng socket có lịch sử → đi qua _contextualize_query."""
    started = threading.Event()
    slow_ctx = _slow(started, "câu hỏi đầy đủ")

    def fire() -> None:
        with client.websocket_connect("/api/v1/ws/loop-ctx") as ws:
            for q in ("Mức lương tối thiểu vùng I?", "Còn vùng II?"):
                ws.send_json({"query": q})
                while '"done"' not in (ws.receive().get("text") or ""):
                    pass

    with patch("src.agent.graph._contextualize_query", slow_ctx), \
         patch("src.rag.retriever.retrieve_with_context", return_value=["c"]), \
         patch.object(query, "stream_answer", _fake_stream), \
         patch.object(query, "_cited_sources", return_value=[]):
        elapsed = _health_latency_while(client, started, fire)
    assert elapsed < HEALTH_BUDGET_S
