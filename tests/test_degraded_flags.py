"""P1.2: trạng thái suy giảm phải lộ ra response và trace, không chỉ nằm trong log.

Hết quota embed → nhánh dense bị bỏ, chỉ còn BM25; hết mọi cặp Gemini → trả lời
bằng trích nguyên văn. Cả hai vẫn ra câu trả lời có trích dẫn trông như bình thường,
nên người dùng (và dashboard) không có cách nào biết — trừ khi gắn cờ.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.rag.context import degraded_flags, mark_degraded, reset_degraded, start_degraded
from src.rag.embedder import EmbeddingUnavailable
from src.rag.retriever import RetrievedChunk, _DenseOrSkip


class TestFlagPlumbing:
    def test_mark_outside_a_turn_is_noop(self):
        mark_degraded("dense_skipped")
        assert degraded_flags() == []

    def test_flags_are_deduplicated(self):
        token = start_degraded()
        try:
            mark_degraded("dense_skipped")
            mark_degraded("dense_skipped")
            assert degraded_flags() == ["dense_skipped"]
        finally:
            reset_degraded(token)

    async def test_flag_set_in_worker_thread_reaches_the_turn(self):
        """Retriever chạy qua to_thread — cờ đặt ở thread phải về được lượt gọi."""
        token = start_degraded()
        try:
            await asyncio.to_thread(mark_degraded, "dense_skipped")
            assert degraded_flags() == ["dense_skipped"]
        finally:
            reset_degraded(token)

    def test_turns_do_not_leak_into_each_other(self):
        token = start_degraded()
        mark_degraded("extractive_fallback")
        reset_degraded(token)
        token = start_degraded()
        try:
            assert degraded_flags() == []
        finally:
            reset_degraded(token)


class TestSources:
    def test_dense_skip_marks_flag(self):
        class Quota:
            def retrieve(self, _q):
                raise EmbeddingUnavailable("quota")

        token = start_degraded()
        try:
            assert _DenseOrSkip(Quota()).retrieve("q") == []
            assert degraded_flags() == ["dense_skipped"]
        finally:
            reset_degraded(token)

    def test_extractive_answer_marks_flag(self):
        from src.rag import generator

        chunk = RetrievedChunk(text="Điều 90. Tiền lương ...", score=0.9,
                               metadata={"title": "BLLĐ", "dieu_header": "Điều 90"})
        token = start_degraded()
        try:
            with patch.object(generator, "_call_gemini", side_effect=RuntimeError("503 unavailable")):
                out = generator.generate_answer("Tiền lương là gì?", [chunk])
            assert out["used_llm"] == "extractive_fallback"
            assert degraded_flags() == ["extractive_fallback"]
        finally:
            reset_degraded(token)


class _FakeGraph:
    async def ainvoke(self, state):
        await asyncio.to_thread(mark_degraded, "dense_skipped")  # như node sync của LangGraph
        return {**state, "answer": "trả lời"}


async def test_run_agent_returns_flags_and_tags_trace():
    from src.agent import graph

    tags = {}

    def fake_end_trace(*_a, extra_tags=None, **_kw):
        tags["value"] = extra_tags

    with patch.object(graph, "get_graph", return_value=_FakeGraph()), \
         patch.object(graph, "end_trace", fake_end_trace):
        result = await graph.run_agent("Lương tối thiểu vùng I?")
    assert result["degraded"] == ["dense_skipped"]
    assert "degraded:dense_skipped" in tags["value"]


def test_rest_response_carries_flags():
    from src.api.routes import query

    async def noop():
        return None

    async def fake_run_agent(**_kw):
        return {"answer": "ok", "sources": [], "retrieved_chunks": [], "degraded": ["dense_skipped"]}

    app = FastAPI()
    app.include_router(query.router)
    with patch("src.api.main.ensure_rag_initialized", noop), \
         patch.object(query, "run_agent", fake_run_agent):
        r = TestClient(app).post("/api/v1/query", json={"query": "Lương tối thiểu?"})
    assert r.json()["degraded"] == ["dense_skipped"]


def test_ws_done_frame_carries_flags():
    from src.api.routes import query

    async def noop():
        return None

    def retrieve_marking(_q, _ctx):
        mark_degraded("dense_skipped")
        return [SimpleNamespace(score=1.0)]

    async def fake_stream(_query, _chunks):
        yield "ok"

    app = FastAPI()
    app.include_router(query.router)
    with patch("src.api.main.ensure_rag_initialized", noop), \
         patch("src.rag.retriever.retrieve_with_context", retrieve_marking), \
         patch.object(query, "stream_answer", fake_stream), \
         patch.object(query, "_cited_sources", return_value=[]):
        with TestClient(app).websocket_connect(f"/api/v1/ws/d-{uuid.uuid4().hex[:8]}") as ws:
            ws.send_json({"query": "Mức lương tối thiểu vùng I?"})
            while True:
                text = ws.receive().get("text") or ""
                if '"done"' in text:
                    break
    assert json.loads(text)["degraded"] == ["dense_skipped"]
