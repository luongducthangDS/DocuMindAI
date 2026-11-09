"""Token / cost accounting: every LLM call is counted, priced, logged per request and rolled up."""
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import usage
from src.langfuse_otel import record_generation

ADMIN_KEY = "test-admin-key-usage"


@pytest.fixture
def db(tmp_path, monkeypatch):
    from src.config import get_settings

    monkeypatch.setenv("SQLITE_DB", str(tmp_path / "u.db"))
    monkeypatch.setenv("API_SECRET_KEY", ADMIN_KEY)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_meter_prices_known_models_and_flags_unknown(monkeypatch):
    monkeypatch.setenv("LLM_PRICES_JSON", '{"my-model": [1.0, 4.0]}')
    tok = usage.begin()
    usage.add("my-model", 1_000_000, 500_000)
    usage.add("gemini-2.5-flash", 2_000_000, 0)
    usage.add("Llama-3.3-70B", 10, 10)  # no price on file
    usage.add("my-model", 0, 0, failed=True)
    t = usage.end(tok)
    assert t["calls"] == 4 and t["failed_calls"] == 1
    assert t["cost_usd"] == pytest.approx(1.0 + 2.0 + 0.6)
    assert t["unpriced_models"] == ["Llama-3.3-70B"]


def test_record_generation_feeds_meter_even_without_langfuse(monkeypatch):
    monkeypatch.setattr("src.langfuse_otel._auth", lambda: None)
    tok = usage.begin()
    now = datetime.now(timezone.utc)
    record_generation("answer", "gemini-2.5-flash", "q", "a", now, now, prompt_tokens=1200, completion_tokens=300)
    t = usage.end(tok)
    assert (t["prompt_tokens"], t["completion_tokens"], t["calls"]) == (1200, 300, 1)


def test_no_meter_is_a_no_op():
    usage.add("gemini-2.5-flash", 5, 5)  # eval scripts call the generator outside a request


def test_summary_and_endpoint(db):
    for ms, cost in ((100, 0.001), (200, 0.002), (3000, 0.010)):
        usage.log_request("rest", "simple_qa", ms, {"calls": 1, "failed_calls": 0, "prompt_tokens": 100,
                                                    "completion_tokens": 50, "cost_usd": cost, "unpriced_models": []})
    usage.log_request("ws", "simple_qa", 500, {"calls": 0, "failed_calls": 0, "prompt_tokens": 0,
                                               "completion_tokens": 0, "cost_usd": 0.0, "unpriced_models": []}, error=True)
    s = usage.summary(24)
    assert s["requests"] == 4 and s["error_rate"] == 0.25
    assert s["latency_ms"]["p50"] in (200.0, 500.0) and s["latency_ms"]["p95"] == 3000.0
    assert s["cost_usd"] == pytest.approx(0.013) and s["tokens"] == {"prompt": 300, "completion": 150}

    from src.api.routes import metrics

    app = FastAPI()
    app.include_router(metrics.router)
    c = TestClient(app)
    assert c.get("/api/v1/metrics").status_code == 401
    body = c.get("/api/v1/metrics", headers={"X-Admin-Key": ADMIN_KEY}).json()
    assert body["requests"] == 4 and "tool_success_rate" in body["actions"]


def test_rest_query_logs_one_usage_row(db, monkeypatch):
    from unittest.mock import AsyncMock

    from src.api.routes import query

    async def fake_agent(**kw):
        now = datetime.now(timezone.utc)
        record_generation("answer", "gemini-2.5-flash", "q", "a", now, now, prompt_tokens=1000, completion_tokens=200)
        return {"answer": "Theo Điều 25 [1] ...", "sources": [], "retrieved_chunks": [], "intent": "simple_qa",
                "used_llm": "gemini-2.5-flash"}

    monkeypatch.setattr("src.api.main.ensure_rag_initialized", AsyncMock())
    monkeypatch.setattr(query, "run_agent", fake_agent)
    monkeypatch.setattr("src.langfuse_otel._auth", lambda: None)
    app = FastAPI()
    app.include_router(query.router)
    r = TestClient(app).post("/api/v1/query", json={"query": "Thời gian thử việc tối đa là bao lâu?", "session_id": "s1"})
    assert r.status_code == 200, r.text
    s = usage.summary(1)
    assert s["requests"] == 1 and s["tokens"]["prompt"] == 1000
    assert s["cost_usd"] == pytest.approx(1000 / 1e6 * 0.30 + 200 / 1e6 * 2.50)
    assert s["by_intent"]["simple_qa"]["requests"] == 1
