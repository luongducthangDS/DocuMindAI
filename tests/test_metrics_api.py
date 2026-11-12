import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import health


@pytest.fixture
def metrics_client(tmp_path, monkeypatch):
    database = tmp_path / "metrics.sqlite"
    monkeypatch.setattr(health, "get_settings", lambda: SimpleNamespace(sqlite_db=database))
    monkeypatch.setattr(
        health, "get_long_term_memory", lambda: SimpleNamespace(get_error_rate=lambda **_: 0.25)
    )

    async def corpus_chunks():
        return 42

    monkeypatch.setattr(health, "_count_corpus_chunks", corpus_chunks)
    app = FastAPI()
    app.include_router(health.router)
    return TestClient(app), database


def test_metrics_api_reports_database_aggregates(metrics_client):
    client, database = metrics_client
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE query_log (session_id TEXT NOT NULL, latency_ms INTEGER)"
        )
        connection.executemany(
            "INSERT INTO query_log (session_id, latency_ms) VALUES (?, ?)",
            [("session-a", 100), ("session-a", 200), ("session-b", 0)],
        )

    response = client.get("/api/v1/metrics")

    assert response.status_code == 200
    assert response.json() == {
        "total_queries": 3,
        "avg_latency_ms": 150.0,
        "error_rate": 0.25,
        "active_sessions": 2,
        "corpus_chunks": 42,
    }


def test_metrics_api_returns_service_unavailable_when_database_is_invalid(metrics_client):
    client, _ = metrics_client

    response = client.get("/api/v1/metrics")

    assert response.status_code == 503
    assert response.json() == {"detail": "Metrics temporarily unavailable"}
