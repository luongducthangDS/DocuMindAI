"""P0-4: lịch sử hội thoại cô lập theo tenant, và cache session có trần.

Ca gốc: `_sessions` khoá bằng session_id trần, mặc định "default". Mọi client REST
không gửi session_id dùng chung một lịch sử — và lịch sử đó vào thẳng prompt, nên
câu trả lời `confidential` của tenant này lọt vào ngữ cảnh của người ẩn danh.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import query
from src.api.schemas import QueryRequest

ACME_KEY = "demo-acme-0000"
GLOBEX_KEY = "demo-globex-0000"


@pytest.fixture(autouse=True)
def fresh_sessions(monkeypatch):
    monkeypatch.setattr(query, "_sessions", type(query._sessions)())


# SQLite sau get_long_term_memory() là singleton dùng chung cả suite — id cố định như
# "s1" có thể đã bị test khác ghi vào.
SID = f"iso-{uuid.uuid4().hex[:8]}"


class TestSessionKey:
    def test_same_session_id_different_tenants_are_distinct(self):
        a = query._get_session("acme", SID)
        b = query._get_session("public", SID)
        a.add("user", "bí mật của acme")
        assert a is not b
        assert b.as_messages() == []

    def test_isolation_survives_rehydration_from_sqlite(self, monkeypatch):
        query._get_session("globex", SID).add("assistant", "tài liệu confidential")
        monkeypatch.setattr(query, "_sessions", type(query._sessions)())  # giả lập restart
        assert query._get_session("public", SID).as_messages() == []
        assert query._get_session("globex", SID).as_messages()[0]["content"] == "tài liệu confidential"

    def test_same_tenant_same_id_is_reused(self):
        assert query._get_session("acme", SID) is query._get_session("acme", SID)


class TestLruBound:
    def test_evicts_least_recently_used(self, monkeypatch):
        monkeypatch.setattr(query, "MAX_SESSIONS", 2)
        s1 = query._get_session("public", "s1")
        query._get_session("public", "s2")
        query._get_session("public", "s1")          # s1 thành mới dùng nhất
        query._get_session("public", "s3")          # đẩy s2 ra
        assert list(query._sessions) == [("public", "s1"), ("public", "s3")]
        assert query._get_session("public", "s1") is s1

    def test_never_exceeds_cap(self, monkeypatch):
        monkeypatch.setattr(query, "MAX_SESSIONS", 5)
        for i in range(20):
            query._get_session("public", f"s{i}")
        assert len(query._sessions) == 5

    def test_default_cap_is_1000(self):
        assert query.MAX_SESSIONS == 1000


class TestSchemaSessionId:
    def test_missing_session_id_gets_fresh_uuid(self):
        a = QueryRequest(query="Lương tối thiểu?")
        b = QueryRequest(query="Lương tối thiểu?")
        assert a.session_id != b.session_id
        uuid.UUID(a.session_id)  # ném nếu không phải uuid

    def test_no_shared_default(self):
        assert QueryRequest(query="xin chào").session_id != "default"


@pytest.fixture
def client() -> Iterator[TestClient]:
    async def noop() -> None:
        return None

    app = FastAPI()
    app.include_router(query.router)
    with patch("src.api.main.ensure_rag_initialized", noop):
        yield TestClient(app)


def test_two_tenants_on_rest_never_see_each_others_history(client):
    seen: dict[str, list[dict]] = {}

    async def fake_run_agent(*, query, session_id, history, as_of_date, retrieval_ctx):
        seen[retrieval_ctx.tenant_id] = list(history)
        return {"answer": f"trả lời riêng cho {retrieval_ctx.tenant_id}", "sources": [],
                "retrieved_chunks": []}

    with patch.object(query, "run_agent", fake_run_agent):
        # id duy nhất: SQLite của get_long_term_memory() là singleton dùng chung cả suite.
        body = {"query": "Chính sách cắt giảm lao động?", "session_id": f"shared-{uuid.uuid4().hex[:8]}"}
        assert client.post("/api/v1/query", json=body,
                           headers={"x-api-key": GLOBEX_KEY}).status_code == 200
        assert client.post("/api/v1/query", json=body).status_code == 200            # ẩn danh
        assert client.post("/api/v1/query", json=body,
                           headers={"x-api-key": ACME_KEY}).status_code == 200

    assert seen == {"globex": [], "public": [], "acme": []}


def test_rest_without_session_id_returns_generated_id(client):
    async def fake_run_agent(**_kw):
        return {"answer": "ok", "sources": [], "retrieved_chunks": []}

    with patch.object(query, "run_agent", fake_run_agent):
        r1 = client.post("/api/v1/query", json={"query": "Lương tối thiểu?"})
        r2 = client.post("/api/v1/query", json={"query": "Lương tối thiểu?"})
    assert r1.json()["session_id"] != r2.json()["session_id"]
