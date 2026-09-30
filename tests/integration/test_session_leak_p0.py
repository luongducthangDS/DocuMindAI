"""Q02 (P0) — lịch sử hội thoại không rò chéo tenant qua bước contextualize.

Ca gốc: session_id mặc định "default" và `_sessions` khoá bằng session_id trần.
Client A hỏi "Kế hoạch sa thải 50 nhân sự", client B (tenant khác, không gửi
session_id) hỏi "Chi tiết thế nào?" → _contextualize_query của B nhận lịch sử
của A và viết lại câu hỏi thành câu về kế hoạch sa thải của A.

Gemini giả ở đây CỐ TÌNH rò rỉ: câu viết lại = mọi dòng "Người dùng:" trong
lịch sử + câu hỏi. Nên nếu lịch sử của A tới được prompt của B, "sa thải" chắc
chắn xuất hiện — test đối chứng (cùng tenant, cùng session) chứng minh điều đó.
"""

from __future__ import annotations

import json
import re
import uuid
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import query as query_route

ACME_KEY, GLOBEX_KEY = "demo-acme-0000", "demo-globex-0000"
Q_A = "Kế hoạch sa thải 50 nhân sự"
Q_B = "Chi tiết thế nào?"
LEAK_MARKERS = ("sa thải", "50 nhân sự")


@pytest.fixture(autouse=True)
def fresh_sessions(monkeypatch):
    monkeypatch.setattr(query_route, "_sessions", type(query_route._sessions)())


@pytest.fixture
def leaky_gemini(fake_gemini):
    def responder(prompt, model):
        history = re.findall(r"^Người dùng: (.*)$", prompt, re.M)
        question = re.search(r"Câu hỏi[^:\n]*:\s*(.*)", prompt)
        return " ".join([*history, question.group(1) if question else prompt[-80:]])

    fake_gemini.responder = responder
    return fake_gemini


def _contains_a(text: str) -> bool:
    return any(m in text.lower() for m in LEAK_MARKERS)


# ── REST ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def rest(leaky_gemini):
    """/api/v1/query thật tới tận history + _contextualize_query thật; phần còn
    lại của graph (retrieve/answer) được thay bằng một run_agent tối giản."""
    from src.agent.graph import _contextualize_query

    rewritten: dict[str, str] = {}

    async def fake_run_agent(*, query, session_id, history, as_of_date, retrieval_ctx):
        rewritten[retrieval_ctx.tenant_id] = _contextualize_query(query, history) if history else query
        return {"answer": "Đã ghi nhận.", "sources": [], "retrieved_chunks": []}

    async def noop():
        return None

    app = FastAPI()
    app.include_router(query_route.router)
    with patch("src.api.main.ensure_rag_initialized", noop), \
         patch.object(query_route, "run_agent", fake_run_agent):
        yield TestClient(app), rewritten


def test_rest_default_session_does_not_leak_across_tenants(rest, leaky_gemini):
    client, rewritten = rest
    assert client.post("/api/v1/query", json={"query": Q_A}, headers={"x-api-key": ACME_KEY}).status_code == 200
    assert client.post("/api/v1/query", json={"query": Q_B}, headers={"x-api-key": GLOBEX_KEY}).status_code == 200

    assert not _contains_a(rewritten["globex"]), rewritten["globex"]
    assert not any(_contains_a(p) for p in (c.prompt for c in leaky_gemini.calls)), \
        "lịch sử của tenant A đã tới Gemini trong lượt của tenant B"


def test_rest_same_explicit_session_id_is_still_isolated_per_tenant(rest):
    client, rewritten = rest
    body_a = {"query": Q_A, "session_id": "shared"}
    body_b = {"query": Q_B, "session_id": "shared"}
    client.post("/api/v1/query", json=body_a, headers={"x-api-key": ACME_KEY})
    client.post("/api/v1/query", json=body_b, headers={"x-api-key": GLOBEX_KEY})
    client.post("/api/v1/query", json=body_b)  # ẩn danh, cùng session_id
    assert not _contains_a(rewritten["globex"])
    assert not _contains_a(rewritten["public"])


def test_control_same_tenant_same_session_does_carry_context(rest):
    """Đối chứng: fake Gemini thật sự rò khi được đưa lịch sử — test trên không rỗng."""
    client, rewritten = rest
    client.post("/api/v1/query", json={"query": Q_A, "session_id": "s1"}, headers={"x-api-key": ACME_KEY})
    client.post("/api/v1/query", json={"query": Q_B, "session_id": "s1"}, headers={"x-api-key": ACME_KEY})
    assert _contains_a(rewritten["acme"])


def test_rest_missing_session_id_gets_its_own_session(rest):
    client, _ = rest
    r1 = client.post("/api/v1/query", json={"query": Q_A})
    r2 = client.post("/api/v1/query", json={"query": Q_B})
    assert r1.json()["session_id"] != r2.json()["session_id"]
    assert "default" not in (r1.json()["session_id"], r2.json()["session_id"])


# ── WebSocket: tenant đổi theo từng lượt (api_key trong message) ─────────────

def _ws_turns(session_id: str, turns: list[dict]) -> None:
    async def noop():
        return None

    app = FastAPI()
    app.include_router(query_route.router)
    with patch("src.api.main.ensure_rag_initialized", noop), \
         patch("src.rag.retriever.retrieve_with_context", return_value=[]), \
         patch("src.rag.retriever.retrieve_direct_chroma", return_value=[]):
        with TestClient(app).websocket_connect(f"/api/v1/ws/{session_id}") as ws:
            for payload in turns:
                ws.send_json(payload)
                while True:
                    text = ws.receive().get("text") or ""
                    try:
                        msg = json.loads(text)
                    except ValueError:
                        continue
                    if isinstance(msg, dict) and (msg.get("done") or "error" in msg):
                        break


def test_ws_turn_under_another_key_never_sees_previous_tenants_history(leaky_gemini):
    sid = f"p0-{uuid.uuid4().hex[:12]}"
    _ws_turns(sid, [{"query": Q_A, "api_key": ACME_KEY}])
    leaky_gemini.calls.clear()
    _ws_turns(sid, [{"query": Q_B, "api_key": GLOBEX_KEY}])  # cùng URL session, khác tenant
    assert not any(_contains_a(c.prompt) for c in leaky_gemini.calls)


def test_ws_control_same_tenant_does_contextualize(leaky_gemini):
    sid = f"p0-{uuid.uuid4().hex[:12]}"
    _ws_turns(sid, [{"query": Q_A, "api_key": ACME_KEY}, {"query": Q_B, "api_key": ACME_KEY}])
    assert any(_contains_a(c.prompt) for c in leaky_gemini.calls)
