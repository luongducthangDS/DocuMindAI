"""Human-in-the-loop actions: proposed by the agent, executed once, only after an admin approves."""
import sqlite3
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agent import actions

ADMIN_KEY = "test-admin-key-actions"


@pytest.fixture
def db(tmp_path, monkeypatch):
    from src.config import get_settings

    monkeypatch.setenv("SQLITE_DB", str(tmp_path / "t.db"))
    monkeypatch.setenv("API_SECRET_KEY", ADMIN_KEY)
    get_settings.cache_clear()
    yield tmp_path / "t.db"
    get_settings.cache_clear()


def _hr_rows(path) -> int:
    with sqlite3.connect(path) as c:
        return c.execute("SELECT COUNT(*) FROM hr_requests").fetchone()[0]


def test_proposal_waits_then_executes_once(db):
    row = actions.propose("create_hr_request", actions.proposal_from_query("Tạo yêu cầu nghỉ phép 3 ngày"), "acme", "s1")
    assert row["status"] == "pending" and _hr_rows(db) == 0

    done = actions.approve(row["id"], "chi.hr")
    assert done["status"] == "executed" and done["decided_by"] == "chi.hr"
    assert done["result"]["request_id"].startswith("HR-") and _hr_rows(db) == 1

    assert actions.approve(row["id"], "someone") is None   # already decided
    assert actions.reject(row["id"], "someone") is None
    assert _hr_rows(db) == 1


def test_concurrent_approvals_execute_once(db):
    row = actions.propose("create_hr_request", actions.proposal_from_query("Tạo yêu cầu cấp lại sổ BHXH"), "acme", "s1")
    barrier, results = threading.Barrier(8), []

    def go():
        barrier.wait()
        results.append(actions.approve(row["id"], "x"))

    threads = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(r is not None for r in results) == 1
    assert _hr_rows(db) == 1


def test_handler_failure_is_recorded(db, monkeypatch):
    def boom(conn, args):
        raise RuntimeError("HRM unavailable")

    monkeypatch.setitem(actions.ACTIONS, "create_hr_request",
                        actions.Action("create_hr_request", "", True, boom))
    row = actions.propose("create_hr_request", {"department": "nhan_su", "title": "t", "body": "b"}, "acme", "s")
    out = actions.approve(row["id"], "x")
    assert out["status"] == "failed" and "HRM unavailable" in out["error"]
    assert actions.stats()["tool_success_rate"] == 0.0


def test_department_guess():
    assert actions.guess_department("tạo yêu cầu điều chỉnh lương tháng 10") == "ke_toan"
    assert actions.guess_department("gửi đơn khiếu nại kỷ luật") == "phap_che"
    assert actions.guess_department("đăng ký nghỉ phép") == "nhan_su"


def test_anonymous_cannot_propose(db):
    from src.agent.graph import action_node
    from src.rag.context import PUBLIC_CONTEXT

    out = action_node({"query": "tạo yêu cầu nghỉ phép", "retrieval_ctx": PUBLIC_CONTEXT, "steps": []})
    assert "API key" in out["answer"]
    assert actions.list_actions() == []

    out = action_node({"query": "tạo yêu cầu nghỉ phép", "retrieval_ctx": SimpleNamespace(tenant_id="acme"), "steps": []})
    assert actions.list_actions("pending")[0]["tenant_id"] == "acme"


def test_api_requires_admin_and_reports_conflicts(db):
    from src.api.routes import actions as routes

    app = FastAPI()
    app.include_router(routes.router)
    c = TestClient(app)
    row = actions.propose("create_hr_request", actions.proposal_from_query("tạo ticket cấp laptop"), "acme", "s")

    assert c.get("/api/v1/actions").status_code == 401
    assert c.post(f"/api/v1/actions/{row['id']}/approve").status_code == 401
    h = {"X-Admin-Key": ADMIN_KEY, "X-Approver": "lan.nguyen<script>"}
    assert [a["id"] for a in c.get("/api/v1/actions?status=pending", headers=h).json()] == [row["id"]]
    r = c.post(f"/api/v1/actions/{row['id']}/approve", headers=h)
    assert r.status_code == 200 and r.json()["decided_by"] == "lan.nguyenscript"
    assert c.post(f"/api/v1/actions/{row['id']}/approve", headers=h).status_code == 409
    assert c.post("/api/v1/actions/nope/reject", headers=h).status_code == 404
    assert c.get("/api/v1/actions/stats", headers=h).json()["tool_success_rate"] == 1.0
