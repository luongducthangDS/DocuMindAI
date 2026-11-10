"""
Business actions the agent may PROPOSE but never execute on its own.

Flow: agent -> propose() -> row 'pending' -> a human with the admin key approves or rejects via
/api/v1/actions -> execute once. Approval can arrive hours later or after a restart, so state lives
in SQLite, not in the LangGraph run. Every transition is a conditional UPDATE (WHERE status = ...),
so two approvers clicking at once cannot execute an action twice.

Handlers write to `hr_requests`, a stand-in for the real system of record (ticketing / HRM).
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from loguru import logger

from src.config import get_settings

DEPARTMENTS = ("nhan_su", "ke_toan", "phap_che")
_DEPT_HINTS = {"ke_toan": ("lương", "thuế", "thanh toán", "kế toán"),
               "phap_che": ("hợp đồng", "tranh chấp", "kỷ luật", "pháp chế")}


@dataclass(frozen=True)
class Action:
    name: str
    description: str
    requires_approval: bool
    handler: Callable[[sqlite3.Connection, dict], dict]


def _create_hr_request(conn: sqlite3.Connection, args: dict) -> dict:
    rid = f"HR-{uuid.uuid4().hex[:8].upper()}"
    conn.execute(
        "INSERT INTO hr_requests (id, created_at, tenant_id, department, title, body) VALUES (?, ?, ?, ?, ?, ?)",
        (rid, _now(), args["tenant_id"], args["department"], args["title"], args["body"]),
    )
    return {"request_id": rid, "department": args["department"]}


ACTIONS: dict[str, Action] = {
    "create_hr_request": Action(
        "create_hr_request", "Tạo yêu cầu gửi phòng ban (nhân sự / kế toán / pháp chế)", True, _create_hr_request
    ),
}

ACTION_PHRASES = ("tạo yêu cầu", "gửi yêu cầu", "tạo ticket", "mở ticket", "gửi đơn", "nộp đơn", "đăng ký nghỉ")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db() -> Path:
    return Path(get_settings().sqlite_db)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db()), check_same_thread=False, isolation_level=None)  # autocommit; BEGIN where needed
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS agent_actions (id TEXT PRIMARY KEY, created_at TEXT, tenant_id TEXT, "
        "session_id TEXT, action TEXT, args TEXT, status TEXT, decided_by TEXT, decided_at TEXT, "
        "result TEXT, error TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS hr_requests (id TEXT PRIMARY KEY, created_at TEXT, tenant_id TEXT, "
        "department TEXT, title TEXT, body TEXT)"
    )
    return conn


@contextmanager
def _conn():
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


def wants_action(query: str) -> bool:
    q = query.lower()
    return any(p in q for p in ACTION_PHRASES)


def guess_department(text: str) -> str:
    t = text.lower()
    for dept, hints in _DEPT_HINTS.items():
        if any(h in t for h in hints):
            return dept
    return "nhan_su"


def propose(action: str, args: dict, tenant_id: str, session_id: str) -> dict:
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action}")
    aid = str(uuid.uuid4())
    status = "pending" if ACTIONS[action].requires_approval else "approved"
    with _conn() as conn:
        conn.execute(
            "INSERT INTO agent_actions (id, created_at, tenant_id, session_id, action, args, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (aid, _now(), tenant_id, session_id, action, json.dumps(args, ensure_ascii=False), status),
        )
    logger.info("Action proposed {} {} status={}", aid, action, status)
    if status == "approved":
        return execute(aid, decided_by="auto")
    return get(aid)


def get(aid: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM agent_actions WHERE id = ?", (aid,)).fetchone()
    return _row(row) if row else None


def list_actions(status: str | None = None, limit: int = 100) -> list[dict]:
    q, p = "SELECT * FROM agent_actions", []
    if status:
        q, p = q + " WHERE status = ?", [status]
    with _conn() as conn:
        rows = conn.execute(q + " ORDER BY created_at DESC LIMIT ?", (*p, min(limit, 500))).fetchall()
    return [_row(r) for r in rows]


def _transition(aid: str, frm: str, to: str, decided_by: str) -> bool:
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE agent_actions SET status = ?, decided_by = ?, decided_at = ? WHERE id = ? AND status = ?",
            (to, decided_by, _now(), aid, frm),
        )
        return cur.rowcount == 1


def reject(aid: str, decided_by: str) -> dict | None:
    if not _transition(aid, "pending", "rejected", decided_by):
        return None
    return get(aid)


def approve(aid: str, decided_by: str) -> dict | None:
    if not _transition(aid, "pending", "approved", decided_by):
        return None  # not pending: already decided, or unknown id
    return execute(aid, decided_by)


def execute(aid: str, decided_by: str) -> dict:
    """approved -> executing (claimed atomically) -> executed | failed. Runs at most once."""
    if not _transition(aid, "approved", "executing", decided_by):
        return get(aid)
    row = get(aid)
    act = ACTIONS[row["action"]]
    with _conn() as conn:
        try:
            conn.execute("BEGIN")
            result = act.handler(conn, row["args"] | {"tenant_id": row["tenant_id"]})
            conn.execute("UPDATE agent_actions SET status = 'executed', result = ? WHERE id = ?",
                         (json.dumps(result, ensure_ascii=False), aid))
            conn.execute("COMMIT")
        except Exception as exc:  # handler failure is recorded, never retried silently
            conn.execute("ROLLBACK")
            conn.execute("UPDATE agent_actions SET status = 'failed', error = ? WHERE id = ?", (str(exc)[:500], aid))
            logger.error("Action {} failed: {}", aid, exc)
    return get(aid)


def stats() -> dict:
    """Tool-call success rate = executed / (executed + failed); approval outcomes for the dashboard."""
    with _conn() as conn:
        counts = dict(conn.execute("SELECT status, COUNT(*) FROM agent_actions GROUP BY status").fetchall())
    done = counts.get("executed", 0) + counts.get("failed", 0)
    return {"by_status": counts, "tool_success_rate": counts.get("executed", 0) / done if done else None}


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["args"] = json.loads(d["args"] or "{}")
    d["result"] = json.loads(d["result"]) if d.get("result") else None
    return d


def proposal_from_query(query: str) -> dict:
    """Arguments for create_hr_request from the user's own words (no LLM: nothing to hallucinate)."""
    title = re.sub(r"\s+", " ", query).strip()[:120]
    return {"department": guess_department(query), "title": title, "body": query.strip()[:4000]}
