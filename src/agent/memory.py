"""
Two-tier memory:
  - Short-term: in-session list (Python dict, per request_id)
  - Long-term:  SQLite with parameterized queries (no SQL injection)
"""

import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.config import get_settings


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ShortTermMemory:
    """In-process session memory. Not shared across workers."""

    def __init__(self, max_turns: int = 10):
        self._max = max_turns
        self._history: list[Message] = []

    def add(self, role: str, content: str) -> None:
        self._history.append(Message(role=role, content=content))
        if len(self._history) > self._max * 2:
            # Keep last max_turns pairs — trim oldest
            self._history = self._history[-(self._max * 2):]

    def as_messages(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self._history]

    def clear(self) -> None:
        self._history.clear()


class LongTermMemory:
    """
    SQLite-backed cross-session memory.
    Stores session summaries and extracted legal entities.
    All queries parameterized to prevent SQL injection.
    """

    def __init__(self, db_path: Path | None = None):
        settings = get_settings()
        self._db = db_path or settings.sqlite_db
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self._db))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    query_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS query_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer_snippet TEXT,
                    latency_ms INTEGER,
                    used_llm TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE INDEX IF NOT EXISTS idx_query_log_session
                    ON query_log(session_id);
            """)
        logger.debug("LongTermMemory DB initialized: {}", self._db)

    def create_session(self, session_id: str | None = None) -> str:
        sid = session_id or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id, summary, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (sid, "", now, now),
            )
        return sid

    def update_session_summary(self, session_id: str, summary: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET summary = ?, updated_at = ?, query_count = query_count + 1 WHERE id = ?",
                (summary[:2000], now, session_id),  # cap summary length
            )

    def get_session_summary(self, session_id: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return row["summary"] if row else ""

    def log_query(
        self,
        session_id: str,
        query: str,
        answer_snippet: str,
        latency_ms: int,
        used_llm: str,
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO query_log
                   (session_id, query, answer_snippet, latency_ms, used_llm, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    query[:500],          # truncate long queries
                    answer_snippet[:300], # truncate for storage
                    latency_ms,
                    used_llm,
                    now,
                ),
            )

    def get_recent_queries(self, session_id: str, limit: int = 5) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT query, answer_snippet, latency_ms, used_llm, created_at
                   FROM query_log
                   WHERE session_id = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]
