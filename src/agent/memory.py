"""
Two-tier memory:
  - Short-term: in-session list (Python dict, per request_id)
  - Long-term:  SQLite with parameterized queries (no SQL injection)
"""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
        conn = sqlite3.connect(str(self._db), check_same_thread=False)
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
                CREATE TABLE IF NOT EXISTS query_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer_snippet TEXT,
                    latency_ms INTEGER,
                    used_llm TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_query_log_session
                    ON query_log(session_id);

                CREATE TABLE IF NOT EXISTS upload_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_size_bytes INTEGER,
                    indexed_chunks INTEGER,
                    status TEXT NOT NULL,
                    error_detail TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS error_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    session_id TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_error_log_created
                    ON error_log(created_at);
            """)
        logger.debug("LongTermMemory DB initialized: {}", self._db)

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

    def log_upload(
        self,
        filename: str,
        file_size_bytes: int,
        indexed_chunks: int,
        status: str,
        error_detail: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO upload_log
                   (filename, file_size_bytes, indexed_chunks, status, error_detail,
                    ip_address, user_agent, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (filename[:260], file_size_bytes, indexed_chunks, status,
                 error_detail, ip_address, user_agent, now),
            )

    def log_error(
        self,
        endpoint: str,
        error_type: str,
        session_id: str | None = None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO error_log (endpoint, error_type, session_id, created_at)
                   VALUES (?, ?, ?, ?)""",
                (endpoint, error_type, session_id, now),
            )

    def get_error_rate(self, window_minutes: int = 60) -> float:
        since = (datetime.utcnow() - timedelta(minutes=window_minutes)).isoformat()
        with self._connect() as conn:
            errors = conn.execute(
                "SELECT COUNT(*) FROM error_log WHERE created_at >= ?", (since,)
            ).fetchone()[0]
            queries = conn.execute(
                "SELECT COUNT(*) FROM query_log WHERE created_at >= ?", (since,)
            ).fetchone()[0]
        if queries == 0:
            return 0.0
        return round(errors / queries, 4)


# Process-level singleton — one instance per worker process
_ltm: LongTermMemory | None = None


def get_long_term_memory() -> LongTermMemory:
    global _ltm
    if _ltm is None:
        _ltm = LongTermMemory()
    return _ltm
