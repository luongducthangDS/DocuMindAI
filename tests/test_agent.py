"""
Tests for LangGraph agent: memory, tools, graph routing.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.memory import LongTermMemory, ShortTermMemory


# ── Memory Tests ───────────────────────────────────────────────────────────────

class TestShortTermMemory:
    def test_add_and_retrieve_messages(self):
        mem = ShortTermMemory(max_turns=3)
        mem.add("user", "câu hỏi 1")
        mem.add("assistant", "trả lời 1")
        messages = mem.as_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "câu hỏi 1"

    def test_max_turns_trims_oldest(self):
        mem = ShortTermMemory(max_turns=2)
        for i in range(5):
            mem.add("user", f"q{i}")
            mem.add("assistant", f"a{i}")
        # Should keep only last 2 turns = 4 messages
        assert len(mem.as_messages()) <= 4

    def test_clear_empties_history(self):
        mem = ShortTermMemory()
        mem.add("user", "test")
        mem.clear()
        assert mem.as_messages() == []


class TestLongTermMemory:
    def test_create_and_retrieve_session(self, tmp_path):
        db = tmp_path / "test_mem.db"
        mem = LongTermMemory(db_path=db)
        sid = mem.create_session()
        assert len(sid) == 36  # UUID format

        # Should not raise on second call (IGNORE duplicate)
        mem.create_session(session_id=sid)

    def test_log_and_retrieve_queries(self, tmp_path):
        db = tmp_path / "test_mem.db"
        mem = LongTermMemory(db_path=db)
        sid = mem.create_session()

        mem.log_query(
            session_id=sid,
            query="Điều kiện thành lập công ty?",
            answer_snippet="Cần ít nhất 2 thành viên",
            latency_ms=250,
            used_llm="groq",
        )

        recent = mem.get_recent_queries(sid, limit=5)
        assert len(recent) == 1
        assert recent[0]["used_llm"] == "groq"
        assert recent[0]["latency_ms"] == 250

    def test_update_session_summary(self, tmp_path):
        db = tmp_path / "test_mem.db"
        mem = LongTermMemory(db_path=db)
        sid = mem.create_session()
        mem.update_session_summary(sid, "User asked about labor law")
        summary = mem.get_session_summary(sid)
        assert "labor law" in summary

    def test_long_query_is_truncated(self, tmp_path):
        db = tmp_path / "test_mem.db"
        mem = LongTermMemory(db_path=db)
        sid = mem.create_session()

        very_long_query = "A" * 1000
        # Should not raise
        mem.log_query(sid, very_long_query, "short answer", 100, "groq")

        recent = mem.get_recent_queries(sid)
        assert len(recent[0]["query"]) <= 500

    def test_sql_injection_prevention(self, tmp_path):
        db = tmp_path / "test_mem.db"
        mem = LongTermMemory(db_path=db)
        # Injection attempt in session_id should be caught by validation
        # at the API layer, but DB itself must handle safely
        sid = mem.create_session()
        malicious = "'; DROP TABLE sessions; --"
        # Should not raise, logs to query_log with parameterized query
        mem.log_query(sid, malicious, "answer", 100, "groq")
        # Sessions table should still exist
        recent = mem.get_recent_queries(sid)
        assert isinstance(recent, list)


# ── Graph Tests ────────────────────────────────────────────────────────────────

class TestRouter:
    def test_keyword_classify_compare(self):
        from src.agent.graph import _keyword_classify

        assert _keyword_classify("So sánh Luật DN 2020 và 2014") == "compare"
        assert _keyword_classify("Khác nhau giữa công ty TNHH và cổ phần?") == "compare"

    def test_keyword_classify_summarize(self):
        from src.agent.graph import _keyword_classify

        assert _keyword_classify("Tóm tắt Luật lao động") == "summarize"

    def test_keyword_classify_report(self):
        from src.agent.graph import _keyword_classify

        assert _keyword_classify("Tạo báo cáo về thuế") == "report"
        assert _keyword_classify("Xuất PDF về lao động") == "report"

    def test_keyword_classify_default_simple_qa(self):
        from src.agent.graph import _keyword_classify

        assert _keyword_classify("Điều kiện thành lập doanh nghiệp?") == "simple_qa"

    def test_route_by_intent_mapping(self):
        from src.agent.graph import route_by_intent

        assert route_by_intent({"intent": "compare"}) == "do_compare"
        assert route_by_intent({"intent": "summarize"}) == "do_summarize"
        assert route_by_intent({"intent": "report"}) == "do_report"
        assert route_by_intent({"intent": "simple_qa"}) == "do_retrieve"
        assert route_by_intent({"intent": "unknown"}) == "do_retrieve"
        assert route_by_intent({"intent": "invalid"}) == "do_retrieve"


# ── Schema Tests ───────────────────────────────────────────────────────────────

class TestSchemas:
    def test_query_request_rejects_too_short(self):
        from pydantic import ValidationError
        from src.api.schemas import QueryRequest

        with pytest.raises(ValidationError):
            QueryRequest(query="ab")

    def test_query_request_rejects_script_injection(self):
        from pydantic import ValidationError
        from src.api.schemas import QueryRequest

        with pytest.raises(ValidationError):
            QueryRequest(query="<script>alert('xss')</script> is this legal?")

    def test_query_request_rejects_invalid_session_id(self):
        from pydantic import ValidationError
        from src.api.schemas import QueryRequest

        with pytest.raises(ValidationError):
            QueryRequest(query="valid question here", session_id="invalid session!")

    def test_report_request_sanitizes_filename(self):
        from src.api.schemas import ReportRequest

        req = ReportRequest(
            title="Test",
            query="test query",
            filename="../../../etc/passwd",
        )
        assert ".." not in req.filename
        assert "/" not in req.filename
        assert "etc" in req.filename or req.filename  # sanitized but not empty

    def test_report_request_validates_email(self):
        from pydantic import ValidationError
        from src.api.schemas import ReportRequest

        with pytest.raises(ValidationError):
            ReportRequest(
                title="Test",
                query="test query",
                email_to="not-an-email",
            )
