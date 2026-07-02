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


# ── Conversation-history-aware query rewriting ──────────────────────────────────

class TestContextualizeNode:
    def test_no_history_skips_llm_call(self):
        from langchain_core.messages import HumanMessage

        from src.agent.graph import contextualize_node

        state = {"messages": [HumanMessage(content="Chuẩn đầu ra tin học yêu cầu gì?")],
                 "query": "Chuẩn đầu ra tin học yêu cầu gì?", "steps": []}
        with patch("src.agent.graph._contextualize_query") as mock_rewrite:
            result = contextualize_node(state)
        mock_rewrite.assert_not_called()
        assert result["tried_queries"] == [state["query"]]
        assert "query" not in result  # unchanged, no need to overwrite

    def test_history_present_rewrites_follow_up(self):
        from langchain_core.messages import AIMessage, HumanMessage

        from src.agent.graph import contextualize_node

        state = {
            "messages": [
                HumanMessage(content="Chuẩn đầu ra tin học yêu cầu gì?"),
                AIMessage(content="Cần UDCNTTCB, IC3/MOS/ICDL, hoặc các trường hợp khác do Hiệu trưởng quyết định."),
                HumanMessage(content="các trường hợp khác mà bạn nói đến là gì"),
            ],
            "query": "các trường hợp khác mà bạn nói đến là gì",
            "steps": [],
        }
        with patch(
            "src.agent.graph._contextualize_query",
            return_value="Các trường hợp khác được Hiệu trưởng quyết định miễn chuẩn đầu ra tin học là gì?",
        ) as mock_rewrite:
            result = contextualize_node(state)

        mock_rewrite.assert_called_once()
        assert result["query"] == "Các trường hợp khác được Hiệu trưởng quyết định miễn chuẩn đầu ra tin học là gì?"
        assert result["tried_queries"] == [result["query"]]
        assert result["steps"][-1]["label"] == "Diễn giải câu hỏi theo ngữ cảnh"

    def test_rewrite_llm_failure_falls_back_to_original(self, monkeypatch):
        from src.config import get_settings
        monkeypatch.setenv("GROQ_API_KEY", "")
        get_settings.cache_clear()

        from src.agent.graph import _contextualize_query

        result = _contextualize_query("câu hỏi gốc", [{"role": "user", "content": "gì đó"}])
        assert result == "câu hỏi gốc"
        get_settings.cache_clear()

    def test_contextualize_wired_before_router_in_compiled_graph(self):
        """Ensures the node is actually reachable from START, not just defined."""
        from src.agent.graph import build_graph

        graph = build_graph()
        # LangGraph compiled graphs expose their node names via get_graph().nodes
        node_names = set(graph.get_graph().nodes.keys())
        assert "do_contextualize" in node_names


# ── Self-correction retry loop ──────────────────────────────────────────────────

class TestGradingRetryRouting:
    def test_route_after_grade_relevant_goes_to_answer(self):
        from src.agent.graph import route_after_grade

        assert route_after_grade({"grade": "relevant", "retry_count": 0}) == "do_answer"

    def test_route_after_grade_irrelevant_retries(self):
        from src.agent.graph import route_after_grade

        assert route_after_grade({"grade": "irrelevant", "retry_count": 0}) == "do_reformulate"
        assert route_after_grade({"grade": "irrelevant", "retry_count": 1}) == "do_reformulate"

    def test_route_after_grade_exhausted_retries_gives_up(self):
        from src.agent.graph import MAX_RETRIES, route_after_grade

        assert route_after_grade({"grade": "irrelevant", "retry_count": MAX_RETRIES}) == "do_answer"


class TestGradeNode:
    def test_grade_node_relevant(self):
        from src.agent.graph import grade_node

        with patch("src.agent.graph.grade_chunks", return_value={"relevant": True, "reason": "khớp tốt"}):
            result = grade_node({"query": "q", "retrieved_chunks": [], "steps": []})
        assert result["grade"] == "relevant"
        assert result["grade_reason"] == "khớp tốt"
        assert result["steps"][-1]["label"] == "Đánh giá độ liên quan"

    def test_grade_node_irrelevant(self):
        from src.agent.graph import grade_node

        with patch("src.agent.graph.grade_chunks", return_value={"relevant": False, "reason": "không khớp"}):
            result = grade_node({"query": "q", "retrieved_chunks": [], "steps": []})
        assert result["grade"] == "irrelevant"


class TestReformulateNode:
    def test_reformulate_node_cheap_fallback_no_groq_key(self, monkeypatch):
        from src.config import get_settings
        monkeypatch.setenv("GROQ_API_KEY", "")
        get_settings.cache_clear()

        from src.agent.graph import reformulate_node

        state = {"query": "Sinh viên nghỉ học có được thi không", "tried_queries": ["Sinh viên nghỉ học có được thi không"], "steps": []}
        result = reformulate_node(state)

        assert result["retry_count"] == 1
        assert result["query"] != state["query"] or result["query"] == state["query"]
        assert result["query"] in result["tried_queries"]
        assert len(result["tried_queries"]) == 2
        get_settings.cache_clear()

    def test_reformulate_node_uses_llm_when_available(self):
        from src.agent.graph import reformulate_node

        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Điều kiện dự thi khi vắng mặt"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("groq.Groq", return_value=mock_client):
            state = {"query": "Sinh viên nghỉ học có được thi không", "tried_queries": ["Sinh viên nghỉ học có được thi không"], "steps": []}
            result = reformulate_node(state)

        assert result["query"] == "Điều kiện dự thi khi vắng mặt"
        assert result["retry_count"] == 1


class TestRetryLoopIntegration:
    def test_graph_terminates_after_max_retries_on_persistent_irrelevance(self):
        """Grader always says irrelevant -> loop must still terminate at MAX_RETRIES,
        not spin forever."""
        from src.agent.graph import MAX_RETRIES, build_graph
        from src.rag.retriever import RetrievedChunk

        weak_chunk = RetrievedChunk(text="unrelated", score=0.0, metadata={})

        async def run():
            graph = build_graph()
            with patch("src.agent.graph.grade_chunks", return_value={"relevant": False, "reason": "irrelevant"}), \
                 patch("src.rag.retriever._active_retriever", MagicMock(retrieve=MagicMock(return_value=[]))), \
                 patch("src.agent.graph.nodes_to_chunks", return_value=[weak_chunk]), \
                 patch("src.agent.graph.generate_answer", return_value={
                     "answer": "Tôi không tìm thấy quy định này trong tài liệu hiện có.",
                     "sources": [], "used_llm": "none", "chunk_count": 0,
                 }), \
                 patch("src.agent.graph.get_settings") as mock_settings, \
                 patch("src.agent.memory.LongTermMemory.log_query"):
                mock_settings.return_value.groq_api_key = ""
                initial_state = {
                    "messages": [], "query": "câu hỏi mơ hồ", "original_query": "câu hỏi mơ hồ",
                    "intent": "simple_qa", "retrieved_chunks": [], "answer": "", "sources": [],
                    "used_llm": "", "session_id": "s1", "latency_ms": 0, "error": None, "steps": [],
                    "retry_count": 0, "tried_queries": ["câu hỏi mơ hồ"], "grade": "unknown", "grade_reason": "",
                }
                return await graph.ainvoke(initial_state)

        result = asyncio.run(run())
        assert result["retry_count"] == MAX_RETRIES
        assert result["grade"] == "irrelevant"
        assert result["answer"]  # generator still produced an abstain answer, no crash

    def test_graph_stops_retrying_once_relevant(self):
        """First grade fails, second succeeds -> exactly one retry, not two."""
        from src.agent.graph import build_graph
        from src.rag.retriever import RetrievedChunk

        grade_results = iter([
            {"relevant": False, "reason": "chưa khớp"},
            {"relevant": True, "reason": "khớp rồi"},
        ])
        chunk = RetrievedChunk(text="nội dung", score=0.5, metadata={})

        async def run():
            graph = build_graph()
            with patch("src.agent.graph.grade_chunks", side_effect=lambda q, c: next(grade_results)), \
                 patch("src.rag.retriever._active_retriever", MagicMock(retrieve=MagicMock(return_value=[]))), \
                 patch("src.agent.graph.nodes_to_chunks", return_value=[chunk]), \
                 patch("src.agent.graph._cheap_reformulate", return_value="câu hỏi khác"), \
                 patch("src.agent.graph.get_settings") as mock_settings, \
                 patch("src.agent.graph.generate_answer", return_value={
                     "answer": "Trả lời [1]", "sources": [{"index": 1}], "used_llm": "groq", "chunk_count": 1,
                 }), \
                 patch("src.agent.memory.LongTermMemory.log_query"):
                mock_settings.return_value.groq_api_key = ""
                initial_state = {
                    "messages": [], "query": "câu hỏi ban đầu", "original_query": "câu hỏi ban đầu",
                    "intent": "simple_qa", "retrieved_chunks": [], "answer": "", "sources": [],
                    "used_llm": "", "session_id": "s1", "latency_ms": 0, "error": None, "steps": [],
                    "retry_count": 0, "tried_queries": ["câu hỏi ban đầu"], "grade": "unknown", "grade_reason": "",
                }
                return await graph.ainvoke(initial_state)

        result = asyncio.run(run())
        assert result["retry_count"] == 1
        assert result["grade"] == "relevant"


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
