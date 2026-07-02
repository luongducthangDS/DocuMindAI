"""
Tests for src/rag/grader.py — relevance grading for the self-correction retry loop.
"""

from unittest.mock import patch

from src.rag.retriever import RetrievedChunk


class TestGradeChunks:
    def test_no_chunks_is_irrelevant_without_llm_call(self):
        from src.rag.grader import grade_chunks

        with patch("src.rag.grader._call_judge_llm") as mock_judge:
            result = grade_chunks("bất kỳ câu hỏi nào", [])
        assert result["relevant"] is False
        mock_judge.assert_not_called()

    def test_confident_shortcut_skips_llm_when_threshold_positive(self):
        """Reranker enabled (threshold=0.05): a score well above it should
        short-circuit to relevant without calling the judge LLM."""
        from src.rag.grader import grade_chunks

        chunks = [RetrievedChunk(text="nội dung", score=0.5, metadata={})]
        with patch("src.rag.grader._effective_min_score", return_value=0.05), \
             patch("src.rag.grader._call_judge_llm") as mock_judge:
            result = grade_chunks("câu hỏi", chunks)
        assert result["relevant"] is True
        mock_judge.assert_not_called()

    def test_threshold_zero_never_shortcuts_always_calls_judge(self):
        """Regression test: when the reranker is disabled (e.g. Render free
        tier), _effective_min_score() returns 0.0. `best_score >= 0 * 2.0` is
        then true for ANY positive score, which used to short-circuit every
        query as "relevant" without ever consulting the judge LLM — silently
        disabling the whole grading step in production. The judge must always
        be consulted when threshold is 0, regardless of how low or high the
        (unreliable) score is."""
        from src.rag.grader import grade_chunks

        chunks = [RetrievedChunk(text="nội dung không liên quan", score=0.033, metadata={})]
        with patch("src.rag.grader._effective_min_score", return_value=0.0), \
             patch("src.rag.grader._call_judge_llm", return_value={"relevant": False, "reason": "không liên quan"}) as mock_judge:
            result = grade_chunks("câu hỏi", chunks)
        mock_judge.assert_called_once()
        assert result["relevant"] is False

    def test_llm_unavailable_fails_open(self):
        from src.rag.grader import grade_chunks

        chunks = [RetrievedChunk(text="nội dung", score=0.01, metadata={})]
        with patch("src.rag.grader._effective_min_score", return_value=0.05), \
             patch("src.rag.grader._call_judge_llm", return_value=None):
            result = grade_chunks("câu hỏi", chunks)
        assert result["relevant"] is True
