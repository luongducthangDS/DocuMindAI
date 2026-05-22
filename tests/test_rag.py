"""
Tests for RAG pipeline: embedder, retriever, generator.
Uses mocks to avoid real API calls and heavy model downloads.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.retriever import RetrievedChunk, nodes_to_chunks


# ── RetrievedChunk Tests ───────────────────────────────────────────────────────

class TestRetrievedChunk:
    def test_citation_label_with_full_metadata(self):
        chunk = RetrievedChunk(
            text="sample text",
            score=0.9,
            metadata={
                "title": "Luật DN 2020",
                "so_hieu": "59/2020/QH14",
                "dieu_header": "Điều 1. Phạm vi",
                "source_url": "https://vbpl.vn/test",
            },
        )
        label = chunk.citation_label
        assert "59/2020/QH14" in label
        assert "Điều 1" in label
        assert "vbpl.vn" in label

    def test_citation_label_without_so_hieu(self):
        chunk = RetrievedChunk(
            text="text",
            score=0.5,
            metadata={"title": "Some Law"},
        )
        label = chunk.citation_label
        assert "Some Law" in label


# ── Generator Tests ────────────────────────────────────────────────────────────

class TestGenerator:
    def test_generate_answer_empty_chunks_returns_not_found(self):
        from src.rag.generator import generate_answer

        result = generate_answer("some query", [])
        assert "không tìm thấy" in result["answer"].lower()
        assert result["used_llm"] == "none"
        assert result["chunk_count"] == 0

    @patch("src.rag.generator._call_groq")
    def test_generate_answer_calls_groq_first(self, mock_groq):
        from src.rag.generator import generate_answer

        mock_groq.return_value = "Câu trả lời từ Groq"
        chunks = [
            RetrievedChunk(
                text="Điều 1 nội dung",
                score=0.9,
                metadata={"title": "Test Law", "dieu_header": "Điều 1", "source_url": ""},
            )
        ]
        result = generate_answer("câu hỏi", chunks)
        assert result["used_llm"] == "groq"
        assert "Câu trả lời từ Groq" in result["answer"]

    @patch("src.rag.generator._call_groq", side_effect=Exception("Timeout"))
    @patch("src.rag.generator._call_gemini")
    def test_generate_answer_falls_back_to_gemini(self, mock_gemini, mock_groq):
        from src.rag.generator import generate_answer

        mock_gemini.return_value = "Câu trả lời từ Gemini"
        chunks = [
            RetrievedChunk(
                text="nội dung",
                score=0.8,
                metadata={"title": "Law", "dieu_header": "", "source_url": ""},
            )
        ]
        result = generate_answer("câu hỏi", chunks)
        assert "gemini" in result["used_llm"]

    def test_build_context_includes_all_chunks(self):
        from src.rag.generator import _build_context

        chunks = [
            RetrievedChunk(text=f"Text {i}", score=0.9, metadata={
                "title": f"Law {i}", "dieu_header": "", "source_url": ""
            })
            for i in range(3)
        ]
        context, citations = _build_context(chunks)
        assert "[1]" in context
        assert "[2]" in context
        assert "[3]" in context
        assert "[1]" in citations

    @pytest.mark.asyncio
    @patch("src.rag.generator._get_groq_client")
    async def test_stream_answer_handles_no_api_key(self, mock_client):
        from src.rag.generator import stream_answer

        # simulate empty API key
        with patch("src.rag.generator.get_settings") as mock_settings:
            mock_settings.return_value.groq_api_key = ""
            tokens = []
            async for token in stream_answer("q", []):
                tokens.append(token)
            assert any("không" in t.lower() or "chưa" in t.lower() for t in tokens)


# ── Retriever Tests ────────────────────────────────────────────────────────────

class TestNodesConversion:
    def test_nodes_to_chunks_converts_correctly(self):
        mock_node = MagicMock()
        mock_node.node.text = "Legal text content"  # nodes_to_chunks reads .text first
        mock_node.node.get_content.return_value = "Legal text content"
        mock_node.node.metadata = {"title": "Test Law", "dieu_header": "Điều 1"}
        mock_node.score = 0.85

        chunks = nodes_to_chunks([mock_node])
        assert len(chunks) == 1
        assert chunks[0].text == "Legal text content"
        assert chunks[0].score == 0.85
        assert chunks[0].metadata["title"] == "Test Law"

    def test_nodes_to_chunks_handles_none_score(self):
        mock_node = MagicMock()
        mock_node.node.get_content.return_value = "text"
        mock_node.node.metadata = {}
        mock_node.score = None

        chunks = nodes_to_chunks([mock_node])
        assert chunks[0].score == 0.0
