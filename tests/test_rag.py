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

    @patch("src.rag.generator._call_gemini")
    def test_generate_answer_uses_gemini(self, mock_gemini):
        from src.rag.generator import generate_answer

        mock_gemini.return_value = "Cau tra loi tu Gemini [1]"
        chunks = [
            RetrievedChunk(
                text="Dieu 1 noi dung",
                score=0.9,
                metadata={"title": "Test Law", "dieu_header": "Dieu 1", "source_url": ""},
            )
        ]
        result = generate_answer("cau hoi", chunks)
        assert result["used_llm"] == "gemini"
        assert "Gemini" in result["answer"]

    @patch("src.rag.generator._call_gemini", side_effect=Exception("quota exhausted"))
    def test_generate_answer_falls_back_to_extractive(self, mock_gemini):
        """Gemini la nha cung cap duy nhat: het cap (key, model) thi trich nguyen van."""
        from src.rag.generator import generate_answer

        chunks = [
            RetrievedChunk(
                text="noi dung dieu luat",
                score=0.8,
                metadata={"title": "Law", "dieu_header": "Dieu 5", "source_url": ""},
            )
        ]
        result = generate_answer("cau hoi", chunks)
        assert result["used_llm"] == "extractive_fallback"

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
    async def test_stream_answer_handles_no_api_key(self):
        from src.rag.generator import stream_answer

        # không có key Gemini nào ⇒ không cặp (key, model) nào để gọi
        with patch("src.rag.generator._gemini_pairs", return_value=[]):
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


# ── Gemini embedder: key rotation & quota handling ─────────────────────────────

class TestGeminiEmbedderKeyRotation:
    """_GeminiAPIEmbedding must rotate keys itself instead of burning retries.

    Regression guard: the class used to be defined twice in embedder.py, so the
    stale single-key copy silently won.
    """

    @staticmethod
    def _make(keys=("k1", "k2")):
        from src.rag.embedder import _GeminiAPIEmbedding

        return _GeminiAPIEmbedding(model_name="gemini-embedding-001", api_keys=list(keys))

    def test_class_is_defined_once(self):
        import inspect

        import src.rag.embedder as mod

        source = inspect.getsource(mod)
        assert source.count("class _GeminiAPIEmbedding(BaseEmbedding):") == 1

    def test_429_moves_on_to_the_next_key(self):
        embedder = self._make()
        used = []

        def fake_call(key, texts, task_type):
            used.append(key)
            if key == "k1":
                raise RuntimeError("429 quota exceeded")
            return [[0.1, 0.2]]

        with patch.object(type(embedder), "_call_api", staticmethod(fake_call)):
            assert embedder._embed(["xin chào"], "retrieval_document") == [[0.1, 0.2]]
        assert used == ["k1", "k2"]          # đổi key ngay, không chờ backoff

    def test_exhausted_daily_quota_raises_instead_of_sleeping(self):
        import time as time_mod

        embedder = self._make()
        # 3 strikes = cạn quota ngày -> cooldown 1 giờ trên cả hai key
        for key in ("k1", "k2"):
            embedder._cooldown[key] = time_mod.monotonic() + 3600

        with patch("src.rag.embedder.time.sleep") as slept:
            with pytest.raises(RuntimeError, match="cạn quota"):
                embedder._reserve_key(["xin chào"])
        slept.assert_not_called()

    def test_minute_ceiling_still_waits(self):
        import time as time_mod

        embedder = self._make(keys=("k1",))
        embedder._cooldown["k1"] = time_mod.monotonic() + 30   # ngắn -> chờ, không lỗi

        with patch("src.rag.embedder.time.sleep") as slept:
            slept.side_effect = lambda _: embedder._cooldown.update(k1=0.0)
            assert embedder._reserve_key(["xin chào"]) == "k1"
        slept.assert_called_once()
