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


class TestQueryComplexityRouting:
    """Heuristic routing model + kích thước context theo độ khó câu hỏi
    (generator._is_complex_query / _select_models)."""

    def test_short_query_is_simple(self):
        from src.rag.generator import _is_complex_query

        assert _is_complex_query("Luong toi thieu vung I la bao nhieu?") is False

    def test_long_query_is_complex(self):
        from src.rag.generator import _is_complex_query

        assert _is_complex_query("a" * 200) is True

    def test_comparison_keyword_is_complex(self):
        from src.rag.generator import _is_complex_query

        assert _is_complex_query("So sánh mức lương tối thiểu vùng I và vùng II") is True

    def test_two_doc_refs_is_complex(self):
        from src.rag.generator import _is_complex_query

        assert _is_complex_query("So với 74-2024-ND-CP thì 293-2025-ND-CP đổi gì?") is True

    def test_long_history_is_complex(self):
        from src.rag.generator import _is_complex_query

        history = [{"role": "user", "content": "x"}] * 4
        assert _is_complex_query("cau hoi ngan", history=history) is True

    def test_select_models_simple_returns_default(self):
        from src.rag.generator import _select_models

        assert _select_models(False) is None

    def test_select_models_complex_returns_complex_tier(self):
        from src.rag.generator import _select_models

        models = _select_models(True)
        assert models is not None
        assert "gemini-3.7-flash" in models

    @patch("src.rag.generator._call_gemini")
    def test_generate_answer_trims_chunks_for_simple_query(self, mock_gemini):
        from src.rag.generator import _SIMPLE_QUERY_MAX_CHUNKS, generate_answer

        mock_gemini.return_value = "Tra loi [1]"
        chunks = [
            RetrievedChunk(text=f"Noi dung {i}", score=0.9, metadata={"title": f"Law {i}"})
            for i in range(8)
        ]
        generate_answer("cau hoi ngan don gian", chunks)
        called_context = mock_gemini.call_args.args[1]
        assert f"[{_SIMPLE_QUERY_MAX_CHUNKS}]" in called_context
        assert f"[{_SIMPLE_QUERY_MAX_CHUNKS + 1}]" not in called_context

    @patch("src.rag.generator._call_gemini")
    def test_generate_answer_keeps_all_chunks_for_complex_query(self, mock_gemini):
        from src.rag.generator import generate_answer

        mock_gemini.return_value = "So sanh: A [1] con B [8]"
        chunks = [
            RetrievedChunk(text=f"Noi dung {i}", score=0.9, metadata={"title": f"Law {i}"})
            for i in range(8)
        ]
        generate_answer("So sánh điều kiện giữa hai văn bản này khác nhau ra sao", chunks)
        called_context = mock_gemini.call_args.args[1]
        assert "[8]" in called_context


class _ImmediateThread:
    """Stand-in cho threading.Thread chạy target NGAY (đồng bộ) thay vì
    thread thật — để test có thể assert kết quả mà không cần join()."""

    def __init__(self, target=None, daemon=None):
        self._target = target

    def start(self):
        self._target()


class TestLangfuseTracing:
    """REST thuần (không SDK) gửi trace lên Langfuse qua OTLP/HTTP — xem lý do
    trong config.langfuse_* / src/langfuse_otel.py: langfuse-python (OTel-based)
    xung đột opentelemetry version với chromadb trong venv này."""

    @patch("src.langfuse_otel.get_settings")
    def test_no_keys_configured_sends_nothing(self, mock_settings):
        from datetime import datetime, timezone

        from src.langfuse_otel import record_generation

        mock_settings.return_value.langfuse_public_key = ""
        mock_settings.return_value.langfuse_secret_key = ""
        with patch("threading.Thread") as mock_thread:
            record_generation(
                "gemini-generate", "model", "prompt", "output",
                datetime.now(timezone.utc), datetime.now(timezone.utc),
                prompt_tokens=1, completion_tokens=2,
            )
            mock_thread.assert_not_called()

    @patch("src.langfuse_otel.get_settings")
    def test_standalone_generation_sends_one_span_to_otel_endpoint(self, mock_settings):
        """Gọi record_generation() ngoài start_trace() (vd. gemini_generate() từ
        eval script) => tự tạo 1 trace đứng riêng, gửi ngay tới endpoint OTel mới
        (không phải Legacy Ingestion API sunset 16/11/2026)."""
        from datetime import datetime, timezone

        from src.langfuse_otel import record_generation

        mock_settings.return_value.langfuse_public_key = "pk-test"
        mock_settings.return_value.langfuse_secret_key = "sk-test"
        mock_settings.return_value.langfuse_host = "https://cloud.langfuse.com"

        captured = {}

        def fake_post(url, json=None, auth=None, headers=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["auth"] = auth
            captured["headers"] = headers
            return MagicMock()

        with patch("threading.Thread", _ImmediateThread), \
             patch("requests.post", side_effect=fake_post):
            record_generation(
                "gemini-generate", "gemini-3.1-flash-lite", "cau hoi", "cau tra loi",
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
                prompt_tokens=10, completion_tokens=5,
            )

        assert captured["url"] == "https://cloud.langfuse.com/api/public/otel/v1/traces"
        assert captured["auth"] == ("pk-test", "sk-test")
        assert captured["headers"]["x-langfuse-ingestion-version"] == "4"
        spans = captured["json"]["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(spans) == 1
        span = spans[0]
        assert "parentSpanId" not in span  # standalone => tự làm root, không cha
        attrs = {a["key"]: a["value"] for a in span["attributes"]}
        assert attrs["langfuse.observation.type"]["stringValue"] == "generation"
        assert attrs["langfuse.observation.model.name"]["stringValue"] == "gemini-3.1-flash-lite"
        assert '"input":10' in attrs["langfuse.observation.usage_details"]["stringValue"]

    @patch("src.langfuse_otel.get_settings")
    def test_network_error_does_not_raise(self, mock_settings):
        from datetime import datetime, timezone

        from src.langfuse_otel import record_generation

        mock_settings.return_value.langfuse_public_key = "pk-test"
        mock_settings.return_value.langfuse_secret_key = "sk-test"
        mock_settings.return_value.langfuse_host = "https://cloud.langfuse.com"

        with patch("threading.Thread", _ImmediateThread), \
             patch("requests.post", side_effect=ConnectionError("boom")):
            record_generation(  # không raise ra ngoài — quan sát là best-effort
                "gemini-generate", "m", "p", "o",
                datetime.now(timezone.utc), datetime.now(timezone.utc),
            )

    @patch("src.langfuse_otel.get_settings")
    def test_trace_nests_child_spans_under_root(self, mock_settings):
        """start_trace() + record_generation()/record_span() bên trong => children
        KHÔNG gửi ngay, chỉ gửi 1 batch (root + con) khi end_trace()."""
        from datetime import datetime, timezone

        from src.langfuse_otel import end_trace, record_generation, record_span, start_trace

        mock_settings.return_value.langfuse_public_key = "pk-test"
        mock_settings.return_value.langfuse_secret_key = "sk-test"
        mock_settings.return_value.langfuse_host = "https://cloud.langfuse.com"

        captured = {}

        def fake_post(url, json=None, auth=None, headers=None, timeout=None):
            captured["json"] = json
            return MagicMock()

        with patch("threading.Thread", _ImmediateThread), \
             patch("requests.post", side_effect=fake_post):
            ctx, token = start_trace("documind-agent-query", session_id="s1", tags=["legal-qa"])
            now = datetime.now(timezone.utc)
            record_span("retrieve-documents", "retriever", "q", "5 đoạn", now, now)
            record_generation(
                "gemini-generate", "gemini-3.1-flash-lite", "prompt", "answer", now, now,
            )
            # Chưa end_trace() => chưa gửi gì cả.
            assert "json" not in captured
            end_trace(ctx, token, "documind-agent-query", "q", "answer")

        spans = captured["json"]["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(spans) == 3  # root + retriever span + generation span
        root = next(s for s in spans if "parentSpanId" not in s)
        children = [s for s in spans if "parentSpanId" in s]
        assert len(children) == 2
        assert all(c["parentSpanId"] == root["spanId"] for c in children)
        assert all(c["traceId"] == root["traceId"] for c in children)
        root_attrs = {a["key"]: a["value"] for a in root["attributes"]}
        assert root_attrs["langfuse.session.id"]["stringValue"] == "s1"


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
