"""
Tests for RAG pipeline: embedder, retriever, generator.
Uses mocks to avoid real API calls and heavy model downloads.
"""

import asyncio
import time
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

    @patch("src.rag.generator._call_gemini", side_effect=Exception("quota exhausted"))
    def test_extractive_fallback_skips_forms_and_caps_sources(self, mock_gemini):
        """Fallback không dán cả 5 chunk: bỏ biểu mẫu, tối đa 3 nguồn, số [i] khớp chunk gốc."""
        from src.rag.generator import generate_answer

        headers = ["Điều 38. Điều kiện hưởng", "Mẫu số 23", "Điều 39. Mức hưởng",
                   "Điều 15. Mức hưởng", "Điều 8. Thời gian đóng"]
        chunks = [
            RetrievedChunk(text=f"noi dung {h}", score=0.8,
                           metadata={"title": "Law", "dieu_header": h, "source_url": ""})
            for h in headers
        ]
        result = generate_answer("cau hoi", chunks)
        cited = [s["dieu_header"] for s in result["sources"]]
        assert cited == ["Điều 38. Điều kiện hưởng", "Điều 39. Mức hưởng", "Điều 15. Mức hưởng"]
        assert "Mẫu số" not in result["answer"]

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


class _FakeChunk:
    def __init__(self, text: str):
        self.text = text


class _FakeUsage:
    def __init__(self, prompt_tokens: int = 5, completion_tokens: int = 7):
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = completion_tokens


class _FakeStream:
    """Giả lập response.stream=True của genai — iterable đồng bộ, tuỳ chọn
    time.sleep() giữa các chunk để mô phỏng độ trễ mạng thật (chạy trong
    thread nền của _stream_chunks_in_thread, không phải trong test coroutine)."""

    def __init__(self, texts: list[str], delay: float = 0.0):
        self._texts = texts
        self._delay = delay
        self.usage_metadata = _FakeUsage()

    def __iter__(self):
        for t in self._texts:
            if self._delay:
                time.sleep(self._delay)
            yield _FakeChunk(t)


class _FakeModel:
    def __init__(self, texts: list[str], delay: float = 0.0):
        self._texts = texts
        self._delay = delay

    def generate_content(self, prompt, stream=True, **kwargs):
        return _FakeStream(self._texts, delay=self._delay)


def _make_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        text="noi dung", score=0.9,
        metadata={"title": "Law", "dieu_header": "", "source_url": ""},
    )


class TestStreamAnswerConcurrency:
    """_stream_chunks_in_thread + queue bridge trong stream_answer() — thay
    cho việc lặp trực tiếp generator đồng bộ của genai trong 1 coroutine async
    (bug thật: đứng hình CẢ event loop, không chỉ 1 kết nối, khi chờ chunk kế
    tiếp từ mạng — đo được ở local: 1 WS treo kéo REST không liên quan chờ
    hơn 1 phút)."""

    @pytest.mark.asyncio
    async def test_yields_tokens_in_order_and_records_generation(self):
        from src.rag.generator import stream_answer

        chunks = [_make_chunk()]
        fake_model = _FakeModel(["Xin ", "chao"])

        with patch("src.rag.generator._gemini_pairs", return_value=[("key1", "model1")]), \
             patch("src.rag.generator._effective_min_score", return_value=0.0), \
             patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel", return_value=fake_model), \
             patch("src.rag.generator.record_generation") as mock_record:
            tokens = [t async for t in stream_answer("cau hoi", chunks)]

        assert "".join(tokens) == "Xin chao"
        mock_record.assert_called_once()
        args = mock_record.call_args.args
        assert args[0] == "generate-answer-stream"
        assert args[3] == "Xin chao"  # output_text
        assert mock_record.call_args.kwargs["prompt_tokens"] == 5
        assert mock_record.call_args.kwargs["completion_tokens"] == 7

    @pytest.mark.asyncio
    async def test_does_not_block_event_loop_while_waiting_for_network(self):
        from src.rag.generator import stream_answer

        chunks = [_make_chunk()]
        ticks = 0

        async def ticker():
            nonlocal ticks
            for _ in range(20):
                await asyncio.sleep(0.01)
                ticks += 1

        # 2 chunk x 0.1s "độ trễ mạng" (thread nền) = ~0.2s tổng. Nếu event loop
        # KHÔNG bị chặn, ticker (tick mỗi 0.01s) chạy song song suốt lúc đó.
        with patch("src.rag.generator._gemini_pairs", return_value=[("key1", "model1")]), \
             patch("src.rag.generator._effective_min_score", return_value=0.0), \
             patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel",
                   return_value=_FakeModel(["A", "B"], delay=0.1)), \
             patch("src.rag.generator.record_generation"):

            async def consume():
                async for _ in stream_answer("cau hoi", chunks):
                    pass

            ticker_task = asyncio.create_task(ticker())
            await consume()
            ticker_task.cancel()

        assert ticks >= 10


class TestGeminiOverload:
    """Gemini 503 từng làm 1 lời gọi treo tới 600s: SDK tự retry ngầm trên cùng
    cặp, vòng xoay (key, model) không bao giờ chạy (Render 2026-09-23)."""

    def test_failed_attempts_are_traced_as_errors(self, monkeypatch):
        """Bước tổng hợp 44s xoay vòng 503 từng không để lại span nào trên
        Langfuse (reports/daily/2026-09-24.md) — lần thử lỗi cũng phải được ghi."""
        import src.rag.generator as gen

        monkeypatch.setattr(gen, "_GEMINI_PAIR_CURSOR", 0)
        monkeypatch.setattr(gen, "_gemini_pairs", lambda models=None: [("k1", "m1"), ("k2", "m2")])
        results = iter([RuntimeError("429 quota exceeded"), MagicMock(text="ok", usage_metadata=None)])

        def fake_call(*args, **kwargs):
            r = next(results)
            if isinstance(r, Exception):
                raise r
            return r

        with patch("src.rag.generator.gemini_call", side_effect=fake_call), \
             patch("src.rag.generator.record_generation") as mock_record:
            assert gen.gemini_generate("p") == "ok"

        failed, succeeded = mock_record.call_args_list
        assert failed.args[1] == "m1" and "429" in failed.kwargs["error"]
        assert succeeded.args[1] == "m2" and "error" not in succeeded.kwargs

    def test_overloaded_model_and_exhausted_key_are_skipped_without_sdk_retry(self, monkeypatch):
        import src.rag.generator as gen

        monkeypatch.setattr(gen, "_PAIR_DOWN_UNTIL", {})
        monkeypatch.setattr(gen, "_GEMINI_PAIR_CURSOR", 0)
        monkeypatch.setattr(gen, "_gemini_keys", lambda: ["k1", "k2"])
        calls = []

        def fake_generate(prompt, **kwargs):
            # Key đi theo client gắn vào model (genai_client), không qua genai.configure.
            pair = (model_cls.return_value._client.api_key, model_cls.call_args.args[0])
            calls.append(pair)
            assert kwargs["request_options"]["retry"] is None
            assert kwargs["request_options"]["timeout"]
            if pair[1] == "m1":
                raise RuntimeError("503 This model is currently experiencing high demand")
            if pair == ("k1", "m2"):
                raise RuntimeError("429 You exceeded your current quota")
            return MagicMock(text="ok", usage_metadata=None)

        with patch("src.rag.embedder.genai_client", lambda k: MagicMock(api_key=k)), \
             patch("google.generativeai.GenerativeModel") as model_cls, \
             patch("src.rag.generator.record_generation"):
            model_cls.return_value.generate_content.side_effect = fake_generate
            assert gen.gemini_generate("q", models=["m1", "m2"]) == "ok"
            assert gen.gemini_generate("q", models=["m1", "m2"]) == "ok"

        # 503 => cả m1 bị bỏ (k2/m1 không gọi); 429 => chỉ k1/m2; lần 2 đi thẳng k2/m2.
        assert calls == [("k1", "m1"), ("k1", "m2"), ("k2", "m2"), ("k2", "m2")]


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

    def __init__(self, target=None, daemon=None, name=None):
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
    def test_error_generation_marked_error_level(self, mock_settings):
        from datetime import datetime, timezone

        from src.langfuse_otel import record_generation

        mock_settings.return_value.langfuse_public_key = "pk-test"
        mock_settings.return_value.langfuse_secret_key = "sk-test"
        mock_settings.return_value.langfuse_host = "https://cloud.langfuse.com"
        captured = {}

        def fake_post(url, json=None, **_kw):
            captured["json"] = json
            return MagicMock()  # ok=truthy: gửi thành công, không gửi lại

        with patch("threading.Thread", _ImmediateThread), \
             patch("requests.post", side_effect=fake_post):
            record_generation(
                "gemini-generate", "m", "p", "",
                datetime.now(timezone.utc), datetime.now(timezone.utc),
                error="RuntimeError: 503 high demand",
            )

        span = captured["json"]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        attrs = {a["key"]: a["value"]["stringValue"] for a in span["attributes"]}
        assert attrs["langfuse.observation.level"] == "ERROR"
        assert "503" in attrs["langfuse.observation.status_message"]
        assert span["status"]["code"] == 2

    @patch("src.langfuse_otel.get_settings")
    def test_network_error_does_not_raise(self, mock_settings):
        from datetime import datetime, timezone

        from src.langfuse_otel import record_generation

        mock_settings.return_value.langfuse_public_key = "pk-test"
        mock_settings.return_value.langfuse_secret_key = "sk-test"
        mock_settings.return_value.langfuse_host = "https://cloud.langfuse.com"

        with patch("threading.Thread", _ImmediateThread), \
             patch("src.langfuse_otel._RETRY_BACKOFF_S", (0, 0)), \
             patch("requests.post", side_effect=ConnectionError("boom")):
            record_generation(  # không raise ra ngoài — quan sát là best-effort
                "gemini-generate", "m", "p", "o",
                datetime.now(timezone.utc), datetime.now(timezone.utc),
            )

    @pytest.mark.parametrize("statuses, posts, dropped", [
        ([503, 200], 2, False),       # quá tải tạm thời → gửi lại, tới nơi, không cảnh báo
        ([401], 1, True),             # key sai → gửi lại vô ích, nhưng KHÔNG được mất im lặng
        ([429, 503, 503], 3, True),   # hết lượt gửi lại → bỏ, có cảnh báo
    ])
    def test_delivery_retries_transient_and_logs_drops(self, monkeypatch, statuses, posts, dropped):
        from types import SimpleNamespace

        from loguru import logger

        import src.langfuse_otel as lf

        monkeypatch.setattr(lf, "_RETRY_BACKOFF_S", (0, 0))
        codes = iter(statuses)
        posted, warnings = [], []

        def fake_post(*_a, **_kw):
            code = next(codes)
            posted.append(code)
            return SimpleNamespace(ok=code < 300, status_code=code, text="x")

        sink = logger.add(lambda m: warnings.append(m.record["message"]), level="WARNING")
        try:
            with patch("requests.post", fake_post):
                lf._deliver("https://langfuse.test/api/public/otel/v1/traces", {}, ("pk", "sk"))
        finally:
            logger.remove(sink)
        assert len(posted) == posts
        assert bool(warnings) == dropped
        if dropped:
            assert f"HTTP {statuses[-1]}" in warnings[0]

    @staticmethod
    def _exit_right_after_send(marker, send_seconds: float, flush_cap: float = 5.0) -> tuple[float, str]:
        """Process con: gửi 1 batch mất `send_seconds` rồi thoát NGAY; batch gửi xong thì
        tạo `marker`. Trả về (số giây process sống — cả khởi động + import, stderr)."""
        import os
        import subprocess
        import sys
        import time
        from pathlib import Path

        code = (
            "import time, requests\n"
            "from types import SimpleNamespace\n"
            "import src.langfuse_otel as lf\n"
            "def slow_post(*_a, **_kw):\n"
            f"    time.sleep({send_seconds})  # script đã tới dòng cuối trong lúc đang gửi\n"
            f"    open({str(marker)!r}, 'w').close()\n"
            "    return SimpleNamespace(ok=True)\n"
            "lf._auth = lambda: ('pk', 'sk', 'https://langfuse.test')\n"
            f"lf._EXIT_FLUSH_S = {flush_cap}\n"
            "requests.post = slow_post\n"
            "lf._send_batch([{'spanId': 'x'}])\n"
        )
        t0 = time.monotonic()
        proc = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                              check=True, timeout=60, capture_output=True, encoding="utf-8",
                              errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        return time.monotonic() - t0, proc.stderr

    def test_batch_sent_right_before_exit_is_not_lost(self, tmp_path):
        """Script eval/smoke gửi trace rồi thoát ngay: phải gửi xong trước khi process tắt
        (thread daemon trần từng bị giết giữa chừng — mất trace cuối)."""
        self._exit_right_after_send(tmp_path / "sent", send_seconds=0.5)
        assert (tmp_path / "sent").exists()

    def test_exit_wait_is_capped_when_send_hangs(self, tmp_path):
        """DNS treo (timeout của requests không bao): lúc thoát chỉ chờ tới trần rồi bỏ,
        không giữ process lại suốt thời gian treo — và nói ra là đã bỏ."""
        alive, stderr = self._exit_right_after_send(tmp_path / "sent", send_seconds=30, flush_cap=0.5)
        assert not (tmp_path / "sent").exists()
        assert alive < 15  # khởi động + import vài giây, cộng trần 0,5s — không phải 30s
        assert "1 batch trace chưa gửi xong" in stderr

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

    def test_fail_fast_query_on_429_raises_without_backoff(self):
        """Server API: mọi key 429 -> EmbeddingUnavailable ngay, mỗi key thử đúng
        một lần, không qua tenacity (từng làm một câu hỏi mất 231s)."""
        from src.rag.embedder import EmbeddingUnavailable

        embedder = self._make()
        embedder.fail_fast_queries = True
        used = []

        def always_429(key, texts, task_type):
            used.append(key)
            raise RuntimeError("429 quota exceeded")

        with patch.object(type(embedder), "_call_api", staticmethod(always_429)), \
                patch("src.rag.embedder.time.sleep") as slept:
            with pytest.raises(EmbeddingUnavailable):
                embedder._get_query_embedding("cau hoi")
        slept.assert_not_called()
        assert used == ["k1", "k2"]

    def test_query_without_fail_fast_still_waits_for_quota(self):
        """Eval/ingest (cờ tắt) vẫn chờ quota như cũ."""
        import time as time_mod

        embedder = self._make(keys=("k1",))
        embedder._cooldown["k1"] = time_mod.monotonic() + 30

        with patch.object(type(embedder), "_call_api", staticmethod(lambda k, t, tt: [[0.1]])), \
                patch("src.rag.embedder.time.sleep") as slept:
            slept.side_effect = lambda _: embedder._cooldown.update(k1=0.0)
            assert embedder._get_query_embedding("cau hoi") == [0.1]
        slept.assert_called_once()

    @pytest.mark.asyncio
    async def test_aget_query_embedding_does_not_block_event_loop(self):
        """asyncio.to_thread bắt buộc: _get_query_embedding có thể time.sleep()
        hàng chục giây chờ quota (_reserve_key) — gọi trực tiếp (không qua
        thread) từng chặn CẢ event loop, đứng hình mọi WS/REST khác trên cùng
        server khi retrieval gặp lúc hết quota Gemini embedding."""
        embedder = self._make(keys=("k1",))

        def slow_call(key, texts, task_type):
            time.sleep(0.2)  # mô phỏng gọi mạng chậm / chờ quota
            return [[0.1, 0.2]]

        ticks = 0

        async def ticker():
            nonlocal ticks
            for _ in range(20):
                await asyncio.sleep(0.01)
                ticks += 1

        with patch.object(type(embedder), "_call_api", staticmethod(slow_call)):
            ticker_task = asyncio.create_task(ticker())
            result = await embedder._aget_query_embedding("cau hoi")
            ticker_task.cancel()

        assert result == [0.1, 0.2]
        assert ticks >= 10

class TestChromaHttpFailFast:
    """CHROMA_HOST trỏ server không chạy là mặc định ở local dev — không được
    bắt request đầu tiên trả giá 4.2s heartbeat cho điều đã biết trước."""

    def test_closed_port_is_reported_closed_quickly(self):
        import socket
        import time

        from src.rag.embedder import _port_is_open

        # Cổng 0 không bao giờ lắng nghe được; lấy một cổng đóng thật để chắc chắn.
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            closed_port = s.getsockname()[1]

        t0 = time.perf_counter()
        assert _port_is_open("127.0.0.1", closed_port, timeout=0.5) is False
        assert time.perf_counter() - t0 < 2.0

    def test_open_port_is_reported_open(self):
        import socket

        from src.rag.embedder import _port_is_open

        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            assert _port_is_open("127.0.0.1", server.getsockname()[1]) is True

    def test_http_client_is_not_built_when_port_is_closed(self, monkeypatch):
        """Bằng chứng thật sự: chromadb.HttpClient không được gọi."""
        import chromadb

        from src.config import get_settings
        from src.rag import embedder

        settings = get_settings()
        monkeypatch.setattr(settings, "chroma_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(settings, "chroma_port", 1, raising=False)
        monkeypatch.setattr(embedder, "_port_is_open", lambda *_a, **_k: False)

        def fail(*_args, **_kwargs):
            raise AssertionError("HttpClient không được dựng khi cổng đóng")

        monkeypatch.setattr(chromadb, "HttpClient", fail)
        # conftest trỏ DATA_DIR sang tmp_path nên collection rỗng — điều được
        # khẳng định ở đây là đã đi thẳng xuống PersistentClient, không qua HTTP.
        client, collection = embedder.get_chroma_collection(verify=False)
        assert collection.name == get_settings().chroma_collection
        assert isinstance(client, chromadb.api.client.Client)
