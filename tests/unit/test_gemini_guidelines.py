"""
Unit tests validating compliance with gemini.md guidelines:
1. Default Model: gemini-3.8-flash for generation/reasoning.
2. High-speed / Lightweight: gemini-3.5-flash-lite for classification, extraction.
3. RAG & Vector Search: gemini-embedding-2.
4. Deprecated Notice: No gemini-2.5-flash or gemini-2.5-pro in configuration.
5. Error Handling: Exponential backoff on 429 and fallback to 3.8-flash on 503 from 3.5-flash.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from google.api_core import exceptions as gexc

from src.config import Settings
from src.rag import generator
from src.rag.generator import (
    _backoff_jitter_wait,
    _is_429,
    _is_503,
    gemini_call,
    gemini_generate,
)


def test_guideline_model_settings():
    """Verify default models adhere strictly to gemini.md."""
    settings = Settings(
        google_api_key="test_key",
        _env_file=None,  # Don't read local .env to test class defaults
    )
    # Guideline 1: Default Model contains gemini-3.8-flash as primary
    gen_models = [m.strip() for m in settings.gemini_generation_models.split(",")]
    assert gen_models[0] == "gemini-3.8-flash"

    # Guideline 2: High-speed / Lightweight is gemini-3.5-flash-lite
    assert "gemini-3.5-flash-lite" in settings.gemini_judge_models

    # Guideline 3: Embedding model is gemini-embedding-2
    assert settings.embedding_model == "gemini-embedding-2"

    # Guideline 4: No deprecated 2.5 models
    for m in gen_models:
        assert not m.startswith("gemini-2.5-flash")
        assert not m.startswith("gemini-2.5-pro")


def test_backoff_jitter_wait():
    """Verify exponential backoff calculation: wait_time = (2 ** attempt) + random(0, 1)."""
    for attempt in range(4):
        wait = _backoff_jitter_wait(attempt)
        base = 2 ** attempt
        assert base <= wait <= base + 1.0


def test_is_429_detection():
    assert _is_429(gexc.ResourceExhausted("429 Resource exhausted"))
    assert _is_429(gexc.TooManyRequests("Too many requests"))
    assert _is_429(RuntimeError("429 quota exceeded"))
    assert not _is_429(RuntimeError("500 Server Error"))


def test_is_503_detection():
    assert _is_503(gexc.ServiceUnavailable("503 Service Unavailable"))
    assert _is_503(RuntimeError("The model is overloaded"))
    assert _is_503(RuntimeError("503 High demand"))
    assert not _is_503(RuntimeError("400 Bad Request"))


def test_gemini_call_retries_on_429_with_exponential_backoff(monkeypatch):
    """gemini_call retries with backoff when max_429_retries > 0."""
    monkeypatch.setattr(generator, "_PAIR_DOWN_UNTIL", {})
    attempts = 0

    class FakeModel:
        def generate_content(self, *args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise gexc.ResourceExhausted("429 Resource exhausted")
            return MagicMock(text="success")

    monkeypatch.setattr("google.generativeai.GenerativeModel", lambda *a, **kw: FakeModel())
    monkeypatch.setattr("src.rag.embedder.genai_client", lambda *a: MagicMock())

    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    res = gemini_call("test_key", "gemini-3.8-flash", "test_prompt", max_429_retries=3)
    assert res.text == "success"
    assert attempts == 3
    assert len(sleep_calls) == 2
    # Verify exponential backoff intervals
    assert 1.0 <= sleep_calls[0] <= 2.0  # 2^0 + [0, 1]
    assert 2.0 <= sleep_calls[1] <= 3.0  # 2^1 + [0, 1]


def test_gemini_generate_fallback_from_35_to_38_on_503(monkeypatch):
    """When gemini-3.5-flash gets 503, gemini_generate falls back to gemini-3.8-flash."""
    monkeypatch.setattr(generator, "_PAIR_DOWN_UNTIL", {})
    monkeypatch.setattr(generator, "_GEMINI_PAIR_CURSOR", 0)
    monkeypatch.setattr(generator, "_gemini_keys", lambda: ["key1"])

    called_models = []

    def mock_gemini_call(api_key, model_name, prompt, **kwargs):
        called_models.append(model_name)
        if model_name.startswith("gemini-3.5-flash"):
            raise gexc.ServiceUnavailable("503 The model is overloaded")
        return MagicMock(text="response from fallback 3.8-flash", usage_metadata=None)

    monkeypatch.setattr(generator, "gemini_call", mock_gemini_call)
    monkeypatch.setattr(generator, "record_generation", lambda *a, **kw: None)

    # Initial model is only 3.5-flash
    res = gemini_generate("test prompt", models=["gemini-3.5-flash-lite"])
    assert res == "response from fallback 3.8-flash"
    assert called_models == ["gemini-3.5-flash-lite", "gemini-3.8-flash"]


@pytest.mark.asyncio
async def test_stream_answer_fallback_from_35_to_38_on_503(monkeypatch):
    """When stream_answer hits 503 on 3.5-flash, it falls back to 3.8-flash."""
    from src.rag.generator import stream_answer
    from src.rag.retriever import RetrievedChunk

    monkeypatch.setattr(generator, "_PAIR_DOWN_UNTIL", {})
    monkeypatch.setattr(generator, "_GEMINI_PAIR_CURSOR", 0)
    monkeypatch.setattr(generator, "_gemini_keys", lambda: ["key1"])
    monkeypatch.setattr(generator, "record_generation", lambda *a, **kw: None)
    monkeypatch.setattr(generator, "_effective_min_score", lambda: 0.0)

    called_models = []

    def mock_gemini_call(api_key, model_name, prompt, stream=False, **kwargs):
        called_models.append(model_name)
        if model_name.startswith("gemini-3.5-flash"):
            raise gexc.ServiceUnavailable("503 The model is overloaded")

        class FakeStream(list):
            usage_metadata = None

        return FakeStream([MagicMock(text="streamed from 3.8-flash")])

    monkeypatch.setattr(generator, "gemini_call", mock_gemini_call)

    # Force initial models to 3.5-flash
    monkeypatch.setattr(generator, "_select_models", lambda is_complex: ["gemini-3.5-flash-lite"])

    chunk = RetrievedChunk(text="legal content", score=0.9, metadata={"title": "Law 1"})
    tokens = [t async for t in stream_answer("test query", [chunk])]
    assert "".join(tokens) == "streamed from 3.8-flash"
    assert called_models == ["gemini-3.5-flash-lite", "gemini-3.8-flash"]


def test_agent_graph_get_llm_uses_gemini_38(monkeypatch):
    """_get_llm in graph.py uses gemini-3.8-flash."""
    from src.agent import graph

    init_args = {}

    class FakeChatGoogle:
        def __init__(self, **kwargs):
            init_args.update(kwargs)

        def bind_tools(self, tools):
            return self

    monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", FakeChatGoogle)
    monkeypatch.setattr(graph, "get_settings", lambda: MagicMock(google_api_key="fake_key"))

    graph._get_llm()
    assert init_args.get("model") == "gemini-3.8-flash"
