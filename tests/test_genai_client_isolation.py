"""P1.3: mỗi lời gọi Gemini đi đúng API key của nó, kể cả khi nhiều thread chạy song song.

`genai.configure(api_key=...)` là trạng thái global của module google.generativeai:
thread A đặt key1, thread B đặt key2, request của A đi bằng key2. Hậu quả: 429 bị
ghi cho nhầm cặp (key, model) nên cooldown khoá sai cặp, quota tiêu lệch key.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import google.ai.generativelanguage as glm
import google.generativeai as genai
import pytest

from src.rag import embedder as embedder_mod
from src.rag import generator


class _FakeClient:
    """GenerativeServiceClient giả: trả lời bằng chính key đã tạo ra nó."""

    def __init__(self, client_options=None, **_kw):
        self.key = client_options["api_key"]

    def generate_content(self, request, **_kw):
        part = glm.Part(text=self.key)
        cand = glm.Candidate(content=glm.Content(parts=[part], role="model"), finish_reason=1)
        return glm.GenerateContentResponse(candidates=[cand])


@pytest.fixture(autouse=True)
def fake_clients(monkeypatch):
    def forbid(*_a, **_kw):
        raise AssertionError("genai.configure là global state — không được gọi trên đường chạy")

    monkeypatch.setattr(genai, "configure", forbid)
    monkeypatch.setattr(glm, "GenerativeServiceClient", _FakeClient)
    embedder_mod.genai_client.cache_clear()
    generator._PAIR_DOWN_UNTIL.clear()
    yield
    embedder_mod.genai_client.cache_clear()


def test_gemini_call_uses_its_own_key():
    assert generator.gemini_call("key-A", "m", "hi").text == "key-A"


def test_concurrent_calls_never_cross_keys():
    keys = [f"key-{i}" for i in range(6)]
    barrier = threading.Barrier(len(keys))

    def call(key: str) -> tuple[str, str]:
        barrier.wait()  # dồn các lời gọi vào cùng một thời điểm
        got = [generator.gemini_call(key, "m", "hi").text for _ in range(50)]
        return key, got

    with ThreadPoolExecutor(len(keys)) as pool:
        for key, got in pool.map(call, keys):
            assert set(got) == {key}


def test_client_is_cached_per_key():
    assert embedder_mod.genai_client("k1") is embedder_mod.genai_client("k1")
    assert embedder_mod.genai_client("k1") is not embedder_mod.genai_client("k2")


def test_embedder_passes_per_key_client(monkeypatch):
    seen = {}

    def fake_embed_content(*, model, content, task_type, client):
        seen["key"] = client.key
        return {"embedding": [[0.1, 0.2] for _ in content]}

    monkeypatch.setattr(genai, "embed_content", fake_embed_content)
    emb = embedder_mod._GeminiAPIEmbedding(model_name="gemini-embedding-001", api_keys=["key-E"])
    assert emb._call_api("key-E", ["xin chào"], "retrieval_query") == [[0.1, 0.2]]
    assert seen["key"] == "key-E"
