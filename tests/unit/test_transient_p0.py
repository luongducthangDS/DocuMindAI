"""Q09 (P0) — phân loại lỗi Gemini: tạm thời (xoay cặp key × model) hay lỗi thật.

Bản cũ so substring: "rate" khớp "gene*rate*Content", "sepa*rate*", "accu*rate*",
nên 400 "API key not valid" bị coi là tạm thời → xoay hết các cặp, cho vào
cooldown, rồi rơi xuống extractive fallback: lỗi cấu hình trông như "LLM quá tải".
"""

from __future__ import annotations

import httpx
import pytest
import requests
from google.api_core import exceptions as gexc

from src.rag import generator
from src.rag.generator import _error_kind, _is_transient

NOT_TRANSIENT = [
    gexc.InvalidArgument("API key not valid. Please pass a valid API key."),
    gexc.PermissionDenied("Permission denied on resource project"),
    gexc.Forbidden("403 Forbidden"),
    gexc.Unauthenticated("Request had invalid authentication credentials"),
    RuntimeError("400 POST https://generativelanguage.googleapis.com/v1beta/models/"
                 "gemini-2.5-flash:generateContent: API key not valid"),
    RuntimeError("400 Request contains an invalid argument: separate parts"),
    RuntimeError("403 Forbidden: accurate billing account required"),
    ValueError("Invalid operation: the response.text quick accessor requires a valid Part"),
]

TRANSIENT = [
    (gexc.ResourceExhausted("Resource has been exhausted (e.g. check quota)."), "key"),
    (gexc.TooManyRequests("429 Too Many Requests"), "key"),
    (RuntimeError("429 You exceeded your current quota"), "key"),
    (Exception("quota exhausted"), "key"),
    (gexc.ServiceUnavailable("The model is overloaded"), "model"),
    (RuntimeError("503 This model is currently experiencing high demand"), "model"),
    (gexc.DeadlineExceeded("Deadline Exceeded"), "model"),
    (requests.exceptions.ReadTimeout("HTTPSConnectionPool: Read timed out."), "model"),
    (httpx.ReadTimeout("timed out"), "model"),
    (TimeoutError(), "model"),
]


@pytest.mark.parametrize("exc", NOT_TRANSIENT, ids=lambda e: f"{type(e).__name__}:{str(e)[:30]}")
def test_real_errors_are_not_transient(exc):
    assert _is_transient(exc) is False


@pytest.mark.parametrize("exc,kind", TRANSIENT, ids=lambda x: str(x)[:30] if not isinstance(x, str) else x)
def test_transient_errors_rotate_with_the_right_scope(exc, kind):
    assert _is_transient(exc) is True
    assert _error_kind(exc) == kind


# ── Hành vi đầu-cuối qua gemini_generate (SDK giả, code xoay vòng thật) ──────

def test_invalid_key_raises_immediately_without_rotating(fake_gemini):
    fake_gemini.responder = lambda p, m: gexc.InvalidArgument("API key not valid")
    with pytest.raises(gexc.InvalidArgument):
        generator.gemini_generate("câu hỏi")
    assert len(fake_gemini.calls) == 1, "lỗi thật không được xoay sang cặp khác"
    assert generator._PAIR_DOWN_UNTIL == {}, "lỗi thật không được đẩy cặp nào vào cooldown"


def test_quota_on_one_key_rotates_and_cools_down_only_that_key(fake_gemini):
    def responder(prompt, model):
        if fake_gemini._key == "key-A":
            return gexc.ResourceExhausted("quota")
        return "ok"

    fake_gemini.responder = responder
    assert generator.gemini_generate("câu hỏi") == "ok"
    assert [c.api_key for c in fake_gemini.calls] == ["key-A", "key-B"]
    assert set(generator._PAIR_DOWN_UNTIL) == {("key-A", "model-x")}


def test_overloaded_model_cools_down_for_every_key(fake_gemini):
    def responder(prompt, model):
        return gexc.ServiceUnavailable("overloaded") if model == "model-x" else "ok"

    fake_gemini.responder = responder
    assert generator.gemini_generate("câu hỏi") == "ok"
    # 1 lần thử model-x là đủ biết cả model đang quá tải — không chờ lần lượt 3 key.
    assert [c.model for c in fake_gemini.calls] == ["model-x", "model-y"]
    assert set(generator._PAIR_DOWN_UNTIL) == {("key-A", "model-x"), ("key-B", "model-x"), ("key-C", "model-x")}
