"""P0-2: khoá rate limit không giả mạo được qua X-Forwarded-For.

Proxy của Render NỐI địa chỉ nó thấy vào CUỐI header; mọi phần tử phía trước là
do client tự gửi. Lấy phần tử đầu = client tự chọn bucket rate limit của mình.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.api.routes.query import _get_client_ip


def _req(xff: str | None = None, host: str = "10.0.0.9") -> Request:
    headers = [(b"x-forwarded-for", xff.encode())] if xff is not None else []
    return Request({"type": "http", "headers": headers, "client": (host, 1234)})


class TestGetClientIp:
    def test_takes_last_hop_not_first(self):
        assert _get_client_ip(_req("8.8.8.8, 1.1.1.1")) == "1.1.1.1"

    def test_spoofed_prefix_does_not_change_key(self):
        a = _get_client_ip(_req("8.8.8.8, 1.1.1.1"))
        b = _get_client_ip(_req("9.9.9.9, 7.7.7.7, 1.1.1.1"))
        assert a == b == "1.1.1.1"

    def test_single_hop(self):
        assert _get_client_ip(_req("1.1.1.1")) == "1.1.1.1"

    def test_no_header_falls_back_to_socket_peer(self):
        assert _get_client_ip(_req()) == "10.0.0.9"

    def test_blank_or_empty_entries_fall_back(self):
        assert _get_client_ip(_req(" , ")) == "10.0.0.9"


def test_rotating_spoofed_prefix_still_hits_rate_limit():
    """Kịch bản tấn công thật: đổi phần tử đầu mỗi request để có bucket mới."""
    limiter = Limiter(key_func=_get_client_ip, default_limits=["2/minute"])
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    @app.get("/ping")
    async def ping() -> dict:
        return {"ok": True}

    client = TestClient(app)
    codes = [
        client.get("/ping", headers={"x-forwarded-for": f"6.6.6.{i}, 1.1.1.1"}).status_code
        for i in range(3)
    ]
    assert codes == [200, 200, 429]
