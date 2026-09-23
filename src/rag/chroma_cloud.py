"""Chroma Cloud qua REST v2 (dùng bởi scripts/copy_chroma_to_cloud.py, eval/provider_bench.py).

Không dùng `chromadb.CloudClient`: client 0.6.3 trong venv không parse được response
của server Cloud (1.x) — KeyError '_type' (đã thử thật 2026-09-23). Nâng chromadb thì
đụng store local + DLL Rust (Smart App Control). App production KHÔNG đọc từ đây.
"""

from __future__ import annotations

import requests

from src.config import get_settings

API = "https://api.trychroma.com"


class ChromaCloud:
    def __init__(self) -> None:
        s = get_settings()
        if not (s.chroma_cloud_api_key and s.chroma_cloud_tenant and s.chroma_cloud_database):
            raise RuntimeError(
                "Thiếu CHROMA_CLOUD_API_KEY / CHROMA_CLOUD_TENANT / CHROMA_CLOUD_DATABASE trong .env"
            )
        self.base = (f"{API}/api/v2/tenants/{s.chroma_cloud_tenant}"
                     f"/databases/{s.chroma_cloud_database}/collections")
        self.http = requests.Session()
        self.http.headers["x-chroma-token"] = s.chroma_cloud_api_key

    def call(self, method: str, path: str = "", **kw):
        r = self.http.request(method, f"{self.base}{path}", timeout=120, **kw)
        if not r.ok:
            raise RuntimeError(f"{method} {path or '/'} -> {r.status_code}: {r.text[:300]}")
        return r.json()

    def collection_id(self, name: str) -> str:
        for c in self.call("GET"):
            if c["name"] == name:
                return c["id"]
        raise RuntimeError(f"Chroma Cloud không có collection '{name}'")
