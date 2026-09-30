"""
Hotfix an ninh P0 (2026-09-27):
  1. /upload và /reload cần X-Admin-Key; upload bị chặn theo Content-Length TRƯỚC khi đọc body.
  2. Extract + chunk PDF chạy trong thread, có timeout; PDF > MAX_PDF_PAGES bị từ chối.
  3. Rate limit lấy IP ở phần tử CUỐI của X-Forwarded-For (client không tự chọn được).
  4. scripts/sanitize_qdrant_corpus.py chỉ xoá point user_upload thuộc tenant public.
"""

from __future__ import annotations

import importlib.util
import io
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from src.api.principal import admin_key_valid
from src.api.routes import documents
from src.api.routes.query import _get_client_ip

ADMIN_KEY = "test-admin-key-0123456789"
PDF_FILE = {"file": ("luat.pdf", b"%PDF-1.4 fake", "application/pdf")}


@pytest.fixture
def admin_env(monkeypatch):
    from src.config import get_settings

    monkeypatch.setenv("API_SECRET_KEY", ADMIN_KEY)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client(monkeypatch) -> TestClient:
    """Chỉ router documents; RAG init và audit log là giả — không chạm Gemini/Qdrant."""
    monkeypatch.setattr("src.api.main.ensure_rag_initialized", AsyncMock())
    monkeypatch.setattr(documents, "get_long_term_memory", MagicMock())
    app = FastAPI()
    app.include_router(documents.router)
    return TestClient(app)


@pytest.fixture
def form_spy(monkeypatch):
    """Ghi lại mọi lần route parse body multipart."""
    calls = []
    real_form = Request.form

    def spy(self, *a, **kw):
        calls.append(self.url.path)
        return real_form(self, *a, **kw)

    monkeypatch.setattr(Request, "form", spy)
    return calls


# ── 1. Admin key ──────────────────────────────────────────────────────────────

class TestAdminKey:
    @pytest.mark.parametrize("headers", [{}, {"X-Admin-Key": "wrong"}, {"X-Admin-Key": ""}])
    def test_upload_rejected_without_valid_key(self, admin_env, client, form_spy, headers):
        r = client.post("/api/v1/upload", files=PDF_FILE, headers=headers)
        assert r.status_code == 401
        assert r.json()["detail"] == "Unauthorized"
        # Điểm mấu chốt của hotfix: bị từ chối trước khi body được parse.
        assert form_spy == []

    @pytest.mark.parametrize("headers", [{}, {"X-Admin-Key": "wrong"}])
    def test_reload_rejected_without_valid_key(self, admin_env, client, headers):
        r = client.post("/api/v1/reload", headers=headers)
        assert r.status_code == 401
        assert r.json()["detail"] == "Unauthorized"

    def test_reload_accepted_with_valid_key(self, admin_env, client, monkeypatch):
        import src.rag.retriever as r_module

        monkeypatch.setattr(r_module, "_active_index", object())
        rebuild = MagicMock()
        monkeypatch.setattr(documents, "_rebuild_retriever", rebuild)
        r = client.post("/api/v1/reload", headers={"X-Admin-Key": ADMIN_KEY})
        assert r.status_code == 200
        rebuild.assert_called_once()

    def test_unset_secret_fails_closed(self, monkeypatch, client):
        """compare_digest("", "") là True — secret rỗng không được mở cửa cho header rỗng."""
        from src.config import get_settings

        monkeypatch.setenv("API_SECRET_KEY", "")
        get_settings.cache_clear()
        assert client.post("/api/v1/reload", headers={"X-Admin-Key": ""}).status_code == 401
        assert client.post("/api/v1/reload").status_code == 401

    def test_non_ascii_key_is_rejected_not_crashing(self, admin_env):
        # compare_digest trên str không phải ASCII ném TypeError → sẽ thành 500.
        assert admin_key_valid({"x-admin-key": "khóa-sai"}) is False
        assert admin_key_valid({"x-admin-key": ADMIN_KEY}) is True


# ── 1b. Content-Length ────────────────────────────────────────────────────────

class TestContentLength:
    def test_default_limit_is_10mb(self, monkeypatch):
        from src.config import Settings

        monkeypatch.delenv("MAX_UPLOAD_SIZE_MB", raising=False)
        assert Settings(_env_file=None).max_upload_size_mb == 10

    def test_oversized_body_rejected_before_parse(self, admin_env, client, form_spy, monkeypatch):
        from src.config import get_settings

        monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "1")
        get_settings.cache_clear()
        big = {"file": ("big.pdf", b"%PDF-" + b"0" * (1024 * 1024 + 100 * 1024), "application/pdf")}
        r = client.post("/api/v1/upload", files=big, headers={"X-Admin-Key": ADMIN_KEY})
        assert r.status_code == 413
        assert form_spy == []

    @pytest.mark.parametrize("headers,code", [
        ({}, 411),
        ({"content-length": "abc"}, 400),
        ({"content-length": "-1"}, 413),
        ({"content-length": str(10**12)}, 413),
    ])
    def test_header_checks(self, headers, code):
        with pytest.raises(HTTPException) as exc:
            documents._check_content_length(SimpleNamespace(headers=headers))
        assert exc.value.status_code == code

    def test_within_limit_passes(self):
        documents._check_content_length(SimpleNamespace(headers={"content-length": "2048"}))


# ── 2. Thread + timeout + page cap ────────────────────────────────────────────

def _blank_pdf(pages: int) -> bytes:
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for _ in range(pages):
        c.showPage()
    c.save()
    return buf.getvalue()


class TestPdfProcessing:
    def test_valid_upload_runs_off_the_event_loop_thread(self, admin_env, client, monkeypatch):
        seen = {}

        def fake_load(data, filename):
            seen["load"] = threading.get_ident()
            return {"title": "luat", "content": "Điều 1. ..."}

        def fake_chunk(content, doc):
            seen["chunk"] = threading.get_ident()
            return ["c1", "c2", "c3"]

        monkeypatch.setattr(documents, "load_pdf", fake_load)
        monkeypatch.setattr(documents, "chunk_by_dieu", fake_chunk)
        monkeypatch.setattr(documents, "_index_chunks", AsyncMock(return_value=3))

        r = client.post("/api/v1/upload", files=PDF_FILE, headers={"X-Admin-Key": ADMIN_KEY})
        assert r.status_code == 201, r.text
        assert r.json()["indexed_chunks"] == 3
        # TestClient chạy event loop ở một thread riêng; so với thread của loop đó
        # thì khó — nhưng load/chunk chạy chung MỘT worker thread, khác thread test.
        assert seen["load"] == seen["chunk"] != threading.get_ident()

    def test_slow_pdf_times_out(self, admin_env, client, monkeypatch):
        monkeypatch.setattr(documents, "PDF_PROCESSING_TIMEOUT_S", 0.2)

        def slow_load(data, filename):
            time.sleep(0.6)
            return None

        monkeypatch.setattr(documents, "load_pdf", slow_load)
        r = client.post("/api/v1/upload", files=PDF_FILE, headers={"X-Admin-Key": ADMIN_KEY})
        assert r.status_code == 422
        assert r.json()["detail"] == "PDF processing timed out"

    def test_pdf_over_page_limit_rejected(self):
        from src.ingestion.loader import MAX_PDF_PAGES, PdfTooManyPages, load_pdf

        with pytest.raises(PdfTooManyPages):
            load_pdf(_blank_pdf(MAX_PDF_PAGES + 1), "qua_dai.pdf")

    def test_pdf_at_page_limit_not_rejected(self):
        from src.ingestion.loader import MAX_PDF_PAGES, load_pdf

        # Trang trắng → không đủ text → None; điều cần kiểm là KHÔNG ném lỗi giới hạn trang.
        assert load_pdf(_blank_pdf(MAX_PDF_PAGES), "vua_du.pdf") is None

    def test_page_limit_maps_to_400(self, admin_env, client):
        files = {"file": ("qua_dai.pdf", _blank_pdf(301), "application/pdf")}
        r = client.post("/api/v1/upload", files=files, headers={"X-Admin-Key": ADMIN_KEY})
        assert r.status_code == 400
        assert "300" in r.json()["detail"]

    def test_missing_file_field(self, admin_env, client):
        r = client.post("/api/v1/upload", data={"other": "x"}, headers={"X-Admin-Key": ADMIN_KEY})
        assert r.status_code == 422


# ── 3. Client IP ──────────────────────────────────────────────────────────────

def _req(xff: str | None, peer: str = "10.0.0.1") -> Request:
    headers = [(b"x-forwarded-for", xff.encode())] if xff is not None else []
    return Request({"type": "http", "headers": headers, "client": (peer, 1234)})


class TestClientIp:
    def test_uses_last_hop_not_client_supplied_first(self):
        assert _get_client_ip(_req("6.6.6.6, 203.0.113.7")) == "203.0.113.7"

    def test_spoofed_prefix_does_not_change_bucket(self):
        assert _get_client_ip(_req("1.1.1.1, 203.0.113.7")) == _get_client_ip(_req("2.2.2.2, 203.0.113.7"))

    def test_single_entry(self):
        assert _get_client_ip(_req("203.0.113.7")) == "203.0.113.7"

    @pytest.mark.parametrize("xff", [None, "", "  ", "1.1.1.1, "])
    def test_falls_back_to_peer(self, xff):
        assert _get_client_ip(_req(xff, peer="198.51.100.9")) == "198.51.100.9"


# ── 4. Cleanup script ─────────────────────────────────────────────────────────

def _load_cleanup_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "sanitize_qdrant_corpus.py"
    spec = importlib.util.spec_from_file_location("sanitize_qdrant_corpus", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeQdrant:
    def __init__(self, pages):
        self.pages = pages
        self.filters = []
        self.deleted = []

    def scroll(self, *, collection_name, scroll_filter, limit, offset, with_payload, with_vectors):
        self.filters.append(scroll_filter)
        i = offset or 0
        nxt = i + 1 if i + 1 < len(self.pages) else None
        return self.pages[i], nxt

    def delete(self, *, collection_name, points_selector, wait):
        self.deleted.append(list(points_selector.points))


def _pt(pid, source, title="t"):
    return SimpleNamespace(id=pid, payload={"source": source, "title": title})


class TestCleanupScript:
    def test_selects_only_user_uploads_across_pages(self):
        mod = _load_cleanup_script()
        fake = FakeQdrant([
            [_pt(1, "user_upload", "gia_mao"), _pt(2, "manifest"), _pt(3, "")],
            [_pt(4, "user_upload", "gia_mao"), SimpleNamespace(id=5, payload=None)],
        ])
        found = mod.find_poison_points(fake, "documind_legal")
        assert [pid for pid, _ in found] == [1, 4]
        assert len(fake.filters) == 2  # đi hết mọi trang

    def test_server_filter_scopes_to_public_tenant(self):
        mod = _load_cleanup_script()
        f = mod._public_filter()
        keys = {getattr(c, "key", None) or c.is_empty.key for c in f.should}
        assert keys == {"tenant_id"}
        match = next(c for c in f.should if getattr(c, "key", None))
        assert match.match.value == "public"
        # `source` không có payload index — lọc nó phía server là 400 dưới strict mode.
        assert f.must is None

    def test_delete_batches(self):
        mod = _load_cleanup_script()
        fake = FakeQdrant([])
        assert mod.delete_points(fake, "c", list(range(600))) == 600
        assert [len(b) for b in fake.deleted] == [256, 256, 88]
