"""
Tests for ingestion pipeline: cleaner, chunker, loader, crawler.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.cleaner import clean_legal_text, normalize_unicode, remove_noise
from src.ingestion.chunker import LegalChunk, chunk_by_dieu, _fallback_chunks
from src.ingestion.loader import _validate_pdf_bytes


# ── Cleaner Tests ──────────────────────────────────────────────────────────────

class TestCleaner:
    def test_normalize_unicode_removes_bom(self):
        result = normalize_unicode("﻿Hello")
        assert "﻿" not in result

    def test_normalize_unicode_replaces_curly_quotes(self):
        result = normalize_unicode("“xin chào”")
        assert '"' in result

    def test_remove_noise_collapses_whitespace(self):
        result = remove_noise("text    with    spaces")
        assert "    " not in result

    def test_clean_legal_text_empty_returns_empty(self):
        assert clean_legal_text("") == ""
        assert clean_legal_text("   ") == ""

    def test_clean_legal_text_adds_newlines_before_dieu(self):
        text = "some textĐiều 1. Title\nsome more text"
        result = clean_legal_text(text)
        assert "\n\nĐiều" in result

    def test_clean_legal_text_handles_page_numbers(self):
        text = "Trang 1 / 10\nĐiều 1. Title"
        result = clean_legal_text(text)
        assert "Trang 1 / 10" not in result

    def test_clean_legal_text_preserves_vietnamese(self):
        text = "Điều 5. Quyền của doanh nghiệp"
        result = clean_legal_text(text)
        assert "Điều 5" in result
        assert "doanh nghiệp" in result


# ── Chunker Tests ──────────────────────────────────────────────────────────────

class TestChunker:
    def test_chunk_by_dieu_extracts_articles(self, sample_legal_text, sample_doc_meta):
        chunks = chunk_by_dieu(sample_legal_text, sample_doc_meta)
        assert len(chunks) == 3  # Điều 1, 2, 3
        assert all(isinstance(c, LegalChunk) for c in chunks)

    def test_chunk_metadata_has_required_keys(self, sample_legal_text, sample_doc_meta):
        chunks = chunk_by_dieu(sample_legal_text, sample_doc_meta)
        required_keys = {"source_url", "title", "doc_type", "dieu_header", "char_count"}
        for chunk in chunks:
            assert required_keys.issubset(chunk.metadata.keys())

    def test_chunk_dieu_header_is_correct(self, sample_legal_text, sample_doc_meta):
        chunks = chunk_by_dieu(sample_legal_text, sample_doc_meta)
        headers = [c.metadata["dieu_header"] for c in chunks]
        assert any("Điều 1" in h for h in headers)
        assert any("Điều 2" in h for h in headers)

    def test_chunk_filters_short_texts(self, sample_doc_meta):
        short_text = "Điều 1. Ngắn"
        chunks = chunk_by_dieu(short_text, sample_doc_meta)
        assert all(c.char_count >= 50 for c in chunks)

    def test_fallback_chunks_for_plain_text(self, sample_doc_meta):
        plain = "This is plain text without legal structure.\n\n" * 20
        chunks = _fallback_chunks(plain, sample_doc_meta)
        assert len(chunks) > 0
        assert all(c.metadata.get("chunk_strategy") == "fallback_window" for c in chunks)

    def test_chunk_is_valid_property(self):
        valid = LegalChunk(text="a" * 50, metadata={})
        invalid = LegalChunk(text="short", metadata={})
        assert valid.is_valid
        assert not invalid.is_valid

    def test_chunk_by_dieu_fallback_when_no_dieu(self, sample_doc_meta):
        no_dieu = "This document has no article structure at all.\n\n" * 10
        chunks = chunk_by_dieu(no_dieu, sample_doc_meta)
        assert len(chunks) > 0  # fallback returns results


# ── Loader Tests ───────────────────────────────────────────────────────────────

class TestLoader:
    def test_validate_pdf_bytes_rejects_non_pdf(self):
        fake_pdf = b"NOT A PDF HEADER"
        with pytest.raises(ValueError, match="not a valid PDF"):
            _validate_pdf_bytes(fake_pdf)

    def test_validate_pdf_bytes_accepts_valid_magic(self):
        valid_pdf = b"%PDF-1.4" + b"\x00" * 100
        # Should not raise
        _validate_pdf_bytes(valid_pdf)

    def test_validate_pdf_bytes_rejects_oversized(self, monkeypatch):
        monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "1")
        from src.config import get_settings
        get_settings.cache_clear()

        large = b"%PDF-1.4" + b"\x00" * (2 * 1024 * 1024)  # 2MB
        with pytest.raises(ValueError, match="exceeds"):
            _validate_pdf_bytes(large)

    def test_load_json_file_returns_none_on_empty(self, tmp_path):
        from src.ingestion.loader import load_json_file

        empty_json = tmp_path / "empty.json"
        empty_json.write_text('{"title": "test", "content": ""}', encoding="utf-8")
        result = load_json_file(empty_json)
        assert result is None

    def test_load_json_file_returns_none_on_invalid_json(self, tmp_path):
        from src.ingestion.loader import load_json_file

        bad_json = tmp_path / "bad.json"
        bad_json.write_text("not valid json", encoding="utf-8")
        result = load_json_file(bad_json)
        assert result is None


# ── Crawler Tests ──────────────────────────────────────────────────────────────

class TestCrawler:
    def test_safe_filename_sanitizes_special_chars(self):
        from src.ingestion.crawler import VbplCrawler

        crawler = VbplCrawler(output_dir=Path("/tmp"))
        result = crawler._safe_filename("Luật../../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_safe_filename_max_length(self):
        from src.ingestion.crawler import VbplCrawler

        crawler = VbplCrawler(output_dir=Path("/tmp"))
        long_name = "A" * 200
        result = crawler._safe_filename(long_name)
        assert len(result) <= 80

    def test_fetch_document_rejects_non_vbpl_url(self, tmp_path):
        from src.ingestion.crawler import VbplCrawler

        crawler = VbplCrawler(output_dir=tmp_path)
        result = crawler.fetch_document("https://evil.com/path")
        assert result is None
