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


# ── Manifest Tests (corpus lao động/BHXH) ──────────────────────────────────────

MANIFEST_PATH = Path("docs/corpus/corpus_manifest.yaml")


@pytest.fixture(scope="module")
def manifest() -> dict:
    from src.ingestion.manifest import load_manifest

    return load_manifest(MANIFEST_PATH)


class TestManifest:
    def test_load_manifest_returns_locked_documents(self, manifest):
        from src.ingestion.manifest import DocEntry

        assert len(manifest) >= 18
        assert all(isinstance(e, DocEntry) for e in manifest.values())
        assert "45-2019-QH14" in manifest          # BLLĐ
        assert "41-2024-QH15" in manifest          # Luật BHXH 2024

    def test_skips_documents_outside_locked_list(self, manifest):
        # nhóm F / optional đã hoãn sang vòng sau — không được lọt vào corpus
        for deferred in ("12-2022-ND-CP", "152-2020-ND-CP", "88-2020-ND-CP", "58-2020-ND-CP"):
            assert deferred not in manifest

    def test_every_locked_document_is_verified(self, manifest):
        # CORPUS_SPEC §2: chỉ record VERIFIED mới được index
        unverified = [d for d, e in manifest.items() if not e.is_verified]
        assert unverified == []

    def test_effective_window_uses_open_sentinel(self, manifest):
        from src.ingestion.manifest import EFFECTIVE_TO_OPEN

        assert manifest["145-2020-ND-CP"].effective_to == EFFECTIVE_TO_OPEN  # còn hiệu lực
        assert manifest["58-2014-QH13"].effective_to == "2025-07-01"         # bị 41/2024 thay

    def test_dates_are_iso_strings_not_date_objects(self, manifest):
        # ChromaDB metadata chỉ nhận str/int/float/bool
        for entry in manifest.values():
            assert isinstance(entry.ngay_hieu_luc, str)
            assert isinstance(entry.effective_to, str)

    def test_in_place_amendments_parsed_for_bo_luat_lao_dong(self, manifest):
        amendments = manifest["45-2019-QH14"].in_place_amended_clauses
        # T4: đúng 4 khoản/điểm bị sửa in-place
        assert len(amendments) == 4
        by_uid = {a.clause_uid: a for a in amendments}
        d139 = by_uid["45-2019-QH14__d139_k1"]
        assert (d139.dieu, d139.khoan) == (139, 1)
        assert d139.effective_from == "2026-07-01"
        assert d139.amended_by_doc == "113/2025/QH15"
        assert d139.has_old_version

    def test_newly_inserted_clause_has_no_old_version(self, manifest):
        by_uid = {a.clause_uid: a for a in manifest["45-2019-QH14"].in_place_amended_clauses}
        # Điều 154 khoản 8a do 71/2025 BỔ SUNG — không tồn tại bản cũ để dựng version
        assert by_uid["45-2019-QH14__d154_k8a"].has_old_version is False

    def test_amendment_for_filters_by_dieu(self, manifest):
        entry = manifest["45-2019-QH14"]
        assert len(entry.amendment_for(139)) == 1
        assert entry.amendment_for(1) == ()

    def test_bo_luat_lao_dong_text_comes_from_consolidated_file(self, manifest):
        entry = manifest["45-2019-QH14"]
        # text hiện hành lấy từ VBHN (đã gồm 4 sửa đổi); bản 2019 gốc chỉ dùng dựng version cũ
        assert entry.source_file == "18-VBHN-VPQH.md"
        assert entry.consolidated_from == "18/VBHN-VPQH"
        assert entry.file_ban_goc == "45-2019-QH14.md"

    def test_source_file_defaults_to_doc_id(self, manifest):
        assert manifest["145-2020-ND-CP"].source_file == "145-2020-ND-CP.md"

    def test_source_files_exist_on_disk(self, manifest):
        corpus_dir = Path("data/raw/lao_dong")
        missing = [e.source_file for e in manifest.values() if not (corpus_dir / e.source_file).exists()]
        assert missing == []


# ── Clause-level chunking + versions (T8/T9) ───────────────────────────────────

CORPUS_DIR = Path("data/raw/lao_dong")


@pytest.fixture(scope="module")
def bo_luat_chunks(manifest):
    """Chunks of Bộ luật Lao động built exactly like the ingest script does."""
    from scripts.ingest_documents import build_clause_chunks

    return build_clause_chunks(manifest["45-2019-QH14"], CORPUS_DIR, "2026-09-12")


class TestClauseChunking:
    def test_every_chunk_has_clause_and_version_fields(self, bo_luat_chunks):
        required = {"doc_id", "clause_uid", "version_id", "effective_from",
                    "effective_to", "status", "dieu", "dieu_tieu_de"}
        for chunk in bo_luat_chunks:
            assert required.issubset(chunk.metadata.keys())
            assert chunk.metadata["clause_uid"]
            assert chunk.metadata["version_id"]

    def test_legacy_metadata_still_present(self, bo_luat_chunks):
        # retriever/generator hiện tại đọc các trường này — không được mất
        legacy = {"source_url", "title", "doc_type", "so_hieu", "dieu_header",
                  "khoan_count", "char_count", "source"}
        assert legacy.issubset(bo_luat_chunks[0].metadata.keys())

    def test_version_id_follows_convention(self, bo_luat_chunks):
        for chunk in bo_luat_chunks:
            md = chunk.metadata
            assert md["version_id"] == f"{md['clause_uid']}__v{md['effective_from']}"

    def test_unamended_dieu_stays_one_chunk(self, bo_luat_chunks):
        d20 = [c for c in bo_luat_chunks if c.metadata["clause_uid"] == "45-2019-QH14__d20"]
        assert len(d20) == 1
        assert d20[0].metadata["khoan"] == ""        # cấp điều, không tách khoản

    def test_amended_dieu_is_split_down_to_khoan(self, bo_luat_chunks):
        uids = {c.metadata["clause_uid"] for c in bo_luat_chunks}
        # 4 khoản/điểm bị sửa in-place (T4) đều phải có chunk riêng
        assert "45-2019-QH14__d139_k1" in uids
        assert "45-2019-QH14__d154_k8a" in uids
        assert "45-2019-QH14__d61_k3" in uids
        assert "45-2019-QH14__d59_k2_pa" in uids

    def test_amended_khoan_carries_its_own_effective_date(self, bo_luat_chunks):
        d139_k1 = [c for c in bo_luat_chunks
                   if c.metadata["clause_uid"] == "45-2019-QH14__d139_k1"]
        by_date = {c.metadata["effective_from"]: c for c in d139_k1}
        assert set(by_date) == {"2021-01-01", "2026-07-01"}   # T9: đúng 2 version
        assert by_date["2026-07-01"].metadata["amended_by_doc"] == "113/2025/QH15"

    def test_superseded_version_links_to_the_new_one(self, bo_luat_chunks):
        versions = {c.metadata["version_id"]: c.metadata for c in bo_luat_chunks}
        old = versions["45-2019-QH14__d139_k1__v2021-01-01"]
        assert old["status"] == "superseded"
        assert old["effective_to"] == "2026-07-01"
        assert old["superseded_by"] == "45-2019-QH14__d139_k1__v2026-07-01"
        assert old["superseded_by"] in versions          # trỏ tới version có thật

    def test_old_and_new_text_actually_differ(self, bo_luat_chunks):
        texts = {c.metadata["effective_from"]: c.text for c in bo_luat_chunks
                 if c.metadata["clause_uid"] == "45-2019-QH14__d139_k1"}
        assert "07 tháng" in texts["2026-07-01"]        # sinh con thứ hai
        assert "07 tháng" not in texts["2021-01-01"]

    def test_clause_uids_are_unique_per_version(self, bo_luat_chunks):
        from collections import Counter

        dup = [v for v, n in Counter(c.metadata["version_id"] for c in bo_luat_chunks).items() if n > 1]
        assert dup == []

    def test_all_220_articles_present(self, bo_luat_chunks):
        dieu = {c.metadata["dieu"] for c in bo_luat_chunks if isinstance(c.metadata["dieu"], int)}
        assert min(dieu) == 1 and max(dieu) == 220
        assert sorted(set(range(1, 221)) - dieu) == []

    def test_consolidated_footnotes_stripped_from_text(self, bo_luat_chunks):
        joined = "\n".join(c.text for c in bo_luat_chunks)
        assert "Khoản này được sửa đổi, bổ sung theo quy định tại" not in joined


class TestClauseStatus:
    def test_status_depends_on_as_of(self):
        from src.ingestion.chunker import clause_status

        assert clause_status("2026-07-01", "9999-12-31", "2026-01-01") == "not_yet_in_force"
        assert clause_status("2026-07-01", "9999-12-31", "2026-09-12") == "in_force"
        assert clause_status("2021-01-01", "2026-07-01", "2026-09-12") == "superseded"
        assert clause_status("2021-01-01", "2026-07-01", "2024-05-01") == "in_force"


class TestAppendixHandling:
    def test_form_templates_are_not_numbered_as_articles(self, manifest):
        from scripts.ingest_documents import build_clause_chunks

        chunks = build_clause_chunks(manifest["374-2025-ND-CP"], CORPUS_DIR, "2026-09-12")
        appendix = [c for c in chunks if c.metadata.get("chunk_strategy") == "appendix"]
        assert appendix, "nghị định có biểu mẫu — phải nhận ra phụ lục"
        # "Điều 1." trong quyết định mẫu không được thành Điều 1 của nghị định
        assert all(c.metadata["dieu"] == "" for c in appendix)
        dieu = [c.metadata["dieu"] for c in chunks if isinstance(c.metadata["dieu"], int)]
        assert max(dieu) == 46          # NĐ 374/2025 có đúng 46 điều


class TestVerifyGate:
    def test_unverified_document_is_not_ingested(self, tmp_path):
        from scripts.ingest_documents import discover_lao_dong

        manifest = tmp_path / "m.yaml"
        manifest.write_text(
            "documents:\n"
            "  - doc_id: \"fake-doc\"\n"
            "    so_hieu: \"00/2026/ND-CP\"\n"
            "    ten: \"Văn bản chưa xác minh\"\n"
            "    loai: \"Nghị định\"\n"
            "    ngay_ban_hanh: \"2026-01-01\"\n"
            "    ngay_hieu_luc: \"2026-01-01\"\n"
            "    verify_status: \"UNVERIFIED\"\n"
            "locked_list_v1:\n"
            "  con_hieu_luc:\n"
            "    - \"fake-doc\"\n",
            encoding="utf-8",
        )
        (tmp_path / "fake-doc.md").write_text(
            "---\ndoc_id: \"fake-doc\"\n---\nĐiều 1. Thử\n\n" + "x" * 600, encoding="utf-8"
        )
        assert discover_lao_dong(manifest, tmp_path, "2026-09-12") == []
