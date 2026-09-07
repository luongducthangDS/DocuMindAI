"""
Tests for the compliance-check module (src/rag/compliance.py).
"""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _criteria_fixture(tmp_path, monkeypatch):
    """conftest's patch_settings points DATA_DIR at tmp_path/data — copy the
    real, hand-curated criteria.json there so tests exercise the actual data,
    and clear compliance.load_criteria's lru_cache between tests."""
    real_path = Path(__file__).resolve().parents[1] / "data" / "compliance" / "criteria.json"
    dest_dir = tmp_path / "data" / "compliance"
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(real_path, dest_dir / "criteria.json")

    from src.rag import compliance
    compliance.load_criteria.cache_clear()
    yield
    compliance.load_criteria.cache_clear()


class TestEvaluateCondition:
    def test_operators(self):
        from src.rag.compliance import evaluate_condition

        assert evaluate_condition({"operator": ">=", "value": 2.5}, 2.5) is True
        assert evaluate_condition({"operator": ">=", "value": 2.5}, 2.4) is False
        assert evaluate_condition({"operator": "<=", "value": 20}, 20) is True
        assert evaluate_condition({"operator": "<", "value": 20}, 20) is False
        assert evaluate_condition({"operator": ">", "value": 5}, 5.1) is True
        assert evaluate_condition({"operator": "==", "value": 65}, 65) is True

    def test_unsupported_operator_raises(self):
        from src.rag.compliance import evaluate_condition

        with pytest.raises(ValueError):
            evaluate_condition({"operator": "!=", "value": 1}, 2)


class TestExtractSituationValue:
    def test_labeled_number_near_keyword(self):
        from src.rag.compliance import extract_situation_value

        condition = {"field": "thu nhập", "unit": "triệu VNĐ/tháng"}
        assert extract_situation_value("Khách hàng có thu nhập 15 triệu đồng/tháng có được vay không?", condition) == 15.0

    def test_decimal_interest_rate_value(self):
        from src.rag.compliance import extract_situation_value

        condition = {"field": "lãi suất", "unit": "%/năm"}
        assert extract_situation_value("Ngân hàng áp dụng lãi suất 4,5% kỳ hạn 3 tháng", condition) == 4.5

    def test_percent_value(self):
        from src.rag.compliance import extract_situation_value

        condition = {"field": "tỷ lệ nợ", "unit": "percent"}
        assert extract_situation_value("Khách hàng có tỷ lệ nợ 55% thì có đủ chuẩn không?", condition) == 55.0

    def test_no_number_falls_back_to_llm_then_none_without_key(self, monkeypatch):
        from src.config import get_settings
        from src.rag.compliance import extract_situation_value

        monkeypatch.setenv("GROQ_API_KEY", "")
        get_settings.cache_clear()
        condition = {"field": "thu nhập", "unit": "triệu VNĐ/tháng"}
        assert extract_situation_value("Khách hàng muốn vay tín chấp thì có được duyệt không?", condition) is None
        get_settings.cache_clear()


class TestMatchCriteria:
    def test_matches_vay_tin_chap_by_keyword(self):
        from src.rag.compliance import load_criteria, match_criteria

        criteria = load_criteria()
        result = match_criteria("Khách hàng có thu nhập 8 triệu muốn vay tín chấp tiêu dùng", criteria)
        assert result is not None
        assert result["id"] == "vay_tin_chap_thu_nhap"

    def test_matches_dti_by_keyword(self):
        from src.rag.compliance import load_criteria, match_criteria

        criteria = load_criteria()
        result = match_criteria("Tỷ lệ nợ trên thu nhập DTI của khách hàng là 50% có vay được không?", criteria)
        assert result is not None
        assert result["id"] == "ty_le_dti_cho_vay"

    def test_matches_the_tin_dung_by_keyword(self):
        from src.rag.compliance import load_criteria, match_criteria

        criteria = load_criteria()
        result = match_criteria("Hạn mức thẻ tín dụng tín chấp 80 triệu có được phê duyệt không?", criteria)
        assert result is not None
        assert result["id"] == "han_muc_the_tin_dung_tin_chap"

    def test_matches_lai_suat_khong_ky_han_by_keyword(self):
        from src.rag.compliance import load_criteria, match_criteria

        criteria = load_criteria()
        result = match_criteria("Lãi suất tiền gửi không kỳ hạn là 0.8% có hợp lệ không?", criteria)
        assert result is not None
        assert result["id"] == "lai_suat_tien_gui_khong_ky_han"

    def test_no_keyword_match_falls_back_to_embedding_and_returns_none_below_threshold(self):
        from src.rag.compliance import load_criteria, match_criteria

        criteria = load_criteria()
        with patch("src.rag.compliance._match_by_embedding", return_value=None) as mock_embed:
            result = match_criteria("Giờ mở cửa của chi nhánh ngân hàng là mấy giờ?", criteria)
        assert result is None
        mock_embed.assert_called_once()

    def test_empty_criteria_returns_none(self):
        from src.rag.compliance import match_criteria

        assert match_criteria("bất kỳ câu hỏi nào", []) is None


class TestCheckCompliance:
    def test_pass_verdict(self):
        from src.rag.compliance import check_compliance

        result = check_compliance("Thu nhập 8 triệu có đủ điều kiện vay tín chấp không?")
        assert result["matched"] is True
        assert result["criterion_id"] == "vay_tin_chap_thu_nhap"
        assert result["verdict"] == "pass"
        assert result["extracted_value"] == 8.0
        assert "CV-05/2023/NH" in result["citation"]["so_hieu"]

    def test_fail_verdict(self):
        from src.rag.compliance import check_compliance

        result = check_compliance("Thu nhập 3.5 triệu có đủ điều kiện vay tín chấp không?")
        assert result["matched"] is True
        assert result["criterion_id"] == "vay_tin_chap_thu_nhap"
        assert result["verdict"] == "fail"
        assert result["extracted_value"] == 3.5

    def test_boundary_value_passes(self):
        """Income exactly at the 5.0 threshold should pass (>=)."""
        from src.rag.compliance import check_compliance

        result = check_compliance("Khách hàng có thu nhập 5 triệu đồng/tháng có được vay tín chấp không?")
        assert result["verdict"] == "pass"
        assert result["extracted_value"] == 5.0

    def test_no_match_when_unrelated(self):
        from src.rag.compliance import check_compliance

        with patch("src.rag.compliance._match_by_embedding", return_value=None):
            result = check_compliance("Thời gian làm việc phòng giao dịch là khi nào?")
        assert result["matched"] is False
        assert result["verdict"] == "no_match"

    def test_insufficient_info_when_no_number(self, monkeypatch):
        from src.config import get_settings
        from src.rag.compliance import check_compliance

        monkeypatch.setenv("GROQ_API_KEY", "")
        get_settings.cache_clear()
        result = check_compliance("Khách hàng muốn vay tín chấp thì có đủ điều kiện không?")
        assert result["matched"] is True
        assert result["verdict"] == "insufficient_info"
        assert result["extracted_value"] is None
        get_settings.cache_clear()

    def test_citation_matches_curated_source(self):
        from src.rag.compliance import check_compliance

        result = check_compliance("Lãi suất tiền gửi không kỳ hạn là 0.2% có đúng quy định không?")
        assert result["citation"]["so_hieu"] == "Quyết định 1124/QĐ-NHNN"
        assert result["citation"]["dieu_khoan"] == "Điều 1"
