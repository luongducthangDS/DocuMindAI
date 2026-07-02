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

        condition = {"field": "diem_ren_luyen", "unit": "diem_100"}
        assert extract_situation_value("Sinh viên có điểm rèn luyện 70 thì xếp loại gì?", condition) == 70.0

    def test_decimal_gpa_value(self):
        from src.rag.compliance import extract_situation_value

        condition = {"field": "gpa_scale4", "unit": "gpa_scale_4"}
        assert extract_situation_value("GPA của em là 2,6 thì có được học bổng không?", condition) == 2.6

    def test_percent_value(self):
        from src.rag.compliance import extract_situation_value

        condition = {"field": "x", "unit": "percent"}
        assert extract_situation_value("Sinh viên nghỉ quá 20% số tiết", condition) == 20.0

    def test_no_number_falls_back_to_llm_then_none_without_key(self, monkeypatch):
        from src.config import get_settings
        from src.rag.compliance import extract_situation_value

        monkeypatch.setenv("GROQ_API_KEY", "")
        get_settings.cache_clear()
        condition = {"field": "gpa_scale4", "unit": "gpa_scale_4"}
        assert extract_situation_value("Em có đủ điều kiện làm khóa luận không?", condition) is None
        get_settings.cache_clear()


class TestMatchCriteria:
    def test_matches_scholarship_by_keyword(self):
        from src.rag.compliance import load_criteria, match_criteria

        criteria = load_criteria()
        result = match_criteria("Điểm trung bình chung học tập của em là 2,6 thì có được học bổng khuyến khích học tập không?", criteria)
        assert result is not None
        assert result["id"] == "hoc_bong_kkht_dtb"

    def test_matches_ren_luyen_by_keyword(self):
        from src.rag.compliance import load_criteria, match_criteria

        criteria = load_criteria()
        result = match_criteria("Điểm rèn luyện 70 có xếp loại khá trở lên không?", criteria)
        assert result is not None
        assert result["id"] == "diem_ren_luyen_kha"

    def test_matches_khoa_luan_by_keyword(self):
        from src.rag.compliance import load_criteria, match_criteria

        criteria = load_criteria()
        result = match_criteria("GPA 2.6 có đủ điều kiện đăng ký khóa luận tốt nghiệp không?", criteria)
        assert result is not None
        assert result["id"] == "khoa_luan_dtb_tich_luy"

    def test_matches_tieng_anh_dau_vao_by_keyword(self):
        from src.rag.compliance import load_criteria, match_criteria

        criteria = load_criteria()
        result = match_criteria("Điểm kiểm tra trình độ tiếng Anh đầu vào 6 có đăng ký được Tiếng Anh cơ bản 1 không?", criteria)
        assert result is not None
        assert result["id"] == "tieng_anh_dau_vao"

    def test_no_keyword_match_falls_back_to_embedding_and_returns_none_below_threshold(self):
        from src.rag.compliance import load_criteria, match_criteria

        criteria = load_criteria()
        with patch("src.rag.compliance._match_by_embedding", return_value=None) as mock_embed:
            result = match_criteria("Học phí kỳ này là bao nhiêu?", criteria)
        assert result is None
        mock_embed.assert_called_once()

    def test_empty_criteria_returns_none(self):
        from src.rag.compliance import match_criteria

        assert match_criteria("bất kỳ câu hỏi nào", []) is None


class TestCheckCompliance:
    def test_pass_verdict(self):
        from src.rag.compliance import check_compliance

        result = check_compliance("GPA 2,6 có đủ điều kiện làm khóa luận tốt nghiệp không?")
        assert result["matched"] is True
        assert result["criterion_id"] == "khoa_luan_dtb_tich_luy"
        assert result["verdict"] == "pass"
        assert result["extracted_value"] == 2.6
        assert "QĐ-828" in result["citation"]["so_hieu"]

    def test_fail_verdict(self):
        from src.rag.compliance import check_compliance

        result = check_compliance("Điểm rèn luyện 40 có xếp loại khá trở lên không?")
        assert result["matched"] is True
        assert result["criterion_id"] == "diem_ren_luyen_kha"
        assert result["verdict"] == "fail"
        assert result["extracted_value"] == 40.0

    def test_boundary_value_passes(self):
        """GPA exactly at the 2.5 threshold should pass (>=)."""
        from src.rag.compliance import check_compliance

        result = check_compliance("Điểm trung bình chung tích lũy 2,5 có đủ điều kiện làm khóa luận không?")
        assert result["verdict"] == "pass"

    def test_no_match_when_unrelated(self):
        from src.rag.compliance import check_compliance

        with patch("src.rag.compliance._match_by_embedding", return_value=None):
            result = check_compliance("Học phí học kỳ này là bao nhiêu?")
        assert result["matched"] is False
        assert result["verdict"] == "no_match"

    def test_insufficient_info_when_no_number(self, monkeypatch):
        from src.config import get_settings
        from src.rag.compliance import check_compliance

        monkeypatch.setenv("GROQ_API_KEY", "")
        get_settings.cache_clear()
        result = check_compliance("Em có đủ điều kiện làm khóa luận tốt nghiệp không?")
        assert result["matched"] is True
        assert result["verdict"] == "insufficient_info"
        assert result["extracted_value"] is None
        get_settings.cache_clear()

    def test_citation_matches_curated_source(self):
        from src.rag.compliance import check_compliance

        result = check_compliance("Điểm rèn luyện 90 có xếp loại khá trở lên không?")
        assert result["citation"]["so_hieu"] == "QĐ-848/ĐHKTKTCN"
        assert result["citation"]["dieu_khoan"] == "Điều 12"
