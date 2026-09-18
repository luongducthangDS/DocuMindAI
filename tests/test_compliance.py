"""
Tests for the compliance-check module (src/rag/compliance.py).

Situations are phrased the way a user actually types them, against the real
data/compliance/criteria.json (labour / social-insurance criteria).
"""

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

        condition = {"field": "giờ làm thêm", "unit": "giờ/năm"}
        assert extract_situation_value(
            "Công ty cho làm thêm 250 giờ trong 01 năm có đúng luật không?", condition
        ) == 250.0

    def test_decimal_percent_value(self):
        from src.rag.compliance import extract_situation_value

        condition = {"field": "tiền lương thử việc", "unit": "%"}
        assert extract_situation_value(
            "Công ty trả lương thử việc bằng 82,5% lương chính thức", condition
        ) == 82.5

    def test_day_count_value(self):
        from src.rag.compliance import extract_situation_value

        condition = {"field": "thời gian thử việc", "unit": "ngày"}
        assert extract_situation_value("Thử việc 90 ngày có hợp lệ không?", condition) == 90.0

    def test_full_amount_in_dong_converts_to_millions(self):
        """"4.500.000 đồng" must read as 4.5 triệu, not as the trailing "000"."""
        from src.rag.compliance import extract_situation_value

        condition = {"field": "mức lương", "unit": "triệu đồng/tháng"}
        assert extract_situation_value(
            "Công ty trả lương 4.500.000 đồng/tháng ở vùng I có đúng luật không?", condition
        ) == 4.5

    def test_amount_without_separators_converts_to_millions(self):
        from src.rag.compliance import extract_situation_value

        condition = {"field": "mức lương", "unit": "triệu đồng/tháng"}
        assert extract_situation_value("Lương tối thiểu vùng 1 trả 5310000", condition) == 5.31

    def test_amount_written_in_trieu_is_kept(self):
        from src.rag.compliance import extract_situation_value

        condition = {"field": "mức lương", "unit": "triệu đồng/tháng"}
        assert extract_situation_value("Lương tối thiểu vùng I trả 5,5 triệu", condition) == 5.5

    def test_no_number_falls_back_to_llm_then_none_without_key(self, monkeypatch):
        from src.config import get_settings
        from src.rag.compliance import extract_situation_value

        monkeypatch.setenv("GROQ_API_KEY", "")
        get_settings.cache_clear()
        condition = {"field": "thời gian thử việc", "unit": "ngày"}
        assert extract_situation_value("Thử việc như vậy có hợp lệ không?", condition) is None
        get_settings.cache_clear()


class TestMatchCriteria:
    @pytest.mark.parametrize(
        "situation,expected_id",
        [
            ("Công ty cho nhân viên làm thêm 250 giờ trong 01 năm có đúng luật không?",
             "lam_them_gio_trong_nam"),
            ("Tháng này tôi làm thêm 30 giờ có vượt quy định không?",
             "lam_them_gio_trong_thang"),
            ("Tăng ca 45 giờ trong 1 tháng có hợp lệ không?",
             "lam_them_gio_trong_thang"),
            ("Thử việc 90 ngày cho vị trí kế toán có hợp lệ không?",
             "thoi_gian_thu_viec"),
            ("Công ty trả lương thử việc bằng 80% lương chính thức có được không?",
             "tien_luong_thu_viec"),
            ("Công ty chỉ cho nghỉ phép năm 8 ngày có đúng không?",
             "nghi_hang_nam_dieu_kien_binh_thuong"),
            ("Công ty trả lương 4.500.000 đồng/tháng ở vùng I có đúng luật không?",
             "luong_toi_thieu_thang_vung_1"),
        ],
    )
    def test_matches_by_keyword(self, situation, expected_id):
        from src.rag.compliance import load_criteria, match_criteria

        result = match_criteria(situation, load_criteria())
        assert result is not None, f"không khớp tiêu chí nào: {situation}"
        assert result["id"] == expected_id

    def test_generic_word_alone_is_not_a_match(self):
        """"ngày"/"tháng"/"%" on their own are below _MIN_KEYWORD_SCORE — the
        engine emits a ✅/❌ verdict with a citation, so a weak match must fall
        through to embeddings instead of guessing a criterion."""
        from src.rag.compliance import load_criteria, match_criteria

        with patch("src.rag.compliance._match_by_embedding", return_value=None) as mock_embed:
            result = match_criteria("Hợp đồng này ký được mấy tháng?", load_criteria())
        assert result is None
        mock_embed.assert_called_once()

    def test_no_keyword_match_falls_back_to_embedding_and_returns_none_below_threshold(self):
        from src.rag.compliance import load_criteria, match_criteria

        with patch("src.rag.compliance._match_by_embedding", return_value=None) as mock_embed:
            result = match_criteria("Giá vàng SJC hôm nay bao nhiêu?", load_criteria())
        assert result is None
        mock_embed.assert_called_once()

    def test_empty_criteria_returns_none(self):
        from src.rag.compliance import match_criteria

        assert match_criteria("bất kỳ câu hỏi nào", []) is None


class TestCheckCompliance:
    def test_pass_verdict(self):
        from src.rag.compliance import check_compliance

        result = check_compliance("Tăng ca 30 giờ trong 1 tháng có đúng quy định không?")
        assert result["matched"] is True
        assert result["criterion_id"] == "lam_them_gio_trong_thang"
        assert result["verdict"] == "pass"
        assert result["extracted_value"] == 30.0
        assert result["citation"]["so_hieu"] == "Bộ luật Lao động 45/2019/QH14"

    def test_fail_verdict(self):
        from src.rag.compliance import check_compliance

        result = check_compliance("Công ty cho làm thêm 250 giờ trong 01 năm có đúng luật không?")
        assert result["matched"] is True
        assert result["criterion_id"] == "lam_them_gio_trong_nam"
        assert result["verdict"] == "fail"
        assert result["extracted_value"] == 250.0

    def test_boundary_value_passes(self):
        """Exactly 200 giờ/năm is still within the cap (<=)."""
        from src.rag.compliance import check_compliance

        result = check_compliance("Làm thêm đúng 200 giờ trong 01 năm có vi phạm không?")
        assert result["verdict"] == "pass"
        assert result["extracted_value"] == 200.0

    def test_minimum_wage_below_threshold_fails(self):
        from src.rag.compliance import check_compliance

        result = check_compliance(
            "Công ty trả lương 4.500.000 đồng/tháng ở vùng I có đúng luật không?"
        )
        assert result["criterion_id"] == "luong_toi_thieu_thang_vung_1"
        assert result["verdict"] == "fail"
        assert result["extracted_value"] == 4.5

    def test_no_match_when_unrelated(self):
        from src.rag.compliance import check_compliance

        with patch("src.rag.compliance._match_by_embedding", return_value=None):
            result = check_compliance("Hôm nay trời mưa có nên đi làm không?")
        assert result["matched"] is False
        assert result["verdict"] == "no_match"

    def test_insufficient_info_when_no_number(self, monkeypatch):
        from src.config import get_settings
        from src.rag.compliance import check_compliance

        monkeypatch.setenv("GROQ_API_KEY", "")
        get_settings.cache_clear()
        result = check_compliance("Công ty cho nghỉ phép năm như vậy có đúng không?")
        assert result["matched"] is True
        assert result["verdict"] == "insufficient_info"
        assert result["extracted_value"] is None
        get_settings.cache_clear()

    def test_citation_matches_curated_source(self):
        from src.rag.compliance import check_compliance

        result = check_compliance(
            "Công ty trả lương thử việc bằng 80% lương chính thức có được không?"
        )
        assert result["citation"]["so_hieu"] == "Bộ luật Lao động 45/2019/QH14"
        assert result["citation"]["dieu_khoan"] == "Điều 26"


class TestCriteriaDataIntegrity:
    """The verdict text goes straight to the UI with a ✅/❌ marker, so a
    criterion pointing at a document that is not in the corpus would be an
    authoritative wrong answer (see AUDIT.md R1)."""

    def test_every_criterion_cites_a_document_in_the_corpus(self):
        import re

        from src.rag.compliance import load_criteria

        corpus = Path(__file__).resolve().parents[1] / "data" / "raw" / "lao_dong"
        indexed = {p.stem for p in corpus.glob("*.md")}  # "293-2025-ND-CP"
        for c in load_criteria():
            so_hieu = c["so_hieu"]  # "Nghị định 293/2025/NĐ-CP"
            number = re.search(r"\d+/\d+/[A-ZĐ0-9\-]+", so_hieu)
            assert number, f"{c['id']}: so_hieu '{so_hieu}' không có số hiệu văn bản"
            doc_id = number.group(0).replace("/", "-").replace("Đ", "D")
            assert doc_id in indexed, (
                f"{c['id']} trích dẫn '{so_hieu}' — không có trong data/raw/lao_dong/"
            )

    def test_no_banking_vocabulary_left(self):
        from src.rag.compliance import load_criteria

        blob = str(load_criteria()).lower()
        for term in ("ngân hàng", "nhnn", "tín chấp", "thẻ tín dụng", "dti", "lãi suất"):
            assert term not in blob, f"còn từ vựng ngân hàng trong criteria.json: {term}"

    def test_verdict_templates_render(self):
        from src.rag.compliance import load_criteria

        for c in load_criteria():
            for outcome in ("pass", "fail"):
                rendered = c["verdict_template"][outcome].format(
                    dieu_khoan=c["dieu_khoan"], so_hieu=c["so_hieu"]
                )
                assert c["so_hieu"] in rendered
