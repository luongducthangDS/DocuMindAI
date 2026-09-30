"""Q03 (P0) — parser số tiếng Việt của engine compliance.

Engine này phát ✅/❌ kèm trích dẫn điều luật, nên đọc sai một con số là một câu
trả lời sai có thẩm quyền. Ca gốc (đo 2026-09-27, trước khi sửa):
  "làm thêm 1.200 giờ trong năm"                    → 1.2  → ✅ pass
  "làm thêm vào ngày 30/4, cả năm 320 giờ trong năm" → 30   → ✅ pass
  "lương thử việc bằng 0,85 lương chính thức"        → 0.85 → ❌ fail (so với 85%)
Dung sai = 0: mọi so sánh số là `==`.
"""

from __future__ import annotations

import pytest

from src.rag import compliance

pytestmark = pytest.mark.usefixtures("real_criteria", "no_network")


@pytest.fixture(autouse=True)
def _no_fallbacks(monkeypatch):
    """Regex phải tự trích được — không cho rơi xuống LLM/embedding che lỗi parser."""
    def boom(*_a, **_kw):
        raise AssertionError("parser rơi xuống fallback LLM/embedding")

    monkeypatch.setattr(compliance, "_extract_value_via_llm", boom)
    monkeypatch.setattr(compliance, "_match_by_embedding", boom)


CASES = [
    # (câu, criterion_id, giá trị trích, verdict)
    ("Công ty cho nhân viên làm thêm 1.200 giờ trong năm có đúng luật không?",
     "lam_them_gio_trong_nam", 1200.0, "fail"),
    ("Lương thử việc bằng 0,85 lương chính thức có đúng luật không?",
     "tien_luong_thu_viec", 0.85, "pass"),
    ("Công ty cho làm thêm vào ngày 30/4, cả năm 320 giờ trong năm có đúng luật không?",
     "lam_them_gio_trong_nam", 320.0, "fail"),
]


@pytest.mark.parametrize("situation,criterion_id,value,verdict", CASES)
def test_match_criteria_picks_the_right_criterion(situation, criterion_id, value, verdict):
    matched = compliance.match_criteria(situation, compliance.load_criteria())
    assert matched is not None and matched["id"] == criterion_id


@pytest.mark.parametrize("situation,criterion_id,value,verdict", CASES)
def test_extract_situation_value_exact(situation, criterion_id, value, verdict):
    criterion = next(c for c in compliance.load_criteria() if c["id"] == criterion_id)
    assert compliance.extract_situation_value(situation, criterion["condition"]) == value


@pytest.mark.parametrize("situation,criterion_id,value,verdict", CASES)
def test_check_compliance_verdict(situation, criterion_id, value, verdict):
    result = compliance.check_compliance(situation)
    assert result["criterion_id"] == criterion_id
    assert result["verdict"] == verdict


def test_ratio_is_reported_in_the_criterion_unit():
    """0,85 lương = 85% — giá trị trả về phải cùng đơn vị với ngưỡng (85%)."""
    result = compliance.check_compliance("Lương thử việc bằng 0,85 lương chính thức có đúng không?")
    assert result["extracted_value"] == 85.0


def test_explicit_percent_is_not_rescaled():
    """"0,5%" là nửa phần trăm, không phải tỷ lệ 0,5 — không được nhân 100."""
    result = compliance.check_compliance("Lương thử việc bằng 0,5% lương chính thức có đúng không?")
    assert result["extracted_value"] == 0.5
    assert result["verdict"] == "fail"


@pytest.mark.parametrize("raw,expected", [
    ("1.200", 1200.0),       # chấm tách nghìn
    ("12.000", 12000.0),
    ("1.200.000", 1200000.0),
    ("0,85", 0.85),          # phẩy thập phân
    ("5,31", 5.31),
    ("5.31", 5.31),          # nhóm 2 chữ số sau chấm vẫn là thập phân
    ("200", 200.0),
])
def test_parse_number_vietnamese_convention(raw, expected):
    assert compliance._parse_number(raw) == expected


@pytest.mark.parametrize("situation", [
    "Ngày 01/01/2026 công ty cho làm thêm 250 giờ trong năm có đúng không?",
    "Từ 15/3 tới nay công ty cho làm thêm 250 giờ trong năm có đúng không?",
])
def test_dates_are_never_read_as_the_quantity(situation):
    criterion = next(c for c in compliance.load_criteria() if c["id"] == "lam_them_gio_trong_nam")
    assert compliance.extract_situation_value(situation, criterion["condition"]) == 250.0
