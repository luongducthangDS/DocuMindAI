"""F1 + F2 (USER_FEEDBACK.md) — đơn vị thời gian và lương thử việc cho bằng hai số tiền.

F1: tiêu chí thử việc tính bằng NGÀY (BLLĐ 45/2019 Điều 25), người dùng viết
"3 tháng" / "7 tuần". So thẳng 3 với 60 cho ✅ một vụ thử việc 90 ngày.
F2: "lương thử việc 9 triệu, chính thức 10 triệu" bị đọc thành 9 rồi so với 85%.

Ngưỡng tra từ corpus (data/raw/lao_dong/45-2019-QH14.md, Điều 25): khoản 2 —
cao đẳng trở lên ≤ 60 ngày; khoản 3 — trung cấp/công nhân kỹ thuật ≤ 30 ngày.
Tiêu chí hiện có chỉ là khoản 2 (60 ngày).
"""

from __future__ import annotations

import pytest

from src.rag import compliance

pytestmark = pytest.mark.usefixtures("real_criteria", "no_network")


@pytest.fixture(autouse=True)
def _no_fallbacks(monkeypatch):
    def boom(*_a, **_kw):
        raise AssertionError("rơi xuống fallback LLM/embedding")

    monkeypatch.setattr(compliance, "_extract_value_via_llm", boom)
    monkeypatch.setattr(compliance, "_match_by_embedding", boom)


@pytest.mark.parametrize("situation,criterion_id,value,verdict", [
    # 1. 3 tháng = 90 ngày > 60 → ❌
    ("Thử việc vị trí chuyên môn 3 tháng có đúng luật không?", "thoi_gian_thu_viec", 90.0, "fail"),
    # 2. 7 tuần = 49 ngày. Cao đẳng thuộc Điều 25 KHOẢN 2 (≤ 60 ngày), không phải
    #    khoản 3 (≤ 30 ngày, trung cấp) → 49 ≤ 60 → ✅.
    ("Thử việc 7 tuần cho công việc cao đẳng", "thoi_gian_thu_viec", 49.0, "pass"),
    # 3. 8,5 / 10 = 85% ≥ 85% → ✅ (đúng biên)
    ("Lương thử việc 8.5 triệu, lương chính thức 10 triệu", "tien_luong_thu_viec", 85.0, "pass"),
    # 4. 8 / 10 = 80% < 85% → ❌
    ("Lương thử việc 8 triệu trên mức chính thức 10 triệu", "tien_luong_thu_viec", 80.0, "fail"),
])
def test_required_cases(situation, criterion_id, value, verdict):
    result = compliance.check_compliance(situation)
    assert result["criterion_id"] == criterion_id
    assert result["extracted_value"] == value
    assert result["verdict"] == verdict


# ── F1 ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,unit,days", [
    (3, "tháng", 90), (7, "tuần", 49), (1, "năm", 365), (45, "ngày", 45), (1.5, "tháng", 45),
])
def test_normalize_time_unit(value, unit, days):
    assert compliance.normalize_time_unit(value, unit) == days


def test_months_straddling_the_threshold_are_not_judged():
    """"2 tháng" dài 59–62 ngày tuỳ tháng bắt đầu; ngưỡng là 60 NGÀY → không phán."""
    result = compliance.check_compliance("Công ty cho thử việc 2 tháng có đúng luật không?")
    assert result["criterion_id"] == "thoi_gian_thu_viec"
    assert result["verdict"] == "insufficient_info"
    assert "59" in result["explanation"] and "62" in result["explanation"]


def test_month_plus_days_are_added():
    result = compliance.check_compliance("Thử việc 2 tháng 15 ngày có đúng luật không?")
    assert result["extracted_value"] == 75.0
    assert result["verdict"] == "fail"


def test_days_written_directly_are_unchanged():
    result = compliance.check_compliance("Thử việc 45 ngày có hợp lệ không?")
    assert result["extracted_value"] == 45.0
    assert result["verdict"] == "pass"


# ── F2 ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("situation,percent", [
    ("Lương thử việc 9 triệu, lương chính thức 10 triệu có đúng luật không?", 90.0),
    ("Lương chính thức 10 triệu, lương thử việc 9 triệu có được không?", 90.0),
    ("10 triệu lương chính thức, thử việc trả 8 triệu có đúng không?", 80.0),
    ("Lương thử việc 9tr / 10tr có đúng luật không?", 90.0),
    ("Lương thử việc 8.500.000 đồng, lương chính thức 10.000.000 đồng", 85.0),
])
def test_wage_ratio_from_two_amounts(situation, percent):
    result = compliance.check_compliance(situation)
    assert result["criterion_id"] == "tien_luong_thu_viec"
    assert result["extracted_value"] == percent
    assert result["verdict"] == ("pass" if percent >= 85 else "fail")


def test_official_contract_mention_does_not_pull_in_the_wage_criterion():
    """"hợp đồng chính thức" không phải lương chính thức — vẫn là tiêu chí thời gian."""
    matched = compliance.match_criteria(
        "Thử việc 3 tháng rồi mới ký hợp đồng chính thức có đúng không?", compliance.load_criteria()
    )
    assert matched["id"] == "thoi_gian_thu_viec"


def test_single_amount_is_not_compared_with_a_percentage():
    """"9 triệu" so với ngưỡng 85% là so tiền với phần trăm — không phán."""
    result = compliance.check_compliance("Lương thử việc 9 triệu có đúng luật không?")
    assert result["criterion_id"] == "tien_luong_thu_viec"
    assert result["verdict"] == "insufficient_info"
    assert result["extracted_value"] is None
