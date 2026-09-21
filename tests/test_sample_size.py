"""Kiểm tra phần tính cỡ mẫu bằng các giá trị kinh điển tra được trong sách.

Con số ở eval/sample_size.py dùng để biện minh quy mô gold set, nên nó sai thì cả
lập luận "vì sao n câu" sụp theo — đắt hơn nhiều so với một bug thường.
"""

import math

import pytest

from eval.sample_size import n_for_margin, n_mcnemar, n_one_sided, wilson


class TestWilson:
    def test_khoang_tin_cay_khop_gia_tri_da_biet(self):
        # 25/30 — chính là kết quả pilot temporal_filter
        lo, hi = wilson(25, 30)
        assert lo == pytest.approx(0.664, abs=0.002)
        assert hi == pytest.approx(0.927, abs=0.002)

    def test_khong_bao_gio_vuot_khoi_doan_0_1(self):
        """Điểm Wilson hơn Wald: ở p=1 hoặc p=0 vẫn nằm trong [0, 1]."""
        for n in (5, 30, 100):
            lo, hi = wilson(n, n)
            assert 0.0 <= lo <= hi <= 1.0
            lo0, hi0 = wilson(0, n)
            assert 0.0 <= lo0 <= hi0 <= 1.0

    def test_rule_of_three(self):
        """Quan sát 0 lỗi trong n ca => tỉ lệ lỗi thật <= ~3/n."""
        for n in (30, 100, 200):
            lo, _ = wilson(n, n)
            assert (1 - lo) == pytest.approx(3 / n, rel=0.30)

    def test_cang_nhieu_mau_khoang_cang_hep(self):
        widths = [wilson(round(0.9 * n), n) for n in (30, 100, 300)]
        widths = [hi - lo for lo, hi in widths]
        assert widths == sorted(widths, reverse=True)

    def test_n_bang_khong_tra_ve_toan_doan(self):
        assert wilson(0, 0) == (0.0, 1.0)


class TestCoMau:
    def test_truong_hop_xau_nhat_p_bang_nua(self):
        """Giá trị sách giáo khoa: p=0.5, ±5pp, 95% => 385."""
        assert n_for_margin(0.5, 0.05) == 385

    def test_p_lech_khoi_nua_thi_can_it_mau_hon(self):
        assert n_for_margin(0.9, 0.05) < n_for_margin(0.5, 0.05)

    def test_muon_chinh_xac_gap_doi_thi_ton_gap_bon(self):
        """Sai số tỉ lệ nghịch với căn bậc hai của n."""
        assert n_for_margin(0.8, 0.05) == pytest.approx(4 * n_for_margin(0.8, 0.10), rel=0.02)


class TestMcNemar:
    def test_hieu_ung_cang_nho_cang_can_nhieu_mau(self):
        ns = [n_mcnemar(0.43, e) for e in (0.30, 0.20, 0.15, 0.10, 0.05)]
        assert all(n is not None for n in ns)
        assert ns == sorted(ns)

    def test_bat_dong_loan_xa_thi_can_nhieu_mau_hon(self):
        """Ngược trực giác, nên chốt lại bằng test.

        Với chênh lệch cố định 10pp: discordant 15% cần 101 câu, discordant 45% cần 348.
        Ít bất đồng mà vẫn chênh 10pp => bất đồng gần như dồn hết về một phía, tín hiệu
        rõ. Bất đồng nhiều => hai hệ sai khác nhau loạn xạ, phải gom nhiều câu mới thấy.
        """
        assert n_mcnemar(0.15, 0.10) < n_mcnemar(0.45, 0.10)

    def test_hieu_ung_vuot_qua_ti_le_bat_dong_la_vo_nghia(self):
        # Không thể hơn nhau 40pp khi chỉ 10% số câu cho kết quả khác nhau.
        assert n_mcnemar(0.10, 0.40) is None


class TestNguongCamKet:
    def test_nguong_cang_sat_hieu_nang_that_cang_dat(self):
        ns = [n_one_sided(0.833, p0) for p0 in (0.60, 0.70, 0.75, 0.80)]
        assert all(n is not None for n in ns)
        assert ns == sorted(ns)
        # Bài học chính: cam kết "trên 80%" khi hệ đúng 83.3% là bất khả thi trên thực tế.
        assert ns[-1] > 500

    def test_khong_the_chung_minh_nguong_cao_hon_hieu_nang_that(self):
        assert n_one_sided(0.70, 0.80) is None
        assert n_one_sided(0.80, 0.80) is None


def test_bang_so_dung_de_bien_minh_gold_set():
    """Chốt lại đúng những con số đang dùng làm cơ sở chọn 55 câu core + 30 ca guardrail."""
    assert n_for_margin(0.833, 0.10) == 54          # accuracy ±10pp
    assert n_one_sided(0.833, 0.70) == 65           # chứng minh "đúng > 70%"
    lo, _ = wilson(30, 30)
    assert (1 - lo) == pytest.approx(0.114, abs=0.005)  # guardrail: lọt <= 11.4%
