"""Cỡ mẫu cho gold set: suy từ QUYẾT ĐỊNH cần phục vụ, không chọn số tròn rồi biện minh ngược.

Câu hỏi "tại sao n câu?" chỉ trả lời được khi đã nói rõ bộ eval dùng để quyết định gì.
Ba quyết định khác nhau cho ba cỡ mẫu khác nhau, chênh nhau hàng chục lần:

  1. Ước lượng độ chính xác        -> cần bao nhiêu để CI đủ hẹp
  2. So sánh hai cấu hình          -> cần bao nhiêu để phát hiện chênh lệch quan tâm
  3. Chứng minh vượt ngưỡng cam kết -> cần bao nhiêu để bác bỏ H0
  4. Đo guardrail (kỳ vọng chặn ~100%) -> rule of three

Tham số p (tỉ lệ đúng) và tỉ lệ discordant KHÔNG bịa: đọc thẳng từ kết quả eval đã chạy
(reports/temporal_eval.json) — đó là vai trò của pilot study. Thiếu file thì rơi về giá trị
mặc định và có cảnh báo.

Chạy:
    python eval/sample_size.py
    python eval/sample_size.py --pilot reports/temporal_eval.json --arm temporal_filter
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy import stats

Z95 = stats.norm.ppf(0.975)

# Dùng khi không có pilot. Cố tình bi quan hơn số đo thật để không hạ thấp cỡ mẫu.
_FALLBACK_P = 0.80
_FALLBACK_DISCORDANT = 0.30


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Khoảng tin cậy Wilson.

    Không dùng Wald: ở p gần 1 — đúng vùng hệ RAG tốt hoạt động — Wald cho cận trên
    vượt quá 1 và cận dưới lạc quan giả tạo. Wilson không có bệnh đó.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


def n_for_margin(p: float, margin: float, z: float = Z95) -> int:
    """Cỡ mẫu để nửa bề rộng CI <= margin."""
    return math.ceil(z * z * p * (1 - p) / (margin * margin))


def n_mcnemar(p_discordant: float, effect: float, power: float = 0.80, alpha: float = 0.05):
    """Cỡ mẫu để phát hiện chênh lệch giữa 2 cấu hình chạy trên CÙNG bộ câu hỏi.

    Chỉ những câu hai cấu hình cho kết quả khác nhau (discordant) mới mang thông tin;
    câu nào cả hai cùng đúng hoặc cùng sai đều bị loại khỏi phép kiểm.

    Lưu ý ngược trực giác: với chênh lệch cho trước, tỉ lệ discordant THẤP lại cần ÍT
    câu hơn. Vì khi ít bất đồng mà chênh lệch vẫn giữ nguyên thì gần như mọi bất đồng
    phải nghiêng về một phía (psi xa 0.5), tức tín hiệu rất rõ. Discordant cao nghĩa là
    hai hệ bất đồng loạn xạ cả hai chiều, phải gom nhiều câu mới thấy được xu hướng.
    """
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    psi = 0.5 + effect / (2 * p_discordant)
    if not 0 < psi < 1:
        return None  # hiệu ứng lớn hơn cả tỉ lệ discordant: giả định tự mâu thuẫn
    n_disc = ((za / 2 + zb * math.sqrt(psi * (1 - psi))) / (psi - 0.5)) ** 2
    return math.ceil(n_disc / p_discordant)


def n_one_sided(p_true: float, p0: float, power: float = 0.80, alpha: float = 0.05):
    """Cỡ mẫu để bác bỏ H0: p <= p0, khi tỉ lệ thật là p_true."""
    if p_true <= p0:
        return None
    za, zb = stats.norm.ppf(1 - alpha), stats.norm.ppf(power)
    num = za * math.sqrt(p0 * (1 - p0)) + zb * math.sqrt(p_true * (1 - p_true))
    return math.ceil((num / (p_true - p0)) ** 2)


def load_pilot(path: Path, arm: str) -> tuple[float, float, int, str]:
    """Đọc p̂ và tỉ lệ discordant từ kết quả eval đã chạy."""
    if not path.exists():
        return _FALLBACK_P, _FALLBACK_DISCORDANT, 0, f"KHÔNG có pilot ({path}) — dùng giá trị mặc định"

    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    arms = list(rows[0]["arms"].keys())
    if arm not in arms:
        arm = arms[-1]

    def ok(row, a):
        return bool(row["arms"][a]["score"]["correct"])

    n = len(rows)
    p_hat = sum(ok(r, arm) for r in rows) / n

    # So với arm tốt THỨ NHÌ, không phải arm tệ nhất: cỡ mẫu cần cho lần cải tiến kế
    # tiếp mới là thứ đáng ước lượng, mà lần đó luôn là so với ứng viên gần nhất.
    # Arm tệ nhất bất đồng nhiều hơn (0.53 so với 0.43 trên pilot này) nên sẽ thổi
    # phồng cỡ mẫu — thận trọng quá mức cũng là ước lượng sai.
    others = sorted(
        (a for a in arms if a != arm),
        key=lambda a: sum(ok(r, a) for r in rows),
        reverse=True,
    )
    p_disc = _FALLBACK_DISCORDANT
    runner_up = ""
    if others:
        runner_up = others[0]
        p_disc = sum(1 for r in rows if ok(r, arm) != ok(r, runner_up)) / n

    note = f"pilot {path} — arm '{arm}' vs '{runner_up}', n={n}" if runner_up else f"pilot {path} — arm '{arm}', n={n}"
    return p_hat, p_disc, n, note


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", default="reports/temporal_eval.json")
    ap.add_argument("--arm", default="temporal_filter", help="Arm dùng làm hệ tham chiếu")
    args = ap.parse_args()

    p_hat, p_disc, n_pilot, source = load_pilot(Path(args.pilot), args.arm)

    print("=" * 74)
    print("CƠ SỞ CHỌN CỠ MẪU CHO GOLD SET")
    print("=" * 74)
    print(f"Nguồn tham số: {source}")
    print(f"  p̂ (tỉ lệ đúng)        = {p_hat:.3f}")
    print(f"  tỉ lệ discordant      = {p_disc:.2f}")
    if n_pilot:
        lo, hi = wilson(round(p_hat * n_pilot), n_pilot)
        print(f"  CI95 của pilot        = [{lo:.3f}, {hi:.3f}]  (±{(hi - lo) / 2 * 100:.1f}pp)")

    print()
    print("QUYẾT ĐỊNH 1 — báo cáo độ chính xác với sai số mục tiêu")
    for margin in (0.15, 0.10, 0.08, 0.06, 0.05):
        print(f"   ±{margin * 100:>4.0f}pp  ->  n = {n_for_margin(p_hat, margin):>5}")

    print()
    print("QUYẾT ĐỊNH 2 — so sánh 2 cấu hình (McNemar, power 80%, alpha 5%)")
    for effect in (0.05, 0.10, 0.15, 0.20, 0.30):
        n = n_mcnemar(p_disc, effect)
        print(f"   chênh {effect * 100:>4.0f}pp  ->  " + (f"n = {n:>5}" if n else "hiệu ứng vượt quá tỉ lệ discordant"))

    print()
    print("QUYẾT ĐỊNH 3 — chứng minh vượt ngưỡng cam kết (một phía)")
    for p0 in (0.60, 0.70, 0.75, 0.80):
        n = n_one_sided(p_hat, p0)
        print(f"   p > {p0:.2f}  ->  " + (f"n = {n:>5}" if n else "không thể (p̂ <= ngưỡng)"))

    print()
    print("QUYẾT ĐỊNH 4 — guardrail: chặn hết n ca thì tỉ lệ lọt tối đa là bao nhiêu")
    print("   (rule of three: cận trên 95% ≈ 3/n khi quan sát 0 lỗi)")
    for n in (10, 20, 30, 50):
        lo, _ = wilson(n, n)
        print(f"   n = {n:>3}  ->  tỉ lệ lọt <= {(1 - lo) * 100:>5.1f}%")

    print()
    print("-" * 74)
    print("Giới hạn phải nói kèm mọi con số trên: gold set là bộ phân tầng viết tay,")
    print("cố ý nhồi ca khó, KHÔNG phải mẫu ngẫu nhiên từ câu hỏi người dùng thật.")
    print("Vậy đây là 'độ chính xác trên bộ kiểm thử', không phải độ chính xác khi")
    print("triển khai. CI chỉ mô tả sai số lấy mẫu, không bao gồm sai lệch lựa chọn.")


if __name__ == "__main__":
    main()
