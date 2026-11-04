"""
eval/calibrate_judge.py — đo các bộ chấm tự động so với nhãn tay, trước khi tin con số của chúng.

  python eval/calibrate_judge.py                                   # nhãn mới nhất, 2 lượt judge
  python eval/calibrate_judge.py --labels data/eval/judge_calibration/labels_2026-10-18.json --repeat 3

So ba bộ chấm với người: judge độ đúng (judge_correctness, đối chiếu đáp án gold), chấm chuỗi
(score_answer — nhãn lưu sẵn trong file), faithfulness (judge_faithfulness — nhãn lưu sẵn).
"Sai mà tự tin" = người chấm "sai" và câu trả lời không phải lời từ chối: loại lỗi nguy hiểm nhất.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.temporal_eval import is_refusal, judge_correctness  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=max((ROOT / "data/eval/judge_calibration").glob("labels_*.json")))
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--output", type=Path, default=ROOT / "reports/judge_calibration.json")
    args = ap.parse_args()

    rows = json.loads(args.labels.read_text(encoding="utf-8"))["rows"]
    runs = []
    for k in range(args.repeat):
        out = {}
        for r in rows:
            j = judge_correctness(r["question"], r["ground_truth"], r["answer"], r["as_of"])
            out[r["id"]] = j["verdict"] if j else None
            print(f"[lượt {k + 1}] {r['id']:9} người={r['human']['verdict']:8} judge={out[r['id']]}", flush=True)
        runs.append(out)

    n = len(rows)
    human = {r["id"]: r["human"]["verdict"] for r in rows}
    ok = lambda v: v == "dung"  # noqa: E731
    summary = {"n": n, "labels": str(args.labels.relative_to(ROOT)), "human": dict(Counter(human.values()))}
    for k, out in enumerate(runs, 1):
        judged = [q for q in human if out[q]]
        summary[f"judge_run{k}"] = {
            "n_judged": len(judged),
            "agree_3_muc": sum(out[q] == human[q] for q in judged),
            "agree_dung_vs_khong": sum(ok(out[q]) == ok(human[q]) for q in judged),
            "confusion_nguoi_x_judge": dict(Counter(f"{human[q]}→{out[q]}" for q in judged)),
        }
    if len(runs) > 1:
        summary["judge_on_dinh_giua_cac_luot"] = sum(len({o[q] for o in runs}) == 1 for q in human)
    summary["string_agree_dung_vs_khong"] = sum(r["auto"]["string_correct"] == ok(r["human"]["verdict"]) for r in rows)
    summary["faithfulness_1.0_khi_nguoi_cham_sai"] = sorted(
        r["id"] for r in rows if r["human"]["verdict"] == "sai" and (r["auto"]["judge_faithfulness"] or 0) >= 1.0)
    cw = [r["id"] for r in rows if r["human"]["verdict"] == "sai" and not is_refusal(r["answer"])]
    summary["sai_ma_tu_tin"] = {
        "nguoi": cw,
        **{f"judge_bat_duoc_run{k}": sorted(q for q in cw if out[q] == "sai") for k, out in enumerate(runs, 1)},
        "chuoi_bat_duoc": sorted(q for q in cw if not next(r for r in rows if r["id"] == q)["auto"]["string_correct"]),
    }
    args.output.write_text(json.dumps({"summary": summary, "runs": runs}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
