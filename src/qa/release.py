"""Phân tích defect (so hai report temporal_eval theo từng câu) + release scorecard GO/NO-GO.

Defect = câu baseline đúng, bản mới sai. Phân loại theo tầng hỏng để biết sửa ở đâu:
  infra        — câu đo thiếu bước (embedding/rerank/LLM lỗi): chạy lại, chưa phải defect thật
  oos_leak     — câu phải từ chối mà hệ thống trả lời: hallucination ngoài phạm vi
  stale_law    — câu trả lời chứa con số/văn bản đã hết hiệu lực (expect_absent)
  wrong_verdict— kết luận Có/Không ngược
  generation   — đoạn gold CÓ trong ngữ cảnh mà câu trả lời vẫn sai
  retrieval    — đoạn gold rơi khỏi top-k

  python -m src.qa.release reports/eval_e2e_pr.json --baseline reports/eval_e2e.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SEVERITY = {"oos_leak": "critical", "stale_law": "critical", "wrong_verdict": "critical",
            "generation": "major", "retrieval": "minor", "infra": "info"}
_RANK = {"critical": 0, "major": 1, "minor": 2, "info": 3}
TOP_K = 8
# Hai lượt cùng cấu hình FPT lệch 2–3 câu/208 (≈0.015) — ngưỡng chặt hơn thế là gate nhiễu.
MAX_ACCURACY_DROP = 0.02


def _ok(arm: dict) -> tuple[bool | None, bool]:
    """(câu trả lời đúng? — None nếu report retrieval-only, gold trong top-k?)."""
    ctx = arm.get("context", {})
    score = arm.get("score")
    return (score.get("correct") if score else None), (ctx.get("gold_rank") or 99) <= TOP_K or bool(ctx.get("gold"))


def _classify(row: dict, arm: dict) -> str:
    if row.get("degraded") or arm.get("used_llm") == "extractive_fallback":
        return "infra"
    score = arm.get("score") or {}
    if row.get("expected_behavior") != "answer" and score.get("correct") is False:
        return "oos_leak"
    if score.get("forbidden_present"):
        return "stale_law"
    if score.get("wrong_verdict"):
        return "wrong_verdict"
    if score and arm.get("context", {}).get("gold"):
        return "generation"
    return "retrieval"


def analyze_defects(new: dict, base: dict, arm: str = "pre_filter", gold: str | None = None) -> dict:
    gold = gold or new["meta"]["gold_set"]
    base_rows = {r["id"]: r for r in base["rows"]}
    defects, fixed = [], []
    for r in new["rows"]:
        b = base_rows.get(r["id"])
        if not b or arm not in r["arms"] or arm not in b["arms"]:
            continue
        (ans_n, ctx_n), (ans_b, ctx_b) = _ok(r["arms"][arm]), _ok(b["arms"][arm])
        # Report có câu trả lời ⇒ chấm theo câu trả lời; retrieval-only ⇒ chấm theo ngữ cảnh.
        now, was = (ans_n, ans_b) if ans_n is not None and ans_b is not None else (ctx_n, ctx_b)
        if was and not now:
            a = r["arms"][arm]
            cat = _classify(r, a)
            defects.append({
                "id": r["id"], "question": r["question"], "as_of_date": r.get("as_of_date"),
                "category": cat, "severity": SEVERITY[cat],
                "evidence": {
                    "gold_rank": [b["arms"][arm]["context"].get("gold_rank"), a["context"].get("gold_rank")],
                    "score": a.get("score"), "degraded": r.get("degraded") or [],
                    "model": a.get("model"), "answer": (a.get("answer") or "")[:600],
                },
                "repro": f"python eval/temporal_eval.py --gold {gold} --arms {arm} --ids {r['id']}",
            })
        elif now and not was:
            fixed.append(r["id"])
    defects.sort(key=lambda d: (_RANK[d["severity"]], d["id"]))
    by_cat: dict[str, int] = {}
    for d in defects:
        by_cat[d["category"]] = by_cat.get(d["category"], 0) + 1
    return {"arm": arm, "n_compared": sum(1 for r in new["rows"] if r["id"] in base_rows),
            "defects": defects, "fixed": fixed, "by_category": by_cat}


def release_scorecard(new: dict, base: dict, arm: str = "pre_filter", tests_passed: bool | None = None) -> dict:
    """GO chỉ khi: test xanh (nếu biết), gate metric không tụt quá ngưỡng, không câu đo thiếu bước,
    không defect critical. Defect major/minor = cảnh báo, người duyệt quyết."""
    from eval.ci_gate import gate  # dùng đúng ngưỡng của CI, không chép lại

    blockers, warnings = [], []
    if tests_passed is False:
        blockers.append("pytest đỏ")
    failures, table = gate(new["summary"]["overall"][arm], base["summary"]["overall"][arm])
    blockers += failures
    acc_n, acc_b = (x["summary"]["overall"][arm].get("answer_accuracy") for x in (new, base))
    if acc_n is not None and acc_b is not None and acc_n < acc_b - MAX_ACCURACY_DROP:
        blockers.append(f"answer_accuracy: {acc_b:.3f} → {acc_n:.3f} (cho phép giảm tối đa {MAX_ACCURACY_DROP})")
    if n :=new["meta"].get("n_degraded"):
        blockers.append(f"{n} câu đo thiếu bước — số đo không tin được, chạy lại")
    if new["meta"].get("gold_set") != base["meta"].get("gold_set"):
        blockers.append("gold set khác baseline — không so được")
    d = analyze_defects(new, base, arm)
    crit = [x["id"] for x in d["defects"] if x["severity"] == "critical"]
    if crit:
        blockers.append(f"{len(crit)} defect critical (trả lời sai có thẩm quyền): {', '.join(crit)}")
    other = [x for x in d["defects"] if x["severity"] in ("major", "minor")]
    if other:
        warnings.append(f"{len(other)} câu thoái lui major/minor, {len(d['fixed'])} câu được sửa")
    return {
        "decision": "NO-GO" if blockers else "GO",
        "blockers": blockers, "warnings": warnings,
        "metrics_table": table, "defects": d,
        "index": new["meta"].get("index"), "git_sha": new["meta"].get("git_sha"),
    }


def to_markdown(card: dict) -> str:
    d = card["defects"]
    md = [f"## Release scorecard: **{card['decision']}**", "", f"index `{card['index']}` · git `{card['git_sha']}`", "",
          card["metrics_table"], ""]
    if card["blockers"]:
        md += ["**Blockers**", *[f"- {b}" for b in card["blockers"]], ""]
    if card["warnings"]:
        md += ["**Cảnh báo**", *[f"- {w}" for w in card["warnings"]], ""]
    if d["defects"]:
        md += ["| câu | mức | loại | gold_rank trước→sau |", "|---|---|---|---|"]
        md += [f"| `{x['id']}` | {x['severity']} | {x['category']} | {x['evidence']['gold_rank'][0]}→"
               f"{x['evidence']['gold_rank'][1]} |" for x in d["defects"]]
    return "\n".join(md)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("report", type=Path)
    p.add_argument("--baseline", type=Path, default=Path("reports/eval_baseline.json"))
    p.add_argument("--arm", default="pre_filter")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    load = lambda f: json.loads(f.read_text(encoding="utf-8"))
    card = release_scorecard(load(args.report), load(args.baseline), args.arm)
    print(json.dumps(card, ensure_ascii=False, indent=2) if args.json else to_markdown(card))
    sys.exit(0 if card["decision"] == "GO" else 1)


if __name__ == "__main__":
    main()
