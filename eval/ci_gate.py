"""
eval/ci_gate.py — fail a PR whose retrieval quality regresses against the committed baseline.

Compares one arm of a fresh temporal_eval report with reports/eval_baseline.json and
prints a Markdown table (appended to $GITHUB_STEP_SUMMARY in CI).

  python eval/ci_gate.py reports/ci_eval.json --baseline reports/eval_baseline.json

Updating the baseline is a deliberate act — copy the new report over it in the PR that
earns the improvement, so the diff shows the number moving.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# metric → max allowed drop (absolute). Retrieval over a live index with an API embedder
# is near- but not bit-deterministic, so an exact-equality gate would flake.
GATED = {"recall_at_1": 0.03, "recall_at_5": 0.02, "recall_at_8": 0.02, "mrr": 0.03, "context_clean": 0.02}
SHOWN = [*GATED, "retrieve_ms_p50", "retrieve_ms_p95"]


def gate(new: dict, base: dict) -> tuple[list[str], str]:
    failures, lines = [], ["| metric | baseline | PR | Δ |", "|---|---:|---:|---:|"]
    for m in SHOWN:
        b, n = base.get(m), new.get(m)
        if b is None or n is None:
            continue
        ms = m.endswith("_ms_p50") or m.endswith("_ms_p95")
        fmt = (lambda x: f"{x:.0f} ms") if ms else (lambda x: f"{x:.3f}")
        mark = ""
        if m in GATED and n < b - GATED[m]:
            failures.append(f"{m}: {b:.3f} → {n:.3f} (cho phép giảm tối đa {GATED[m]})")
            mark = " ❌"
        lines.append(f"| {m} | {fmt(b)} | {fmt(n)} | {n - b:+.3f}{mark} |" if not ms
                     else f"| {m} | {fmt(b)} | {fmt(n)} | {n - b:+.0f} ms |")
    return failures, "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("report", type=Path)
    p.add_argument("--baseline", type=Path, default=Path("reports/eval_baseline.json"))
    p.add_argument("--arm", default="pre_filter")
    args = p.parse_args()

    new = json.loads(args.report.read_text(encoding="utf-8"))
    base = json.loads(args.baseline.read_text(encoding="utf-8"))
    if new["meta"]["gold_set"] != base["meta"]["gold_set"] or new["meta"]["n_questions"] != base["meta"]["n_questions"]:
        sys.exit("Gold set khác baseline — cập nhật reports/eval_baseline.json cùng PR đổi gold set.")
    failures, table = gate(new["summary"]["overall"][args.arm], base["summary"]["overall"][args.arm])

    md = f"### Retrieval eval — arm `{args.arm}`, {new['meta']['n_questions']} câu\n\n{table}\n"
    md += "\n" + ("**FAIL**\n- " + "\n- ".join(failures) if failures else "**PASS** — không metric nào tụt quá ngưỡng.") + "\n"
    print(md)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as f:
            f.write(md)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
