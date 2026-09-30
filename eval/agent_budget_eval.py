"""
eval/agent_budget_eval.py — số lời gọi Gemini mỗi câu hỏi, đo trên ĐÚNG đường agent.

Vì sao có file này: temporal_eval.py gọi thẳng generator, bỏ qua graph — router,
grade, reformulate, contextualize không nằm trên đường nó đo, nên nó không thấy được
bất kỳ thay đổi nào về số lời gọi LLM. Ở đây mỗi câu đi qua `run_agent` như REST API.

Đo mỗi câu:
  • llm_calls      — mọi lần gọi generate_content (kể cả lần hỏng, xoay cặp), chia
                     theo mục đích: diacritics / contextualize / router / grade /
                     reformulate / answer / other
  • embeds         — số lần embed câu hỏi (đếm cả khi lấy từ cache — production thì
                     không có cache)
  • correct        — score_answer của temporal_eval (expect_contains / expect_absent)
  • citation       — citation_groundedness/validity trên đúng danh sách chunk mà
                     generator đánh số [1..N]

A/B: chạy một lần trước khi đổi code (baseline), một lần sau với `--compare`, cùng
tập câu (lấy id từ baseline) và cùng MỘT model ghim cứng — để hai lần chạy chỉ khác
nhau ở logic graph, không ở việc vòng xoay rơi vào model nào.

Usage:
  python eval/agent_budget_eval.py --sample 30 --output reports/agent_budget_before.json
  python eval/agent_budget_eval.py --compare reports/agent_budget_before.json \
      --output reports/agent_budget_after.json
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import random
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_GOLD = _REPO_ROOT / "data" / "eval" / "legal_qa_200.json"

# Hàm trên call stack → mục đích của lời gọi Gemini. Hàm đầu tiên khớp (từ trong ra) thắng.
_PURPOSE_BY_FRAME = {
    "_restore_diacritics": "diacritics",
    "_normalize_query": "contextualize",
    "_contextualize_query": "contextualize",
    "router_node": "router",
    "_call_judge_llm": "grade",
    "reformulate_node": "reformulate",
    "_call_gemini": "answer",
    "stream_answer": "answer",
}


def _purpose() -> str:
    for frame in inspect.stack()[2:]:
        if frame.function in _PURPOSE_BY_FRAME:
            return _PURPOSE_BY_FRAME[frame.function]
    return "other"


class _Meter:
    """Bộ đếm cho câu hỏi đang chạy — cài một lần, reset mỗi câu."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.llm: Counter[str] = Counter()
        self.embeds = 0
        self.context_chunks: list = []

    def install(self) -> None:
        from src.rag import generator
        from src.rag.embedder import get_embedder

        meter = self
        real_call = generator.gemini_call

        def counted_call(*args, **kwargs):
            meter.llm[_purpose()] += 1
            return real_call(*args, **kwargs)

        generator.gemini_call = counted_call  # graph.py import lúc gọi → thấy bản này

        real_build = generator._build_context

        def capture_build(chunks):
            meter.context_chunks = list(chunks)  # thứ tự này = số [N] trong câu trả lời
            return real_build(chunks)

        generator._build_context = capture_build

        cls = type(get_embedder())
        real_embed = cls._get_query_embedding

        def counted_embed(self, query):
            meter.embeds += 1
            return real_embed(self, query)

        cls._get_query_embedding = counted_embed


def _pick(questions: list[dict], n: int, seed: int, ids: list[str] | None) -> list[dict]:
    if ids:
        by_id = {q["id"]: q for q in questions}
        return [by_id[i] for i in ids]
    if n <= 0 or n >= len(questions):
        return list(questions)
    return random.Random(seed).sample(questions, n)


def run(questions: list[dict]) -> dict[str, Any]:
    from eval.metrics import citation_groundedness, citation_validity
    from eval.temporal_eval import _build_retriever, score_answer
    from src.agent.graph import run_agent

    _build_retriever()
    meter = _Meter()
    meter.install()

    rows, answers, records = [], [], []
    for i, q in enumerate(questions, 1):
        meter.reset()
        t0 = time.perf_counter()
        error = None
        try:
            result = asyncio.run(run_agent(
                q["question"], session_id=f"budget-{uuid.uuid4().hex[:8]}",
                as_of_date=q.get("as_of_date"),
            ))
            answer = result.get("answer", "")
        except Exception as exc:  # noqa: BLE001 — một câu hỏng không được giết cả lượt đo
            result, answer, error = {}, "", f"{type(exc).__name__}: {exc}"
        ms = int((time.perf_counter() - t0) * 1000)
        recs = [c.metadata for c in meter.context_chunks]
        llm_total = sum(meter.llm.values())
        row = {
            "id": q["id"],
            "category": q.get("category"),
            "correct": bool(score_answer(answer, q)["correct"]) if not error else False,
            "llm_calls": llm_total,
            "llm_by_purpose": dict(meter.llm),
            "embeds": meter.embeds,
            "gemini_total": llm_total + meter.embeds,
            "intent": result.get("intent"),
            "grade": result.get("grade"),
            "retry_count": result.get("retry_count", 0),
            "degraded": result.get("degraded", []),
            "latency_ms": ms,
            "error": error,
            "answer": answer[:600],
        }
        rows.append(row)
        answers.append(answer)
        records.append(recs)
        print(f"[{i}/{len(questions)}] {q['id']:<10} correct={row['correct']!s:<5} "
              f"llm={llm_total} {dict(meter.llm)} embeds={meter.embeds} {ms}ms"
              + (f" ERROR {error[:80]}" if error else ""), flush=True)

    n = len(rows)
    lat = sorted(r["latency_ms"] for r in rows)
    purposes: Counter[str] = Counter()
    for r in rows:
        purposes.update(r["llm_by_purpose"])
    summary = {
        "n": n,
        "accuracy": round(sum(r["correct"] for r in rows) / n, 4),
        "citation_groundedness": citation_groundedness(questions, answers, records),
        "citation_validity": citation_validity(answers, records),
        "llm_calls_mean": round(sum(r["llm_calls"] for r in rows) / n, 3),
        "gemini_total_mean": round(sum(r["gemini_total"] for r in rows) / n, 3),
        "share_gemini_total_le_2": round(sum(r["gemini_total"] <= 2 for r in rows) / n, 4),
        "llm_calls_by_purpose": dict(purposes),
        "latency_ms_p50": lat[n // 2],
        "latency_ms_p95": lat[min(n - 1, int(n * 0.95))],
        "errors": sum(bool(r["error"]) for r in rows),
    }
    return {"summary": summary, "rows": rows}


def compare(before: dict, after: dict) -> dict[str, Any]:
    """So từng câu: câu nào đổi đúng→sai (điều duy nhất n=30 nói được chắc chắn)."""
    b = {r["id"]: r for r in before["rows"]}
    a = {r["id"]: r for r in after["rows"]}
    common = [i for i in a if i in b]
    return {
        "broke": [i for i in common if b[i]["correct"] and not a[i]["correct"]],
        "fixed": [i for i in common if not b[i]["correct"] and a[i]["correct"]],
        "delta": {k: (round(after["summary"][k] - before["summary"][k], 4)
                      if isinstance(after["summary"].get(k), (int, float))
                      and isinstance(before["summary"].get(k), (int, float)) else None)
                  for k in ("accuracy", "citation_groundedness", "citation_validity",
                            "llm_calls_mean", "gemini_total_mean", "share_gemini_total_le_2",
                            "latency_ms_p50", "latency_ms_p95")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--sample", type=int, default=30, help="0 = cả bộ")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path, help="report baseline — dùng lại đúng tập câu của nó")
    parser.add_argument("--pin-model", default="gemini-3.1-flash-lite",
                        help="ghim MỘT model cho mọi vai (generate/complex/judge); '' = theo .env")
    parser.add_argument("--embed-cache", type=Path, help="cache vector câu hỏi (tiết kiệm quota embed)")
    args = parser.parse_args()

    # Không đẩy trace eval lên Langfuse production.
    os.environ["LANGFUSE_PUBLIC_KEY"] = ""
    os.environ["LANGFUSE_SECRET_KEY"] = ""
    if args.pin_model:
        for var in ("GEMINI_GENERATION_MODELS", "GEMINI_GENERATION_MODELS_COMPLEX", "GEMINI_JUDGE_MODELS"):
            os.environ[var] = args.pin_model

    from src.hf_env import use_local_hf_cache

    use_local_hf_cache(offline=True)
    if args.embed_cache:
        from eval.temporal_eval import use_query_embedding_cache

        use_query_embedding_cache(args.embed_cache)

    questions = json.loads(args.gold.read_text(encoding="utf-8"))
    baseline = json.loads(args.compare.read_text(encoding="utf-8")) if args.compare else None
    ids = [r["id"] for r in baseline["rows"]] if baseline else None
    picked = _pick(questions, args.sample, args.seed, ids)

    report = run(picked)
    report["meta"] = {
        "gold": str(args.gold.relative_to(_REPO_ROOT)) if args.gold.is_relative_to(_REPO_ROOT) else str(args.gold),
        "sample": len(picked), "seed": args.seed, "pin_model": args.pin_model,
        "embed_cache": bool(args.embed_cache),
    }
    if baseline:
        report["compare"] = compare(baseline, report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "compare": report.get("compare")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
