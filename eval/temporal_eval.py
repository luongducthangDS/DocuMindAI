"""
eval/temporal_eval.py — A/B benchmark for temporal-aware retrieval.

Question: does filtering retrieved clauses by the version in force at
`as_of_date` actually change how often DocuMind answers a time-sensitive
question correctly, or would telling the LLM the date be enough?

Three arms, all sharing ONE retrieval call per question so the only thing that
varies is what happens above retrieval:

  ① no_temporal      raw retrieval → generator, generator never sees as_of_date
                     (the system as it was before `src/rag/temporal.py`)
  ② prompt_only      raw retrieval → generator, generator IS told as_of_date
                     (isolates "just tell the model the date")
  ③ temporal_filter  versions_in_force(as_of) + out-of-range guard → generator
                     told as_of_date (what `do_temporal_filter` does in the graph)

Metrics per arm:
  • answer_accuracy   — answer contains every expect_contains group and none of
                        expect_absent (gold written before the first run)
  • context_gold      — gold document/clause version present in the context
  • context_distractor— a wrong-version sibling present in the context
  • context_clean     — gold present AND no distractor (retrieval-level, no LLM)

Usage:
  python eval/temporal_eval.py --retrieval-only          # no LLM, deterministic
  python eval/temporal_eval.py                           # full A/B (needs LLM key)
  python eval/temporal_eval.py --arms no_temporal temporal_filter --limit 6
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# Force HF cache to the local path before any sentence_transformers import —
# the system HF_HOME points at a Google Drive mount that is usually offline.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_HF = str(_REPO_ROOT / "data" / "hf_cache")
for _k in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
    os.environ[_k] = _LOCAL_HF
os.environ["HF_HUB_OFFLINE"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(_REPO_ROOT))

from loguru import logger  # noqa: E402

import src.logger  # noqa: E402,F401 — init loguru sinks
from src.ingestion.manifest import corpus_earliest_point_in_time  # noqa: E402
from src.rag.temporal import is_out_of_range, versions_in_force  # noqa: E402

ARMS = ("no_temporal", "prompt_only", "temporal_filter")
DEFAULT_GOLD = _REPO_ROOT / "data" / "eval" / "temporal_questions.json"
DEFAULT_OUTPUT = _REPO_ROOT / "reports" / "temporal_eval.json"


# ── Answer scoring ────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    """Lowercase, drop markdown emphasis, collapse whitespace.

    The generator bolds figures ("ngày làm việc thứ **11**"), which would break
    a plain substring check against the gold string "thứ 11" — that is a defect
    of the measuring instrument, not of the answer, so emphasis markers go.
    """
    text = (text or "").replace(" ", " ")
    text = re.sub(r"[*_`]+", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def score_answer(answer: str, question: dict) -> dict:
    """Gold-string scoring: every `expect_contains` group hit, no `expect_absent`."""
    norm = _norm(answer)
    missing_groups = [
        group for group in question["expect_contains"]
        if not any(_norm(alt) in norm for alt in group)
    ]
    forbidden_hits = [bad for bad in question["expect_absent"] if _norm(bad) in norm]
    return {
        "correct": not missing_groups and not forbidden_hits,
        "missing": [g[0] for g in missing_groups],
        "forbidden_present": forbidden_hits,
    }


# ── Context scoring (no LLM needed) ───────────────────────────────────────────

def score_context(chunks: list, question: dict) -> dict:
    """Did the context carry the right version, and did a wrong one sneak in?"""
    docs = {str(c.metadata.get("doc_id", "")) for c in chunks}
    clauses = {str(c.metadata.get("clause_uid", "")) for c in chunks}
    versions = {str(c.metadata.get("version_id", "")) for c in chunks}

    bad_docs = set(question.get("distractor_docs") or [])
    bad_versions = set(question.get("distractor_versions") or [])

    if question["temporal_kind"] == "out_of_range":
        # Nothing in the corpus applies before coverage starts: the only clean
        # context is an empty one.
        gold = not chunks
        distractor = bool(chunks)
        return {"gold": gold, "distractor": distractor, "clean": gold, "n_chunks": len(chunks)}

    gold_doc = question["source_doc"]
    gold_clause = question["source_clause"]
    gold_hit = True
    if gold_doc:
        gold_hit = gold_doc in docs
    if gold_hit and gold_clause and bad_versions:
        # Clause-level case: the right clause must be present in a version that
        # is not one of the superseded/future siblings.
        gold_hit = any(
            str(c.metadata.get("clause_uid", "")) == gold_clause
            and str(c.metadata.get("version_id", "")) not in bad_versions
            for c in chunks
        )
    elif gold_hit and gold_clause:
        gold_hit = gold_clause in clauses

    distractor_hit = bool(docs & bad_docs) or bool(versions & bad_versions)
    return {
        "gold": gold_hit,
        "distractor": distractor_hit,
        "clean": gold_hit and not distractor_hit,
        "n_chunks": len(chunks),
    }


# ── Arms ──────────────────────────────────────────────────────────────────────

def apply_arm(arm: str, chunks: list, as_of: str, earliest: str) -> tuple[list, bool, str | None]:
    """Return (chunks the generator sees, time_out_of_range, as_of for prompt)."""
    if arm == "no_temporal":
        return chunks, False, None
    if arm == "prompt_only":
        return chunks, False, as_of
    if arm == "temporal_filter":
        if is_out_of_range(as_of, earliest):
            return [], True, as_of
        return versions_in_force(chunks, as_of), False, as_of
    raise ValueError(f"arm không hợp lệ: {arm}")


# ── Runner ────────────────────────────────────────────────────────────────────

def _build_retriever():
    from eval.rag_comparison import _init_rag_shared
    from src.rag.retriever import build_hybrid_retriever

    index, _collection, all_nodes, _embedder = _init_rag_shared()
    return build_hybrid_retriever(index, nodes=all_nodes, rerank=True)


def run(
    questions: list[dict],
    arms: tuple[str, ...],
    retrieval_only: bool,
) -> dict[str, Any]:
    from src.rag.generator import generate_answer
    from src.rag.retriever import nodes_to_chunks

    retriever = _build_retriever()
    earliest = corpus_earliest_point_in_time()
    rows: list[dict] = []

    for i, q in enumerate(questions, 1):
        t0 = time.time()
        chunks = nodes_to_chunks(retriever.retrieve(q["question"]))
        retrieve_ms = int((time.time() - t0) * 1000)
        logger.info(
            "[{}/{}] {} @{} — {} chunks",
            i, len(questions), q["id"], q["as_of_date"], len(chunks),
        )

        row: dict[str, Any] = {
            "id": q["id"],
            "pair_id": q["pair_id"],
            "question": q["question"],
            "as_of_date": q["as_of_date"],
            "temporal_kind": q["temporal_kind"],
            "expected_behavior": q["expected_behavior"],
            "retrieve_ms": retrieve_ms,
            "arms": {},
        }

        for arm in arms:
            used, oor, prompt_as_of = apply_arm(arm, chunks, q["as_of_date"], earliest)
            entry: dict[str, Any] = {"context": score_context(used, q)}

            if not retrieval_only:
                t1 = time.time()
                result = generate_answer(
                    q["question"],
                    used,
                    as_of_date=prompt_as_of,
                    time_out_of_range=oor,
                    earliest_covered=earliest,
                    min_score=0.0,
                )
                entry["answer"] = result["answer"]
                entry["used_llm"] = result["used_llm"]
                entry["generate_ms"] = int((time.time() - t1) * 1000)
                entry["score"] = score_answer(result["answer"], q)

            row["arms"][arm] = entry

        rows.append(row)

    return {
        "meta": {
            "gold_set": str(DEFAULT_GOLD.relative_to(_REPO_ROOT)).replace("\\", "/"),
            "n_questions": len(questions),
            "arms": list(arms),
            "retrieval_only": retrieval_only,
            "corpus_earliest_point_in_time": earliest,
            "retriever": "hybrid BM25 + dense, RRF fusion, cross-encoder rerank → top-8",
        },
        "rows": rows,
        "summary": summarise(rows, arms, retrieval_only),
    }


def summarise(rows: list[dict], arms: tuple[str, ...], retrieval_only: bool) -> dict:
    kinds = sorted({r["temporal_kind"] for r in rows})
    out: dict[str, Any] = {"overall": {}, "by_kind": {}}

    def _agg(subset: list[dict], arm: str) -> dict:
        n = len(subset)
        if not n:
            return {}
        ctx = [r["arms"][arm]["context"] for r in subset]
        agg = {
            "n": n,
            "context_gold": sum(c["gold"] for c in ctx) / n,
            "context_distractor": sum(c["distractor"] for c in ctx) / n,
            "context_clean": sum(c["clean"] for c in ctx) / n,
            "avg_chunks": sum(c["n_chunks"] for c in ctx) / n,
        }
        if not retrieval_only:
            agg["answer_accuracy"] = sum(
                r["arms"][arm]["score"]["correct"] for r in subset
            ) / n
        return agg

    for arm in arms:
        out["overall"][arm] = _agg(rows, arm)
    for kind in kinds:
        subset = [r for r in rows if r["temporal_kind"] == kind]
        out["by_kind"][kind] = {arm: _agg(subset, arm) for arm in arms}
    return out


# ── Reporting ─────────────────────────────────────────────────────────────────

def _pct(x: float | None) -> str:
    return "  —  " if x is None else f"{x * 100:5.1f}%"


def print_report(report: dict) -> None:
    arms = report["meta"]["arms"]
    retrieval_only = report["meta"]["retrieval_only"]
    metrics = ["context_gold", "context_distractor", "context_clean"]
    if not retrieval_only:
        metrics.append("answer_accuracy")

    print("\n" + "=" * 78)
    print(f"TEMPORAL A/B — {report['meta']['n_questions']} câu nhạy thời điểm")
    print("=" * 78)

    width = max(len(m) for m in metrics) + 2
    print(f"\n{'metric':<{width}}" + "".join(f"{a:>18}" for a in arms))
    print("-" * (width + 18 * len(arms)))
    for m in metrics:
        line = f"{m:<{width}}"
        for arm in arms:
            line += f"{_pct(report['summary']['overall'][arm].get(m)):>18}"
        print(line)

    print("\nTheo nhóm câu hỏi:")
    for kind, per_arm in report["summary"]["by_kind"].items():
        n = next(iter(per_arm.values())).get("n", 0)
        print(f"\n  {kind} (n={n})")
        for m in metrics:
            line = f"    {m:<{width}}"
            for arm in arms:
                line += f"{_pct(per_arm[arm].get(m)):>18}"
            print(line)

    if not retrieval_only:
        print("\nCâu sai (arm temporal_filter):")
        wrong = [
            r for r in report["rows"]
            if "temporal_filter" in r["arms"]
            and not r["arms"]["temporal_filter"]["score"]["correct"]
        ]
        if not wrong:
            print("    (không có)")
        for r in wrong:
            s = r["arms"]["temporal_filter"]["score"]
            print(
                f"    {r['id']} @{r['as_of_date']}: "
                f"thiếu={s['missing']} cấm={s['forbidden_present']}"
            )
    print()


def rescore(report: dict, questions: list[dict]) -> dict:
    """Re-apply `score_answer` to answers already stored in a report.

    Used when the scoring rule itself is corrected: the model output stays
    frozen, so no LLM call and no chance of tuning the gold set to the output.
    """
    by_id = {q["id"]: q for q in questions}
    arms = tuple(report["meta"]["arms"])
    for row in report["rows"]:
        q = by_id.get(row["id"])
        if q is None:
            continue
        for arm in arms:
            entry = row["arms"].get(arm)
            if entry and "answer" in entry:
                entry["score"] = score_answer(entry["answer"], q)
    report["summary"] = summarise(report["rows"], arms, report["meta"]["retrieval_only"])
    report["meta"]["rescored"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B benchmark cho temporal-aware retrieval")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--limit", type=int, default=0, help="chỉ chạy N câu đầu (smoke test)")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="chỉ đo ở tầng ngữ cảnh, không gọi LLM",
    )
    parser.add_argument(
        "--rescore",
        type=Path,
        default=None,
        help="chấm lại một báo cáo đã có (không gọi LLM, không đổi câu trả lời)",
    )
    args = parser.parse_args()

    questions = json.loads(args.gold.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    if args.rescore:
        report = rescore(json.loads(args.rescore.read_text(encoding="utf-8")), questions)
    else:
        report = run(questions, tuple(args.arms), args.retrieval_only)
    print_report(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi báo cáo: {args.output}")


if __name__ == "__main__":
    main()
