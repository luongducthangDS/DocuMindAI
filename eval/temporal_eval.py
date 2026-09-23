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
  ④ pre_filter       the effective-date predicate is pushed INTO the vector store,
                     so the candidate pool and the reranker only ever see clauses
                     in force. Unlike ①-③ this arm cannot share the single
                     retrieval call — the filter is part of the query, which is
                     the whole point: ③ spends its top-8 budget on clauses it is
                     about to discard, ④ never retrieves them.

Metrics per arm:
  • answer_accuracy   — answer contains every expect_contains group and none of
                        expect_absent (gold written before the first run)
  • context_gold      — gold document/clause version present in the context
  • context_distractor— a wrong-version sibling present in the context
  • context_clean     — gold present AND no distractor (retrieval-level, no LLM)
  • recall_at_k / mrr — rank of the (complete) gold evidence, over questions that
                        have a gold clause
  • faithfulness      — LLM judge: share of answer claims backed by context (--judge)
  • *_ms_p50 / p95    — retrieval and end-to-end latency

  --mlflow [RUN_NAME] logs params (git sha, models, reranker) + metrics to MLflow.

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

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.hf_env import use_local_hf_cache  # noqa: E402

# Trước mọi import sentence_transformers. Eval chỉ dùng model đã cache nên offline=True.
# Lưu ý: đây là module vẫn bị import từ nơi khác (score_context được dùng lại),
# nên việc set env ở đây phải vô hại với importer — use_local_hf_cache chỉ trỏ cache
# về repo, và mọi entrypoint cần tải model đều tự gọi lại với offline=False.
use_local_hf_cache(offline=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(_REPO_ROOT))

from loguru import logger  # noqa: E402

import src.logger  # noqa: E402,F401 — init loguru sinks
from src.ingestion.manifest import corpus_earliest_point_in_time  # noqa: E402
from src.rag.temporal import is_out_of_range, versions_in_force  # noqa: E402

ARMS = ("no_temporal", "prompt_only", "temporal_filter", "pre_filter")
DEFAULT_GOLD = _REPO_ROOT / "data" / "eval" / "temporal_questions.json"
DEFAULT_OUTPUT = _REPO_ROOT / "reports" / "temporal_eval.json"
RECALL_KS = (1, 3, 5, 8)  # 8 = top_n the generator actually receives


def _percentile(values: list[float], p: int) -> float | None:
    """Nearest-rank percentile — no interpolation, so p95 is a latency that happened."""
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[max(0, -(-p * len(ordered) // 100) - 1)])


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

def gold_rank(chunks: list, question: dict) -> int | None:
    """1-based rank at which the gold evidence is complete — basis of recall@k / MRR.

    Multi-clause questions need every clause, so the rank is the WORST of theirs:
    recall@3 means "all the evidence was in the top 3", not "some of it was".
    A clause present only in a superseded/future version does not count.
    None = the question has no gold clause (refusals) or it never showed up.
    """
    required = list(question.get("source_clauses") or [])
    if not required and question.get("source_clause"):
        required = [question["source_clause"]]
    if not required:
        return None
    bad_versions = set(question.get("distractor_versions") or [])
    worst = 0
    for uid in required:
        rank = next(
            (
                i for i, c in enumerate(chunks, 1)
                if str(c.metadata.get("clause_uid", "")) == uid
                and str(c.metadata.get("version_id", "")) not in bad_versions
            ),
            None,
        )
        if rank is None:
            return None
        worst = max(worst, rank)
    return worst


def has_gold_clause(question: dict) -> bool:
    return bool(question.get("source_clauses") or question.get("source_clause"))


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

    # Multi-clause questions (data/eval/hard_questions.json) name every clause the
    # answer has to combine. All of them must be in context: a context holding the
    # probation *length* but not the probation *pay* cannot support a full answer,
    # however relevant it looks. Coverage is reported too, so "found 1 of 2" is
    # distinguishable from "found neither".
    required = list(question.get("source_clauses") or [])
    if required:
        found = [c for c in required if c in clauses]
        coverage = len(found) / len(required)
        gold_hit = gold_hit and coverage == 1.0
    else:
        coverage = 1.0 if gold_hit else 0.0

    distractor_hit = bool(docs & bad_docs) or bool(versions & bad_versions)
    return {
        "gold_rank": gold_rank(chunks, question),
        "gold_coverage": coverage,
        "gold": gold_hit,
        "distractor": distractor_hit,
        "clean": gold_hit and not distractor_hit,
        "n_chunks": len(chunks),
    }


# ── Faithfulness (LLM judge) ──────────────────────────────────────────────────

_JUDGE_PROMPT = """Bạn là giám khảo kiểm tra tính trung thực của câu trả lời pháp lý.
Tách CÂU TRẢ LỜI thành các khẳng định thực tế (con số, điều kiện, thời hạn, quyền/nghĩa vụ).
Đếm bao nhiêu khẳng định được NGỮ CẢNH hỗ trợ trực tiếp. Bỏ qua câu dẫn, lời khuyên chung,
trích dẫn số điều/văn bản. Chỉ trả về JSON một dòng: {{"claims": <int>, "supported": <int>}}

NGỮ CẢNH:
{context}

CÂU TRẢ LỜI:
{answer}"""


def judge_faithfulness(answer: str, chunks: list) -> float | None:
    """Share of the answer's factual claims supported by the context it was given.

    None when there is nothing to judge (refusal, no claims) or the judge failed —
    a missing score must never be averaged in as 0 or 1.
    """
    from src.rag.generator import gemini_generate

    if not chunks or not answer.strip():
        return None
    context = "\n\n".join(c.text[:3000] for c in chunks)[:15000]
    try:
        raw = gemini_generate(_JUDGE_PROMPT.format(context=context, answer=answer))
        m = re.search(r"\{.*?\}", raw, re.S)
        verdict = json.loads(m.group(0)) if m else {}
        claims, supported = int(verdict["claims"]), int(verdict["supported"])
    except Exception as exc:  # judge is best-effort: quota/parse errors → no score
        logger.warning("faithfulness judge thất bại: {}", exc)
        return None
    if claims <= 0:
        return None
    return min(supported, claims) / claims


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
    """Dựng đúng retriever mà production dùng — kể cả việc có bật reranker hay không.

    Trước đây hard-code rerank=True, nên eval đo một cấu hình mà API có thể không
    chạy (ENABLE_RERANKER=false). Số đo phải nói về hệ thống thật, nên đọc settings.

    Dùng chính `_init_rag_sync` của API (không lắp lại một bản trông giống): cùng
    VECTOR_STORE_PROVIDER (Chroma local / Qdrant ở CI), cùng BM25 corpus, cùng
    singleton `_active_index` mà arm pre_filter (retrieve_with_context) đọc.
    """
    import src.rag.retriever as r_module
    from src.api.main import _init_rag_sync

    _init_rag_sync()
    return r_module._active_retriever


def use_query_embedding_cache(path: Path) -> None:
    """Disk-cache query embeddings for this process (eval only, production untouched).

    Gemini embedding quota is ~1000/day/project and each question embeds twice
    (shared retrieve + pre_filter), so a 200-question run per PR would drain it.
    Gold questions never change between runs, so their vectors needn't either.
    Cached queries skip the API round-trip — latency from such a run is NOT
    production latency; the report marks it (meta.embed_cache).
    """
    import atexit

    from src.rag.embedder import get_embedder

    cache: dict[str, list[float]] = (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    )
    cls = type(get_embedder())
    original = cls._get_query_embedding

    def cached(self, query: str) -> list[float]:
        key = f"{self.model_name}|{query}"
        if key not in cache:
            cache[key] = original.__get__(self, cls)(query)  # llama-index wrapper needs binding
        return cache[key]

    def save() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache), encoding="utf-8")

    cls._get_query_embedding = cached
    atexit.register(save)


def _reranker_active() -> bool:
    import src.rag.retriever as r_module

    return bool(r_module._reranker_active)


def run(
    questions: list[dict],
    arms: tuple[str, ...],
    retrieval_only: bool,
    judge_arms: tuple[str, ...] = (),
) -> dict[str, Any]:
    from src.rag.context import RetrievalContext
    from src.rag.generator import generate_answer
    from src.rag.retriever import nodes_to_chunks, retrieve_with_context

    retriever = _build_retriever()
    earliest = corpus_earliest_point_in_time()
    rows: list[dict] = []
    # Warm-up, untimed: the first retrieve pays for loading the reranker, which
    # would otherwise land in question 1's latency and dominate p95 on small sets.
    if questions:
        retriever.retrieve(questions[0]["question"])

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
            "has_gold": has_gold_clause(q),
            "retrieve_ms": retrieve_ms,
            "arms": {},
        }

        for arm in arms:
            if arm == "pre_filter":
                t_pf = time.time()
                used = retrieve_with_context(
                    q["question"], RetrievalContext(as_of_date=q["as_of_date"])
                )
                oor = is_out_of_range(q["as_of_date"], earliest)
                if oor:
                    used = []
                prompt_as_of = q["as_of_date"]
                pre_filter_ms = int((time.time() - t_pf) * 1000)
            else:
                used, oor, prompt_as_of = apply_arm(arm, chunks, q["as_of_date"], earliest)
                pre_filter_ms = None
            entry: dict[str, Any] = {"context": score_context(used, q)}
            if pre_filter_ms is not None:
                entry["retrieve_ms"] = pre_filter_ms

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
                if arm in judge_arms and result["used_llm"]:
                    entry["faithfulness"] = judge_faithfulness(result["answer"], used)

            row["arms"][arm] = entry

        rows.append(row)

    return {
        "meta": {
            "gold_set": str(DEFAULT_GOLD.relative_to(_REPO_ROOT)).replace("\\", "/"),
            "n_questions": len(questions),
            "arms": list(arms),
            "retrieval_only": retrieval_only,
            "corpus_earliest_point_in_time": earliest,
            "retriever": "hybrid BM25 + dense, RRF fusion → top-8",
            # Config can ask for the reranker while it silently fails to load
            # (missing torch/model, blocked DLL) — record what actually ran.
            "reranker_active": _reranker_active(),
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
            # .get: reports written before multi-clause scoring lack the field.
            "context_gold_coverage": sum(c.get("gold_coverage", float(c["gold"])) for c in ctx) / n,
            "context_distractor": sum(c["distractor"] for c in ctx) / n,
            "context_clean": sum(c["clean"] for c in ctx) / n,
            "avg_chunks": sum(c["n_chunks"] for c in ctx) / n,
        }
        # recall@k / MRR only over questions that HAVE a gold clause: a refusal
        # question has nothing to recall, counting it would dilute or inflate.
        ranked = [r["arms"][arm]["context"].get("gold_rank") for r in subset if r.get("has_gold")]
        if ranked:
            for k in RECALL_KS:
                agg[f"recall_at_{k}"] = sum(1 for x in ranked if x and x <= k) / len(ranked)
            agg["mrr"] = sum(1 / x for x in ranked if x) / len(ranked)
        retrieve = [r["arms"][arm].get("retrieve_ms", r["retrieve_ms"]) for r in subset]
        agg["retrieve_ms_p50"] = _percentile(retrieve, 50)
        agg["retrieve_ms_p95"] = _percentile(retrieve, 95)
        if not retrieval_only:
            agg["answer_accuracy"] = sum(
                r["arms"][arm]["score"]["correct"] for r in subset
            ) / n
            total = [
                r["arms"][arm].get("retrieve_ms", r["retrieve_ms"]) + r["arms"][arm]["generate_ms"]
                for r in subset if "generate_ms" in r["arms"][arm]
            ]
            agg["e2e_ms_p50"] = _percentile(total, 50)
            agg["e2e_ms_p95"] = _percentile(total, 95)
            faith = [
                r["arms"][arm]["faithfulness"] for r in subset
                if r["arms"][arm].get("faithfulness") is not None
            ]
            if faith:
                agg["faithfulness"] = sum(faith) / len(faith)
                agg["faithfulness_n"] = len(faith)
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
    metrics += [f"recall_at_{k}" for k in RECALL_KS] + ["mrr"]
    if not retrieval_only:
        metrics += ["answer_accuracy", "faithfulness"]

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
    for m in ("retrieve_ms_p50", "retrieve_ms_p95", "e2e_ms_p50", "e2e_ms_p95"):
        vals = [report["summary"]["overall"][arm].get(m) for arm in arms]
        if any(v is not None for v in vals):
            print(f"{m:<{width}}" + "".join(
                f"{'—' if v is None else f'{v:.0f} ms':>18}" for v in vals
            ))

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


def _git(*args: str) -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return ""


def log_to_mlflow(report: dict, output: Path, run_name: str | None) -> None:
    """One MLflow run per eval: config as params, overall summary as metrics.

    Tracking URI comes from MLFLOW_TRACKING_URI (default: sqlite ./mlflow.db, git-ignored),
    so CI and local runs land wherever the environment points them.
    """
    import mlflow

    from src.config import get_settings

    s = get_settings()
    mlflow.set_experiment("documind-eval")
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "git_sha": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": str(bool(_git("status", "--porcelain", "--untracked-files=no"))),
            "gold_set": report["meta"]["gold_set"],
        })
        mlflow.log_params({
            "n_questions": report["meta"]["n_questions"],
            "arms": ",".join(report["meta"]["arms"]),
            "retrieval_only": report["meta"]["retrieval_only"],
            "embedding_model": s.embedding_model,
            "enable_reranker": s.enable_reranker,  # requested
            "reranker_active": report["meta"].get("reranker_active"),  # actually loaded
            "reranker_model": s.reranker_model,
            "vector_store": s.vector_store_provider,
            "embed_cache": report["meta"].get("embed_cache", False),
            "generation_models": s.gemini_generation_models,
        })
        for arm, agg in report["summary"]["overall"].items():
            mlflow.log_metrics({
                f"{arm}/{k}": float(v) for k, v in agg.items() if isinstance(v, (int, float))
            })
        mlflow.log_artifact(str(output))


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
    parser.add_argument(
        "--judge",
        nargs="+",
        choices=ARMS,
        default=[],
        help="chấm faithfulness bằng LLM judge cho các arm này (thêm 1 lời gọi Gemini/câu/arm)",
    )
    parser.add_argument(
        "--mlflow",
        nargs="?",
        const="",
        default=None,
        metavar="RUN_NAME",
        help="log kết quả vào MLflow (experiment documind-eval), tuỳ chọn đặt tên run",
    )
    parser.add_argument(
        "--embed-cache",
        type=Path,
        default=None,
        help="cache embedding câu hỏi ra file (tiết kiệm quota; latency khi đó KHÔNG phải latency thật)",
    )
    args = parser.parse_args()
    if args.embed_cache:
        use_query_embedding_cache(args.embed_cache)

    questions = json.loads(args.gold.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    if args.rescore:
        report = rescore(json.loads(args.rescore.read_text(encoding="utf-8")), questions)
    else:
        report = run(questions, tuple(args.arms), args.retrieval_only, judge_arms=tuple(args.judge))
    # run() chỉ biết DEFAULT_GOLD — ghi đè bằng file thực sự chạy (--gold)
    gold = args.gold.resolve()
    report["meta"]["gold_set"] = (
        str(gold.relative_to(_REPO_ROOT)) if gold.is_relative_to(_REPO_ROOT) else str(gold)
    ).replace("\\", "/")
    report["meta"]["embed_cache"] = bool(args.embed_cache)
    print_report(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi báo cáo: {args.output}")
    if args.mlflow is not None:
        log_to_mlflow(report, args.output, args.mlflow or None)


if __name__ == "__main__":
    main()
