"""
eval/scoring_ab.py — A/B giữa HAI CÁCH CHẤM trên CÙNG một lần retrieval.

Câu hỏi: thước đo mới (`clause_uid` + `version_id`, DEC-0004) có thật sự nói
khác thước cũ (token-F1 ≥ 0.15) không, hay chỉ là cùng một con số đội tên mới?

Biến duy nhất là cách chấm. Retrieval chạy **một lần** cho mỗi câu hỏi và cả hai
cách chấm cùng đọc đúng danh sách chunk đó, nên mọi chênh lệch trong bảng kết quả
là do thước, không do hệ thống.

Không gọi LLM — chạy được không cần API key.

Usage:
  python eval/scoring_ab.py
  python eval/scoring_ab.py --limit 5 --output reports/scoring_ab_smoke.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import get_settings  # noqa: E402
from src.hf_env import use_local_hf_cache  # noqa: E402

# Offline chặn mọi lần gọi ra huggingface.co — đúng khi model chạy tại chỗ, nhưng
# EMBEDDING_PROVIDER=hf_api thì chính lời gọi đó là cách lấy vector, và offline
# biến nó thành OfflineModeIsEnabled. Đọc settings trước (src.config chỉ cần
# pydantic, không kéo theo HF) để quyết định.
use_local_hf_cache(offline=get_settings().embedding_provider.strip().lower() != "hf_api")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from loguru import logger  # noqa: E402

import src.logger  # noqa: E402,F401 — init loguru sinks
from eval.metrics import _token_set, mrr_at_k, recall_at_k  # noqa: E402

DEFAULT_GOLD = _REPO_ROOT / "data" / "eval" / "temporal_questions.json"
DEFAULT_OUTPUT = _REPO_ROOT / "reports" / "scoring_ab.json"


# ── Thước cũ (bản sao, chỉ để so sánh) ────────────────────────────────────────
#
# Đây là bản sao nguyên văn logic đã bị gỡ khỏi eval/metrics.py ở DEC-0004. Nó
# sống ở đây, trong script so sánh, chứ không sống trong bộ metric — để không ai
# vô tình chấm điểm thật bằng nó nữa.

_LEGACY_THRESHOLD = 0.15


def _legacy_f1(a: str, b: str) -> float:
    a_toks, b_toks = _token_set(a), _token_set(b)
    if not a_toks or not b_toks:
        return 0.0
    common = len(a_toks & b_toks)
    if common == 0:
        return 0.0
    precision = common / len(a_toks)
    recall = common / len(b_toks)
    return 2 * precision * recall / (precision + recall)


def _legacy_first_hit_rank(records: list[dict], ground_truth: str, k: int) -> int | None:
    for rank, rec in enumerate(records[:k], start=1):
        if _legacy_f1(rec.get("text", ""), ground_truth) >= _LEGACY_THRESHOLD:
            return rank
    return None


def legacy_recall_at_k(items: list[dict], retrieved: list[list[dict]], k: int) -> float:
    hits = sum(
        1 for item, recs in zip(items, retrieved)
        if _legacy_first_hit_rank(recs, item.get("ground_truth", ""), k) is not None
    )
    return round(hits / len(items), 4) if items else 0.0


def legacy_mrr_at_k(items: list[dict], retrieved: list[list[dict]], k: int) -> float:
    total = 0.0
    for item, recs in zip(items, retrieved):
        rank = _legacy_first_hit_rank(recs, item.get("ground_truth", ""), k)
        if rank is not None:
            total += 1.0 / rank
    return round(total / len(items), 4) if items else 0.0


# ── Retrieval (một lần cho cả hai thước) ──────────────────────────────────────

def collect_records(
    questions: list[dict],
    query_cache_path: Path | None = None,
) -> list[list[dict]]:
    """Chạy retrieval một lần cho mỗi câu hỏi, trả bản ghi chunk theo thứ tự rank.

    `query_cache_path` trỏ tới file vector câu hỏi đã embed sẵn: khi có, không
    model nào được load và eval chạy được trên máy thiếu RAM (DEC-0006). Khi
    không có, đi đường bình thường qua get_embedder().
    """
    from eval.rag_comparison import _chunk_record, _init_rag_shared
    from src.config import get_settings as _settings
    from src.rag.retriever import build_hybrid_retriever, nodes_to_chunks

    # Các module eval khác gọi use_local_hf_cache(offline=True) lúc import, ghi đè
    # quyết định ở đầu file này. Đặt lại sau khi import xong.
    use_local_hf_cache(offline=get_settings().embedding_provider.strip().lower() != "hf_api")

    settings = _settings()
    embedder = None
    if query_cache_path:
        from eval.query_cache import QueryEmbeddingCache, make_cached_embedder

        cache = QueryEmbeddingCache.load(query_cache_path, expected_model=settings.embedding_model)
        missing = cache.covers([q["question"] for q in questions])
        if missing:
            raise SystemExit(
                f"Cache thiếu {len(missing)} câu hỏi, ví dụ: {missing[0][:60]!r}. "
                f"Sinh lại: python scripts/build_query_embedding_cache.py"
            )
        embedder = make_cached_embedder(cache)
        logger.info(
            "Dùng cache vector câu hỏi: {} câu, {} chiều, model {}",
            len(cache), cache.dim, cache.model,
        )

    index, _collection, all_nodes, _embedder = _init_rag_shared(embedder=embedder)
    retriever = build_hybrid_retriever(index, nodes=all_nodes, rerank=settings.enable_reranker)
    logger.info("Retriever: rerank={}", settings.enable_reranker)

    out: list[list[dict]] = []
    for i, q in enumerate(questions, 1):
        chunks = nodes_to_chunks(retriever.retrieve(q["question"]))
        out.append([_chunk_record(c) for c in chunks])
        logger.info("[{}/{}] {} — {} chunks", i, len(questions), q["id"], len(chunks))
    return out


# ── Báo cáo ───────────────────────────────────────────────────────────────────

def build_report(
    questions: list[dict],
    retrieved: list[list[dict]],
    embedding_source: str = "local model",
) -> dict[str, Any]:
    from src.config import get_settings

    settings = get_settings()
    ks = (1, 5, 8, 20)

    legacy = {f"recall@{k}": legacy_recall_at_k(questions, retrieved, k) for k in ks}
    legacy["mrr@8"] = legacy_mrr_at_k(questions, retrieved, 8)

    clause = {f"recall@{k}": recall_at_k(questions, retrieved, k) for k in ks}
    clause["mrr@8"] = mrr_at_k(questions, retrieved, 8)

    # Câu mà hai thước nói ngược nhau — phần đáng đọc nhất của báo cáo.
    disagreements = []
    for item, recs in zip(questions, retrieved):
        legacy_hit = _legacy_first_hit_rank(recs, item.get("ground_truth", ""), 8) is not None
        clause_hit = recall_at_k([item], [recs], 8)
        if clause_hit is None:
            continue
        if legacy_hit != bool(clause_hit):
            disagreements.append({
                "id": item["id"],
                "temporal_kind": item.get("temporal_kind", ""),
                "as_of_date": item.get("as_of_date", ""),
                "legacy_says_hit": legacy_hit,
                "clause_says_hit": bool(clause_hit),
                "gold_clause": item.get("source_clause", ""),
                "retrieved_clauses": [r["clause_uid"] for r in recs[:8]],
            })

    n_scored = sum(1 for q in questions if q.get("source_clause") or q.get("gold_clause_uids"))
    return {
        "meta": {
            "gold_set": str(DEFAULT_GOLD.relative_to(_REPO_ROOT)).replace("\\", "/"),
            "n_questions": len(questions),
            "n_scored_by_clause_metric": n_scored,
            "n_skipped_by_clause_metric": len(questions) - n_scored,
            "embedding_model": settings.embedding_model,
            "embedding_source": embedding_source,
            "enable_reranker": settings.enable_reranker,
            "reranker_model": settings.reranker_model if settings.enable_reranker else "",
            "llm_calls": 0,
        },
        "scoring": {"legacy_token_f1": legacy, "clause_uid": clause},
        "disagreements": disagreements,
    }


def print_report(report: dict) -> None:
    legacy = report["scoring"]["legacy_token_f1"]
    clause = report["scoring"]["clause_uid"]
    meta = report["meta"]

    print("\n" + "=" * 72)
    print(f"A/B HAI CÁCH CHẤM — {meta['n_questions']} câu, cùng một lần retrieval")
    print(f"embedding={meta['embedding_model']} · rerank={meta['enable_reranker']}")
    print("=" * 72)
    print(f"{'metric':<14}{'token-F1 ≥0.15':>18}{'clause_uid':>14}{'chênh':>12}")
    print("-" * 72)
    for key in ("recall@1", "recall@5", "recall@8", "recall@20", "mrr@8"):
        a, b = legacy.get(key), clause.get(key)
        delta = "—" if a is None or b is None else f"{b - a:+.4f}"
        a_txt = "—" if a is None else str(a)
        b_txt = "—" if b is None else str(b)
        print(f"{key:<14}{a_txt:>18}{b_txt:>14}{delta:>12}")

    print(f"\nchấm được: {meta['n_scored_by_clause_metric']}/{meta['n_questions']} câu"
          f" · bỏ qua: {meta['n_skipped_by_clause_metric']}")

    dis = report["disagreements"]
    print(f"\nCâu hai thước nói ngược nhau (top-8): {len(dis)}")
    for d in dis[:10]:
        verdict = "cũ TRÚNG / mới TRẬT" if d["legacy_says_hit"] else "cũ TRẬT / mới TRÚNG"
        print(f"  {d['id']:<10} {d['temporal_kind']:<16} @{d['as_of_date']}  {verdict}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="chỉ chạy N câu đầu")
    parser.add_argument(
        "--query-embeddings", type=Path, default=None,
        help="file vector câu hỏi đã embed sẵn (scripts/build_query_embedding_cache.py) — "
             "chạy được khi máy không load nổi model",
    )
    args = parser.parse_args()

    questions = json.loads(args.gold.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    retrieved = collect_records(questions, query_cache_path=args.query_embeddings)
    source = f"cache {args.query_embeddings.name}" if args.query_embeddings else "local model"
    report = build_report(questions, retrieved, embedding_source=source)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(report)
    print(f"Đã ghi báo cáo: {args.output}")


if __name__ == "__main__":
    main()
