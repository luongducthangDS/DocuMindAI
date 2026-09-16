"""
A/B harness: đo tác động của model embedding lên retrieval, giữ mọi thứ khác cố định.

Biến duy nhất thay đổi giữa các arm là model embedding. Chunking, corpus, BM25,
reranker, gold set — tất cả giữ nguyên:
  • Chunk text + metadata đọc thẳng từ collection ChromaDB đang chạy (không re-chunk,
    nên chỉ số so được với reports/temporal_eval.json).
  • BM25 dùng llama_index BM25Retriever y như production (hằng số giữa các arm).
  • Reranker dùng cùng model production (settings.reranker_model).

Ba tầng đo, từ cô lập nhất tới giống production nhất:
  ① dense@20 / dense@8 — thuần embedding, không BM25 không rerank. Đây là tầng
     trả lời câu hỏi "model embedding có phải nút thắt không".
  ② hybrid@20 — sau RRF fusion với BM25. BM25 có thể che khuyết điểm của dense.
  ③ final@8  — sau rerank. Đây là con số đối chiếu trực tiếp với context_gold
     trong temporal_eval.json (nhánh no_temporal).

Chấm điểm tái dùng score_context() của eval/temporal_eval.py — thuần metadata,
không gọi LLM, không tốn quota.

Chạy:
    python eval/embedding_ab.py --models current AITeamVN/Vietnamese_Embedding
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Phải set trước khi bất cứ thứ gì import sentence_transformers: HF_HOME hệ thống
# trỏ vào G:\ (Google Drive, thường offline) — xem CLAUDE.md.
_LOCAL_HF = _REPO_ROOT / "data" / "hf_cache"
for _k in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
    os.environ[_k] = str(_LOCAL_HF)
os.environ["HF_HUB_CACHE"] = str(_LOCAL_HF / "hub")
# KHÔNG set HF_HUB_OFFLINE: arm ứng viên cần tải model mới từ Hub lần đầu.

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

from eval.temporal_eval import score_context  # noqa: E402

# temporal_eval bật HF_HUB_OFFLINE=1 ở module level (nó chỉ dùng model đã cache).
# Harness này thì cần tải model ứng viên về lần đầu, nên gỡ lại sau khi import.
os.environ.pop("HF_HUB_OFFLINE", None)

CURRENT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_GOLD = _REPO_ROOT / "data" / "eval" / "temporal_questions.json"
DEFAULT_OUT = _REPO_ROOT / "reports" / "embedding_ab.json"

CANDIDATE_POOL = 20  # khớp top_k của build_hybrid_retriever
FINAL_TOP_N = 8      # khớp top_n của reranker production


# ── Corpus ────────────────────────────────────────────────────────────────────

def load_corpus() -> tuple[list[str], list[dict]]:
    """Đọc chunk text + metadata từ vector store đang chạy. Không re-chunk."""
    from src.rag.embedder import get_chroma_collection

    _client, collection = get_chroma_collection()
    result = collection.get(include=["documents", "metadatas"])
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    texts, metadatas = [], []
    for d, m in zip(docs, metas):
        if d:
            texts.append(d)
            metadatas.append({k: v for k, v in (m or {}).items() if not k.startswith("_")})
    logger.info("Corpus: {} chunks", len(texts))
    return texts, metadatas


# ── Dense retrieval (numpy, không phụ thuộc vector store) ─────────────────────

def _load_st_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: {}", model_name)
    t0 = time.time()
    model = SentenceTransformer(model_name, trust_remote_code=False, device="cpu")
    logger.info(
        "Loaded in {:.1f}s — dim={}",
        time.time() - t0,
        model.get_sentence_embedding_dimension(),
    )
    return model


def dense_rankings(
    model_name: str,
    texts: list[str],
    questions: list[str],
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Trả (ma trận [n_questions × n_chunks] chỉ số chunk giảm dần theo cosine, stats)."""
    model = _load_st_model(model_name)

    t0 = time.time()
    doc_vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,  # bar của tqdm làm ngập log khi chạy nền
        convert_to_numpy=True,
    )
    index_s = time.time() - t0
    logger.info("Đã embed {} chunks trong {:.0f}s", len(texts), index_s)

    t0 = time.time()
    q_vecs = model.encode(
        questions,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    query_ms = (time.time() - t0) * 1000 / max(len(questions), 1)

    sims = q_vecs @ doc_vecs.T                      # cosine (đã normalize)
    order = np.argsort(-sims, axis=1)               # giảm dần

    stats = {
        "dim": int(doc_vecs.shape[1]),
        "index_seconds": round(index_s, 1),
        "query_ms_per_question": round(query_ms, 1),
        "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
    }
    # Giải phóng ngay: model tiếp theo (hoặc reranker) cần tới 2.3GB, giữ hai
    # model cùng lúc là OOM trên máy 16GB. del không đủ — torch chỉ trả RAM sau gc.
    del model
    gc.collect()
    return order, stats


# ── BM25 (hằng số giữa các arm) ───────────────────────────────────────────────

def bm25_rankings(texts: list[str], metadatas: list[dict], questions: list[str]) -> list[list[int]]:
    """Top-CANDIDATE_POOL chỉ số chunk cho mỗi câu hỏi, dùng đúng BM25 của production."""
    from llama_index.core.schema import TextNode
    from llama_index.retrievers.bm25 import BM25Retriever

    nodes = [TextNode(text=t, metadata=m) for t, m in zip(texts, metadatas)]
    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=CANDIDATE_POOL)

    # BM25Retriever trả node, không trả chỉ số — map ngược bằng text.
    # Chunk trùng text tuyệt đối sẽ map về bản đầu tiên; điều đó giống nhau ở mọi
    # arm nên không làm lệch so sánh.
    text_to_idx: dict[str, int] = {}
    for i, t in enumerate(texts):
        text_to_idx.setdefault(t, i)

    out = []
    for q in questions:
        hits = retriever.retrieve(q)
        idxs: list[int] = []
        for h in hits:
            i = text_to_idx.get(h.node.text or h.node.get_content())
            if i is not None and i not in idxs:
                idxs.append(i)
        out.append(idxs)
    return out


def rrf_fuse(list_a: list[int], list_b: list[int], k: int = 60, top_n: int = CANDIDATE_POOL) -> list[int]:
    """Reciprocal Rank Fusion — cùng công thức QueryFusionRetriever dùng."""
    scores: dict[int, float] = {}
    for lst in (list_a, list_b):
        for rank, idx in enumerate(lst):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return [i for i, _ in sorted(scores.items(), key=lambda kv: -kv[1])][:top_n]


# ── Reranker (hằng số giữa các arm) ───────────────────────────────────────────

def _load_reranker():
    from sentence_transformers import CrossEncoder

    from src.config import get_settings

    name = get_settings().reranker_model
    logger.info("Loading reranker: {}", name)
    return CrossEncoder(name, max_length=512, device="cpu"), name


def rerank(cross_encoder, question: str, texts: list[str], cand: list[int], top_n: int) -> list[int]:
    if not cand:
        return []
    pairs = [(question, texts[i]) for i in cand]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(cand, scores), key=lambda kv: -float(kv[1]))
    return [i for i, _ in ranked[:top_n]]


# ── Chấm điểm ─────────────────────────────────────────────────────────────────

def _chunks_from(idxs: list[int], texts: list[str], metadatas: list[dict]) -> list:
    from src.rag.retriever import RetrievedChunk

    return [RetrievedChunk(text=texts[i], score=0.0, metadata=metadatas[i]) for i in idxs]


def gold_rank(order_row, texts: list[str], metadatas: list[dict], q: dict, limit: int = 200) -> int | None:
    """Vị trí (1-based) của chunk gold đầu tiên trong xếp hạng dense. None nếu ngoài `limit`."""
    for pos, idx in enumerate(order_row[:limit], start=1):
        if score_context(_chunks_from([int(idx)], texts, metadatas), q)["gold"]:
            return pos
    return None


def _cache_path(cache_dir: Path, model_name: str, n_texts: int, n_questions: int) -> Path:
    slug = model_name.replace("/", "__").replace("\\", "__")
    return cache_dir / f"{slug}__{n_texts}c_{n_questions}q.npz"


def embed_arm(
    model_name: str,
    texts: list[str],
    questions: list[dict],
    batch_size: int,
    cache_dir: Path,
) -> dict[str, Any]:
    """Pha 1 — chỉ embedding. Model được giải phóng trước khi hàm trả về, nên các
    arm và reranker không bao giờ nằm trong RAM cùng lúc.

    Kết quả được cache ra đĩa: embed 1146 chunks bằng model 568M trên CPU mất
    ~20 phút, không nên trả lại giá đó mỗi lần chạy lại pha chấm điểm.
    """
    cache = _cache_path(cache_dir, model_name, len(texts), len(questions))
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        logger.info("Dùng cache embedding: {}", cache.name)
        return {"model": model_name, "stats": json.loads(str(data["stats"])), "order": data["order"]}

    order, stats = dense_rankings(model_name, texts, [q["question"] for q in questions], batch_size)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, order=order.astype(np.int32), stats=json.dumps(stats))
    logger.info("Đã cache: {}", cache.name)
    return {"model": model_name, "stats": stats, "order": order}


def score_arm(
    arm: dict[str, Any],
    texts: list[str],
    metadatas: list[dict],
    questions: list[dict],
    bm25_lists: list[list[int]],
    cross_encoder,
) -> dict[str, Any]:
    """Pha 2 — chấm điểm. Reranker dùng chung cho mọi arm (hằng số của thí nghiệm)."""
    order = arm["order"]
    rows = []
    for qi, q in enumerate(questions):
        dense20 = [int(i) for i in order[qi][:CANDIDATE_POOL]]
        dense8 = dense20[:FINAL_TOP_N]
        hybrid20 = rrf_fuse(dense20, bm25_lists[qi])
        final8 = rerank(cross_encoder, q["question"], texts, hybrid20, FINAL_TOP_N)

        rows.append({
            "id": q["id"],
            "temporal_kind": q["temporal_kind"],
            "dense_at_8": score_context(_chunks_from(dense8, texts, metadatas), q)["gold"],
            "dense_at_20": score_context(_chunks_from(dense20, texts, metadatas), q)["gold"],
            "hybrid_at_20": score_context(_chunks_from(hybrid20, texts, metadatas), q)["gold"],
            "final_at_8": score_context(_chunks_from(final8, texts, metadatas), q)["gold"],
            "gold_rank_dense": gold_rank(order[qi], texts, metadatas, q),
        })

    return {"model": arm["model"], "stats": arm["stats"], "rows": rows}


def summarise(arm: dict, questions: list[dict]) -> dict[str, Any]:
    """out_of_range bị loại: gold của nó là 'context rỗng', do temporal filter
    quyết định chứ không phải embedding — giữ lại sẽ làm nhiễu chỉ số recall."""
    kinds = {q["id"]: q["temporal_kind"] for q in questions}
    rows = [r for r in arm["rows"] if kinds[r["id"]] != "out_of_range"]
    n = len(rows)
    ranks = [r["gold_rank_dense"] for r in rows if r["gold_rank_dense"]]

    return {
        "n_scored": n,
        "n_excluded_out_of_range": len(arm["rows"]) - n,
        "dense_recall_at_8": round(sum(r["dense_at_8"] for r in rows) / n, 4),
        "dense_recall_at_20": round(sum(r["dense_at_20"] for r in rows) / n, 4),
        "hybrid_recall_at_20": round(sum(r["hybrid_at_20"] for r in rows) / n, 4),
        "final_gold_at_8": round(sum(r["final_at_8"] for r in rows) / n, 4),
        "dense_mrr": round(sum(1.0 / r for r in ranks) / n, 4),
        "gold_rank_median": int(np.median(ranks)) if ranks else None,
        "gold_never_found": n - len(ranks),
    }


# ── Báo cáo ───────────────────────────────────────────────────────────────────

_METRICS = [
    ("dense_recall_at_8", "dense@8   (thuần embedding, top-8)"),
    ("dense_recall_at_20", "dense@20  (thuần embedding, pool)"),
    ("hybrid_recall_at_20", "hybrid@20 (+BM25 RRF)"),
    ("final_gold_at_8", "final@8   (+rerank — giống production)"),
    ("dense_mrr", "MRR dense"),
]


def print_report(report: dict) -> None:
    arms = report["arms"]
    names = [a["model"].split("/")[-1] for a in arms]
    base = report["summaries"][0]

    print("\n" + "=" * 78)
    print(f"A/B EMBEDDING — {report['meta']['n_questions']} câu gold, "
          f"{report['meta']['corpus_chunks']} chunks, reranker={report['meta']['reranker']}")
    print("=" * 78)
    width = 38
    print(f"{'Chỉ số':<{width}}" + "".join(f"{n[:22]:>24}" for n in names))
    print("-" * 78)
    for key, label in _METRICS:
        line = f"{label:<{width}}"
        for i, s in enumerate(report["summaries"]):
            val = s[key]
            cell = f"{val:.3f}"
            if i > 0:
                delta = val - base[key]
                cell += f" ({delta:+.3f})"
            line += f"{cell:>24}"
        print(line)
    print("-" * 78)
    for key, label in (("gold_rank_median", "Hạng gold trung vị (dense)"),
                       ("gold_never_found", "Gold không thấy trong top-200")):
        line = f"{label:<{width}}"
        for s in report["summaries"]:
            line += f"{str(s[key]):>24}"
        print(line)
    print("-" * 78)
    for key, label in (("dim", "Chiều vector"), ("params_millions", "Tham số (triệu)"),
                       ("index_seconds", "Thời gian index corpus (s)"),
                       ("query_ms_per_question", "Embed 1 truy vấn (ms)")):
        line = f"{label:<{width}}"
        for a in arms:
            line += f"{str(a['stats'][key]):>24}"
        print(line)
    print("=" * 78 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B model embedding trên gold set retrieval")
    parser.add_argument("--models", nargs="+", default=["current", "AITeamVN/Vietnamese_Embedding"],
                        help="'current' = model đang index corpus")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0, help="chỉ chạy N câu đầu (smoke test)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path, default=_REPO_ROOT / "reports" / "_ab_cache",
                        help="nơi cache xếp hạng dense để không phải embed lại")
    parser.add_argument("--embed-only", action="store_true",
                        help="chỉ chạy pha embedding rồi thoát — pha chấm chạy process riêng "
                             "để reranker có RAM sạch")
    args = parser.parse_args()

    models = [CURRENT_MODEL if m == "current" else m for m in args.models]
    questions = json.loads(args.gold.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    texts, metadatas = load_corpus()
    if not texts:
        raise SystemExit("Corpus rỗng — chạy ingest trước khi A/B.")

    q_texts = [q["question"] for q in questions]
    logger.info("BM25 (hằng số giữa các arm)...")
    bm25_lists = bm25_rankings(texts, metadatas, q_texts)

    # Pha 1: embed từng arm, giải phóng model ngay sau mỗi arm (có cache trên đĩa).
    embedded = []
    for m in models:
        logger.info("─── EMBED: {} ───", m)
        embedded.append(embed_arm(m, texts, questions, args.batch_size, args.cache_dir))

    if args.embed_only:
        logger.info("--embed-only: đã cache xong, thoát trước khi nạp reranker.")
        return

    # Pha 2: chỉ tới đây mới nạp reranker (2.2GB) — không arm nào còn giữ RAM.
    cross_encoder, reranker_name = _load_reranker()
    arms, summaries = [], []
    for e in embedded:
        logger.info("─── SCORE: {} ───", e["model"])
        arm = score_arm(e, texts, metadatas, questions, bm25_lists, cross_encoder)
        arms.append(arm)
        summaries.append(summarise(arm, questions))

    report = {
        "meta": {
            "gold_set": str(args.gold.relative_to(_REPO_ROOT)).replace("\\", "/"),
            "n_questions": len(questions),
            "corpus_chunks": len(texts),
            "candidate_pool": CANDIDATE_POOL,
            "final_top_n": FINAL_TOP_N,
            "reranker": reranker_name,
            "note": "Chunking/corpus/BM25/reranker giữ nguyên; biến duy nhất là model embedding.",
        },
        "arms": arms,
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(report)
    logger.info("Đã ghi {}", args.output)


if __name__ == "__main__":
    main()
