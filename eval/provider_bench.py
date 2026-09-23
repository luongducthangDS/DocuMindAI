"""
So sánh Qdrant Cloud vs Chroma Cloud ở TẦNG VECTOR STORE — cùng 1151 vector, cùng query.

Chỉ đo thứ khác nhau giữa hai provider: chất lượng index ANN, độ đúng của filter
đẩy xuống (RetrievalContext — tenant/ACL/as_of_date), độ trễ, thời gian tải corpus.
BM25/reranker/LLM giống hệt nhau ở hai bên nên bị loại khỏi phép đo (chỉ thêm nhiễu).

  • ann_recall@k  — |top-k provider ∩ top-k vét cạn| / |top-k vét cạn|. Vét cạn = cosine
                    numpy trên embeddings Chroma local, cùng predicate filter → "exact".
  • recall@1/3/5/8, mrr — gold clause (dense-only, chỉ mode filtered = như production).
  • violation_rate — % kết quả trả về KHÔNG thoả filter (phải = 0).
  • empty_rate    — % query trả rỗng trong khi exact có kết quả.
  • p50/p95/p99   — ms/query đo từ máy chạy script (gồm mạng). Hai provider chạy XEN KẼ
                    từng query, thứ tự ngẫu nhiên, để trôi mạng ảnh hưởng đều hai bên.

Query embeddings đọc từ cache (reports/_eval_cache/query_embeddings.json) → hai bên
nhận CÙNG vector, không tốn quota Gemini. Tiền đề: hai cloud đã đồng bộ với Chroma local
(scripts/migrate_chroma_to_qdrant.py, scripts/copy_chroma_to_cloud.py) — script kiểm
số lượng + tập id trước khi đo, lệch thì dừng.

Chạy:
    python eval/provider_bench.py
    python eval/provider_bench.py --repeat 3 --mlflow provider-bench
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "eval"))

from loguru import logger  # noqa: E402

from temporal_eval import (  # noqa: E402
    RECALL_KS,
    _git,
    _percentile,
    gold_rank,
    has_gold_clause,
    use_query_embedding_cache,
)

K = 10
MODES = ("filtered", "unfiltered")


# ── Ground truth ──────────────────────────────────────────────────────────────

def matches(where: dict | None, meta: dict) -> bool:
    """Đánh giá `where` kiểu Chroma trên 1 metadata — đúng các toán tử to_where() phát ra.

    Không dùng RetrievalContext.allows(): exact phải áp CÙNG predicate đã gửi cho
    provider, nếu không ann_recall đo lệch giữa hai định nghĩa filter chứ không đo index.
    """
    if not where:
        return True
    if "$and" in where:
        return all(matches(w, meta) for w in where["$and"])
    ((key, cond),) = where.items()
    ((op, val),) = cond.items()
    v = meta.get(key)
    if op == "$in":
        return v in val
    if op == "$nin":
        return v not in val
    if op == "$lte":
        return v is not None and v <= val
    if op == "$gt":
        return v is not None and v > val
    raise ValueError(f"toán tử chưa hỗ trợ: {op}")


def exact_topk(emb: np.ndarray, metas: list[dict], ids: list[str], qvec: np.ndarray,
               where: dict | None, k: int = K) -> list[str]:
    """Top-k cosine vét cạn trên các chunk thoả `where`. `emb` đã chuẩn hoá theo hàng."""
    sims = emb @ (qvec / np.linalg.norm(qvec))
    allowed = [i for i, m in enumerate(metas) if matches(where, m)]
    allowed.sort(key=lambda i: -sims[i])
    return [ids[i] for i in allowed[:k]]


# ── Providers ─────────────────────────────────────────────────────────────────

class QdrantProvider:
    name = "qdrant"

    def __init__(self, local_ids: list[str]) -> None:
        from src.config import get_settings
        from src.rag.embedder import get_qdrant_client_and_collection

        self.client, self.coll = get_qdrant_client_and_collection(timeout=60)
        self.host = get_settings().qdrant_url
        # migrate_chroma_to_qdrant suy id Qdrant = uuid5(id Chroma) → map ngược lại
        self.to_cid = {str(uuid.uuid5(uuid.NAMESPACE_URL, c)): c for c in local_ids}

    def query(self, vec: list[float], k: int, where: dict | None) -> list[tuple[str, dict]]:
        from src.rag.vector_backend import where_to_qdrant_filter

        pts = self.client.query_points(
            collection_name=self.coll, query=vec, limit=k, with_payload=True,
            query_filter=where_to_qdrant_filter(where),
        ).points
        return [
            (self.to_cid.get(str(p.id), str(p.id)),
             {kk: vv for kk, vv in (p.payload or {}).items() if not kk.startswith("_")})
            for p in pts
        ]

    def all_ids(self) -> set[str]:
        """Tải toàn bộ corpus y như lúc API dựng BM25 (fetch_all_chunks)."""
        from src.rag.vector_backend import Backend, fetch_all_chunks

        rows = fetch_all_chunks(Backend(provider="qdrant", client=self.client, collection=self.coll))
        return {self.to_cid.get(str(r[0]), str(r[0])) for r in rows}


class ChromaCloudProvider:
    name = "chroma_cloud"

    def __init__(self, collection: str) -> None:
        from src.rag.chroma_cloud import API, ChromaCloud

        self.cc = ChromaCloud()
        self.cid = self.cc.collection_id(collection)
        self.host = API

    def query(self, vec: list[float], k: int, where: dict | None) -> list[tuple[str, dict]]:
        body = {"query_embeddings": [vec], "n_results": k, "include": ["metadatas"]}
        if where:
            body["where"] = where
        r = self.cc.call("POST", f"/{self.cid}/query", json=body)
        return list(zip(r["ids"][0], r["metadatas"][0]))

    def all_ids(self) -> set[str]:
        out, off = set(), 0
        while True:
            r = self.cc.call("POST", f"/{self.cid}/get", json={
                "limit": 100, "offset": off, "include": ["documents", "metadatas"],
            })
            out.update(r["ids"])
            if len(r["ids"]) < 100:
                return out
            off += 100


# ── Run ───────────────────────────────────────────────────────────────────────

def summarise(rows: list[dict], has_gold: dict[str, bool]) -> dict:
    lat = [ms for r in rows for ms in r["ms"]]
    agg = {
        "n": len(rows),
        f"ann_recall_at_{K}": float(np.mean([r["ann_recall"] for r in rows])),
        "violation_rate": sum(r["violations"] for r in rows) / max(1, sum(r["n_hits"] for r in rows)),
        "empty_rate": sum(1 for r in rows if r["n_hits"] == 0 and r["n_exact"]) / len(rows),
        "p50_ms": _percentile(lat, 50),
        "p95_ms": _percentile(lat, 95),
        "p99_ms": _percentile(lat, 99),
    }
    ranked = [r["gold_rank"] for r in rows if "gold_rank" in r and has_gold[r["id"]]]
    if ranked:
        for k in RECALL_KS:
            agg[f"recall_at_{k}"] = sum(1 for x in ranked if x and x <= k) / len(ranked)
        agg["mrr"] = sum(1 / x for x in ranked if x) / len(ranked)
    return agg


def run(gold: Path, repeat: int, seed: int) -> dict:
    import chromadb

    from src.config import get_settings
    from src.rag.context import RetrievalContext
    from src.rag.embedder import get_embedder

    s = get_settings()
    questions = json.loads(gold.read_text(encoding="utf-8"))
    questions = questions["questions"] if isinstance(questions, dict) else questions

    local = chromadb.PersistentClient(path=str(s.data_dir / "chroma_db")).get_collection(s.chroma_collection)
    got = local.get(include=["metadatas", "embeddings"])
    ids, metas = got["ids"], got["metadatas"]
    emb = np.asarray(got["embeddings"], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    logger.info("Local Chroma: {} chunks, dim {}", len(ids), emb.shape[1])

    embedder = get_embedder()
    qvecs = {q["id"]: embedder.get_query_embedding(q["question"]) for q in questions}

    providers = []
    meta: dict = {"providers": {}}
    for make in (lambda: QdrantProvider(ids), lambda: ChromaCloudProvider(s.chroma_collection)):
        t0 = time.perf_counter()
        p = make()
        init_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        remote = p.all_ids()
        load_ms = (time.perf_counter() - t0) * 1000
        if remote != set(ids):
            raise SystemExit(
                f"{p.name} lệch dữ liệu với Chroma local: thiếu {len(set(ids) - remote)}, "
                f"thừa {len(remote - set(ids))} — đồng bộ lại rồi chạy tiếp."
            )
        meta["providers"][p.name] = {"host": p.host, "count": len(remote),
                                     "init_ms": round(init_ms), "load_corpus_ms": round(load_ms)}
        providers.append(p)
        logger.info("{}: {} chunks khớp local, init {:.0f}ms, tải corpus {:.0f}ms",
                    p.name, len(remote), init_ms, load_ms)

    # Query đầu tiên sau khi dựng client — gần nhất với cold start ta đo được từ đây.
    first = questions[0]
    for p in providers:
        t0 = time.perf_counter()
        p.query(qvecs[first["id"]], K, None)
        meta["providers"][p.name]["first_query_ms"] = round((time.perf_counter() - t0) * 1000)
        for q in questions[1:4]:  # warm-up, không tính
            p.query(qvecs[q["id"]], K, None)

    by_id = dict(zip(ids, metas))
    rng = random.Random(seed)
    rows = {(p.name, m): [] for p in providers for m in MODES}
    exact_rows = []
    for i, q in enumerate(questions, 1):
        where = RetrievalContext(as_of_date=q["as_of_date"]).to_where()
        for mode in MODES:
            w = where if mode == "filtered" else None
            truth = exact_topk(emb, metas, ids, np.asarray(qvecs[q["id"]]), w)
            if mode == "filtered":
                exact_rows.append({"id": q["id"], "ann_recall": 1.0, "violations": 0,
                                   "n_hits": len(truth), "n_exact": len(truth), "ms": [],
                                   "gold_rank": gold_rank(
                                       [SimpleNamespace(metadata=by_id[c]) for c in truth], q)})
            order = providers[:]
            for rep in range(repeat):
                rng.shuffle(order)
                for p in order:
                    t0 = time.perf_counter()
                    hits = p.query(qvecs[q["id"]], K, w)
                    ms = (time.perf_counter() - t0) * 1000
                    if rep:
                        rows[(p.name, mode)][-1]["ms"].append(ms)
                        continue
                    row = {
                        "id": q["id"],
                        "ann_recall": (len({h[0] for h in hits} & set(truth)) / len(truth)) if truth else 1.0,
                        "violations": sum(1 for _, m in hits if not matches(w, m)),
                        "n_hits": len(hits), "n_exact": len(truth), "ms": [ms],
                    }
                    if mode == "filtered":
                        row["gold_rank"] = gold_rank([SimpleNamespace(metadata=m) for _, m in hits], q)
                    rows[(p.name, mode)].append(row)
        if i % 25 == 0:
            logger.info("[{}/{}]", i, len(questions))

    has_gold = {q["id"]: has_gold_clause(q) for q in questions}
    summary = {"exact": {"filtered": summarise(exact_rows, has_gold)}}
    for (name, mode), rs in rows.items():
        summary.setdefault(name, {})[mode] = summarise(rs, has_gold)

    meta.update({
        "gold_set": str(gold), "n_questions": len(questions), "k": K, "repeat": repeat,
        "seed": seed, "embedding_model": s.embedding_model, "n_chunks": len(ids),
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "note": "latency đo từ máy chạy script, gồm RTT mạng tới region của từng cluster",
    })
    return {"meta": meta, "summary": summary,
            "rows": {f"{n}/{m}": rs for (n, m), rs in rows.items()}}


def print_report(report: dict) -> None:
    cols = [f"ann_recall_at_{K}", "recall_at_1", "recall_at_5", "mrr",
            "violation_rate", "empty_rate", "p50_ms", "p95_ms", "p99_ms"]
    print(f"\n{'provider/mode':26}" + "".join(f"{c:>15}" for c in cols))
    for name, modes in report["summary"].items():
        for mode, agg in modes.items():
            cells = []
            for c in cols:
                v = agg.get(c)
                cells.append("—" if v is None else f"{v:.1f}" if c.endswith("_ms") else f"{v:.3f}")
            print(f"{name + '/' + mode:26}" + "".join(f"{x:>15}" for x in cells))
    for name, p in report["meta"]["providers"].items():
        print(f"{name}: {p}")


def log_to_mlflow(report: dict, output: Path, run_name: str) -> None:
    import mlflow

    mlflow.set_experiment("documind-provider-bench")
    with mlflow.start_run(run_name=run_name):
        m = report["meta"]
        mlflow.set_tags({"git_sha": m["git_sha"], "gold_set": m["gold_set"]})
        mlflow.log_params({k: m[k] for k in ("n_questions", "k", "repeat", "embedding_model", "n_chunks")})
        for name, p in m["providers"].items():
            mlflow.log_param(f"{name}_host", p["host"])
            mlflow.log_metrics({f"{name}/{k}": float(v) for k, v in p.items() if isinstance(v, (int, float))})
        for name, modes in report["summary"].items():
            for mode, agg in modes.items():
                mlflow.log_metrics({f"{name}/{mode}/{k}": float(v) for k, v in agg.items()
                                    if isinstance(v, (int, float))})
        mlflow.log_artifact(str(output))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=_REPO_ROOT / "data/eval/legal_qa_200.json")
    ap.add_argument("--output", type=Path, default=_REPO_ROOT / "reports/provider_bench.json")
    ap.add_argument("--embed-cache", type=Path,
                    default=_REPO_ROOT / "reports/_eval_cache/query_embeddings.json")
    ap.add_argument("--repeat", type=int, default=1, help="số lần lặp mỗi query (chỉ để lấy mẫu latency)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--mlflow", metavar="RUN_NAME")
    args = ap.parse_args()

    use_query_embedding_cache(args.embed_cache)
    report = run(args.gold, args.repeat, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(report)
    logger.info("Đã ghi {}", args.output)
    if args.mlflow:
        log_to_mlflow(report, args.output, args.mlflow)


if __name__ == "__main__":
    main()
