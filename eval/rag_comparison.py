"""
RAG Strategy Comparison Benchmark — DocuMind AI
=================================================
Compares three retrieval strategies on the same test set:

  ① Naive RAG    — vector similarity only (ChromaDB top-5, no rerank)
  ② Hybrid RAG   — dense + BM25 RRF + cross-encoder rerank (no agent)
  ③ Agentic RAG  — hybrid + LangGraph intent routing + tool use

Metrics per strategy:
  • RAGAS: faithfulness, answer_relevancy, context_recall, context_precision
  • Latency: mean, p50, p95 (ms)
  • Retrieval: avg chunks returned, avg chunk score

Usage:
  # Quick smoke test (first 5 questions only)
  python eval/rag_comparison.py --limit 5 --output reports/benchmark_quick.json

  # Full evaluation
  python eval/rag_comparison.py \\
    --test-set data/eval/test_questions.json \\
    --output reports/benchmark_results.json

Requirements (eval-only, not in main requirements.txt):
  pip install ragas==0.1.21 datasets>=2.14.0

LLM for RAGAS judge:
  - Uses GROQ_API_KEY (primary) or GOOGLE_API_KEY (fallback) from .env
  - RAGAS needs an LLM to score faithfulness & answer_relevancy
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

from loguru import logger

import src.logger  # noqa: F401 — init loguru sinks
from src.config import get_settings


# ── RAG initialisation helpers ────────────────────────────────────────────────

def _init_rag_shared() -> tuple:
    """
    Bootstrap the shared RAG components used by all three strategies.
    Returns (index, chroma_collection, all_nodes, embedder).
    """
    from llama_index.core import Settings as LlamaSettings, VectorStoreIndex
    from llama_index.core import StorageContext
    from llama_index.core.schema import TextNode
    from llama_index.vector_stores.chroma import ChromaVectorStore

    from src.rag.embedder import get_chroma_collection, get_embedder

    embedder = get_embedder()
    LlamaSettings.embed_model = embedder
    LlamaSettings.llm = None

    chroma_client, collection = get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_ctx,
        embed_model=embedder,
    )

    # Load all nodes for BM25 corpus
    result = collection.get(include=["documents", "metadatas"])
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    all_nodes = [
        TextNode(text=d, metadata=m or {})
        for d, m in zip(docs, metas) if d
    ]
    logger.info("Shared RAG init: {} nodes in corpus", len(all_nodes))
    return index, collection, all_nodes, embedder


# ── Strategy ①: Naive RAG ────────────────────────────────────────────────────

class NaiveRAG:
    """
    Simplest RAG: direct ChromaDB vector similarity search, top-5, no rerank.
    No BM25, no hybrid fusion, no agent routing.
    This is the baseline all other strategies are compared against.
    """

    name = "naive_rag"

    def __init__(self, collection, embedder):
        self._collection = collection
        self._embedder = embedder

    def retrieve_and_answer(self, question: str) -> tuple[str, list[str]]:
        # Embed query using same model as corpus
        query_vec = self._embedder.get_query_embedding(question)

        # Direct vector search in ChromaDB
        results = self._collection.query(
            query_embeddings=[query_vec],
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        # Build context string from raw docs (no chunking niceties)
        contexts = [d for d in docs if d]

        if not contexts:
            return "Tôi không tìm thấy thông tin liên quan.", []

        # Simple LLM call (Groq direct, no agent routing)
        from src.rag.retriever import RetrievedChunk
        from src.rag.generator import generate_answer

        chunks = [
            RetrievedChunk(
                text=doc,
                score=1.0 - float(results["distances"][0][i]) if results.get("distances") else 0.5,
                metadata=meta or {},
            )
            for i, (doc, meta) in enumerate(zip(docs, metas)) if doc
        ]
        result = generate_answer(question, chunks)
        return result["answer"], contexts


# ── Strategy ②: Hybrid RAG ───────────────────────────────────────────────────

class HybridRAG:
    """
    Production hybrid retriever: dense vector + BM25 RRF + cross-encoder rerank.
    Same retrieval stack as production, but WITHOUT LangGraph agent routing.
    """

    name = "hybrid_rag"

    def __init__(self, index, all_nodes):
        from src.rag.retriever import build_hybrid_retriever
        self._retriever = build_hybrid_retriever(index, nodes=all_nodes, rerank=True)

    def retrieve_and_answer(self, question: str) -> tuple[str, list[str]]:
        from src.rag.generator import generate_answer
        from src.rag.retriever import nodes_to_chunks

        # Hybrid retrieval (sync path — safe outside async context)
        nodes = self._retriever.retrieve(question)
        chunks = nodes_to_chunks(nodes)
        contexts = [c.text for c in chunks]

        result = generate_answer(question, chunks)
        return result["answer"], contexts


# ── Strategy ③: Agentic RAG ──────────────────────────────────────────────────

class AgenticRAG:
    """
    Full LangGraph agent: intent routing → [simple_qa | compare | summarize | report]
    Uses hybrid retrieval + specialised tool calls per intent.
    """

    name = "agentic_rag"

    def __init__(self, index, all_nodes):
        # Pre-wire retriever into module globals (same as production startup)
        from src.rag.retriever import build_hybrid_retriever
        import src.rag.retriever as r_module
        from src.agent.tools import configure_tools

        retriever = build_hybrid_retriever(index, nodes=all_nodes, rerank=True)
        r_module._active_index = index
        r_module._active_retriever = retriever
        configure_tools(retriever, index)

    def retrieve_and_answer(self, question: str) -> tuple[str, list[str]]:
        from src.agent.graph import run_agent
        import src.rag.retriever as r_module
        from src.rag.retriever import nodes_to_chunks

        # Run full agent (async → sync bridge)
        result = asyncio.get_event_loop().run_until_complete(
            run_agent(query=question, session_id="eval_bench")
        )
        answer = result.get("answer", "")
        chunks = result.get("retrieved_chunks", [])

        # chunks may already be RetrievedChunk objects or raw nodes
        if chunks and hasattr(chunks[0], "text"):
            contexts = [c.text for c in chunks]
        elif chunks and hasattr(chunks[0], "node"):
            contexts = [c.node.text for c in chunks]
        else:
            contexts = []

        return answer, contexts


# ── RAGAS evaluation ──────────────────────────────────────────────────────────

def _configure_ragas_llm():
    """
    Configure RAGAS to use Groq or Gemini as the judge LLM.
    RAGAS 0.1.x uses a global `ragas.llm` setting.
    """
    try:
        import ragas
        from langchain_groq import ChatGroq
        from langchain_google_genai import ChatGoogleGenerativeAI

        settings = get_settings()

        if settings.groq_api_key:
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=settings.groq_api_key,
                temperature=0,
            )
            logger.info("RAGAS judge LLM: Groq llama-3.3-70b")
        elif settings.google_api_key:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=settings.google_api_key,
                temperature=0,
            )
            logger.info("RAGAS judge LLM: Gemini 1.5 Flash")
        else:
            raise RuntimeError("No LLM API key configured for RAGAS judge")

        # RAGAS 0.1.x: override global LLM
        from ragas.llms import LangchainLLMWrapper
        ragas.llm = LangchainLLMWrapper(llm)

        # Also override embeddings to avoid OpenAI dependency
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        hf_emb = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        ragas.embeddings = LangchainEmbeddingsWrapper(hf_emb)

    except Exception as exc:
        logger.warning("RAGAS LLM config failed (will use OpenAI default if key set): {}", exc)


def run_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict[str, float]:
    """
    Run RAGAS on a (question, answer, contexts, ground_truth) dataset.
    Returns metric dict. Falls back to empty dict if RAGAS is not installed.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        logger.error(
            "RAGAS not installed. Run: pip install 'ragas==0.1.21' datasets>=2.14.0"
        )
        return {}

    # Guard: filter out empty context rows (RAGAS will error on them)
    filtered = [
        (q, a, c, g)
        for q, a, c, g in zip(questions, answers, contexts, ground_truths)
        if c  # at least one context chunk
    ]
    if not filtered:
        logger.warning("All test items had empty contexts — RAGAS skipped")
        return {}

    qs, ans, ctx, gts = zip(*filtered)

    dataset = Dataset.from_dict({
        "question": list(qs),
        "answer": list(ans),
        "contexts": list(ctx),
        "ground_truth": list(gts),
    })

    logger.info("Running RAGAS on {} items...", len(filtered))
    try:
        result = ragas_evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        )
        return {k: round(float(v), 4) for k, v in result.to_pandas().mean(numeric_only=True).items()}
    except Exception as exc:
        logger.error("RAGAS evaluate failed: {}", exc)
        return {}


# ── Single-strategy runner ────────────────────────────────────────────────────

def run_strategy(
    strategy,
    test_items: list[dict],
) -> dict[str, Any]:
    """
    Run one RAG strategy on all test items.
    Collects latencies, answers, contexts; then runs RAGAS.
    """
    latencies_ms: list[float] = []
    questions, answers, contexts_list, ground_truths = [], [], [], []

    logger.info("▶  Running strategy: {}", strategy.name)

    for i, item in enumerate(test_items, 1):
        q = item["question"]
        gt = item.get("ground_truth", "")

        t0 = time.perf_counter()
        try:
            answer, ctx = strategy.retrieve_and_answer(q)
        except Exception as exc:
            logger.warning("[{}/{}] {} failed: {}", i, len(test_items), strategy.name, exc)
            answer, ctx = "", []

        latency = (time.perf_counter() - t0) * 1000
        latencies_ms.append(latency)

        questions.append(q)
        answers.append(answer)
        contexts_list.append(ctx)
        ground_truths.append(gt)

        logger.info(
            "  [{}/{}] {:.0f}ms | Q: {}",
            i, len(test_items), latency, q[:55]
        )

    # Latency stats
    lat_sorted = sorted(latencies_ms)
    p95_idx = int(len(lat_sorted) * 0.95)
    latency_stats = {
        "mean_ms": round(statistics.mean(latencies_ms), 1),
        "median_ms": round(statistics.median(latencies_ms), 1),
        "p95_ms": round(lat_sorted[min(p95_idx, len(lat_sorted) - 1)], 1),
        "min_ms": round(min(latencies_ms), 1),
        "max_ms": round(max(latencies_ms), 1),
    }

    # RAGAS metrics
    ragas_metrics = run_ragas(questions, answers, contexts_list, ground_truths)

    # Retrieval stats
    chunk_counts = [len(c) for c in contexts_list]
    retrieval_stats = {
        "avg_chunks": round(statistics.mean(chunk_counts), 1) if chunk_counts else 0,
        "items_with_context": sum(1 for c in contexts_list if c),
        "total_items": len(test_items),
    }

    return {
        "strategy": strategy.name,
        "latency": latency_stats,
        "ragas": ragas_metrics,
        "retrieval": retrieval_stats,
        "n_items": len(test_items),
    }


# ── Report ────────────────────────────────────────────────────────────────────

def save_report(results: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": get_settings().embedding_model,
        "corpus_chunks": None,  # filled below
        "results": {r["strategy"]: r for r in results},
        "summary": _build_summary(results),
    }

    # Try to get corpus chunk count
    try:
        from src.rag.embedder import get_chroma_collection
        _, col = get_chroma_collection()
        report["corpus_chunks"] = col.count()
    except Exception:
        pass

    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success("Benchmark report saved → {}", output)


def _build_summary(results: list[dict]) -> dict:
    """Compute deltas: hybrid vs naive, agentic vs hybrid."""
    by_name = {r["strategy"]: r for r in results}
    naive = by_name.get("naive_rag", {})
    hybrid = by_name.get("hybrid_rag", {})
    agentic = by_name.get("agentic_rag", {})

    def pct_delta(a_val, b_val):
        if not a_val or not b_val:
            return None
        return round((b_val - a_val) / a_val * 100, 1)

    def ragas_delta(a: dict, b: dict, key: str):
        return pct_delta(
            a.get("ragas", {}).get(key),
            b.get("ragas", {}).get(key),
        )

    return {
        "hybrid_vs_naive": {
            "faithfulness_delta_pct": ragas_delta(naive, hybrid, "faithfulness"),
            "answer_relevancy_delta_pct": ragas_delta(naive, hybrid, "answer_relevancy"),
            "context_recall_delta_pct": ragas_delta(naive, hybrid, "context_recall"),
            "latency_overhead_pct": pct_delta(
                naive.get("latency", {}).get("mean_ms"),
                hybrid.get("latency", {}).get("mean_ms"),
            ),
        },
        "agentic_vs_hybrid": {
            "faithfulness_delta_pct": ragas_delta(hybrid, agentic, "faithfulness"),
            "answer_relevancy_delta_pct": ragas_delta(hybrid, agentic, "answer_relevancy"),
            "context_recall_delta_pct": ragas_delta(hybrid, agentic, "context_recall"),
            "latency_overhead_pct": pct_delta(
                hybrid.get("latency", {}).get("mean_ms"),
                agentic.get("latency", {}).get("mean_ms"),
            ),
        },
    }


def print_table(results: list[dict]) -> None:
    """Print comparison table to stdout."""
    metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

    print("\n" + "=" * 72)
    print("  DocuMind AI — RAG Strategy Benchmark")
    print("=" * 72)
    print(f"  {'Metric':<28} {'Naive RAG':>12} {'Hybrid RAG':>12} {'Agentic RAG':>12}")
    print("-" * 72)

    by_name = {r["strategy"]: r for r in results}
    for m in metrics:
        row = f"  {m:<28}"
        for strat in ["naive_rag", "hybrid_rag", "agentic_rag"]:
            val = by_name.get(strat, {}).get("ragas", {}).get(m)
            row += f"  {val:>10.4f}" if val is not None else f"  {'N/A':>10}"
        print(row)

    print("-" * 72)
    print(f"  {'Avg Latency (ms)':<28}", end="")
    for strat in ["naive_rag", "hybrid_rag", "agentic_rag"]:
        val = by_name.get(strat, {}).get("latency", {}).get("mean_ms")
        print(f"  {val:>10.0f}" if val else f"  {'N/A':>10}", end="")
    print()

    print(f"  {'P95 Latency (ms)':<28}", end="")
    for strat in ["naive_rag", "hybrid_rag", "agentic_rag"]:
        val = by_name.get(strat, {}).get("latency", {}).get("p95_ms")
        print(f"  {val:>10.0f}" if val else f"  {'N/A':>10}", end="")
    print()
    print("=" * 72 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DocuMind AI — RAG Strategy Comparison Benchmark"
    )
    parser.add_argument("--test-set", type=Path, default=Path("data/eval/test_questions.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark_results.json"))
    parser.add_argument("--limit", type=int, default=None, help="Limit items for quick smoke test")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["naive", "hybrid", "agentic"],
        default=["naive", "hybrid", "agentic"],
        help="Which strategies to benchmark (default: all three)",
    )
    parser.add_argument("--skip-ragas", action="store_true", help="Skip RAGAS metrics (latency only)")
    args = parser.parse_args()

    # Load test set
    if not args.test_set.exists():
        raise FileNotFoundError(f"Test set not found: {args.test_set}")
    test_items: list[dict] = json.loads(args.test_set.read_text(encoding="utf-8"))
    if args.limit:
        test_items = test_items[: args.limit]
    logger.info("Test set: {} items from {}", len(test_items), args.test_set)

    # Configure RAGAS LLM (unless skipped)
    if not args.skip_ragas:
        _configure_ragas_llm()

    # Initialise shared RAG components (do once, share across strategies)
    logger.info("Initialising shared RAG components...")
    index, collection, all_nodes, embedder = _init_rag_shared()
    logger.info("Corpus: {} chunks ready", len(all_nodes))

    # Instantiate strategies
    strategy_map = {}
    if "naive" in args.strategies:
        strategy_map["naive"] = NaiveRAG(collection, embedder)
    if "hybrid" in args.strategies:
        strategy_map["hybrid"] = HybridRAG(index, all_nodes)
    if "agentic" in args.strategies:
        strategy_map["agentic"] = AgenticRAG(index, all_nodes)

    # Run benchmark
    results = []
    for name, strategy in strategy_map.items():
        res = run_strategy(strategy, test_items)
        results.append(res)

    # Print table
    print_table(results)

    # Save report
    save_report(results, args.output)
    print(f"Full report: {args.output}\n")


if __name__ == "__main__":
    main()
