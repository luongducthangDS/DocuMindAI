"""
RAGAS evaluation harness for DocuMind AI.
Usage:
  python eval/ragas_eval.py \
    --test-set data/eval/test_questions.json \
    --output reports/ragas_report.json

Test set format:
  [{"question": "...", "ground_truth": "...", "contexts": ["..."]}]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from loguru import logger

import src.logger  # noqa: F401


def load_test_set(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Test set not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Test set must be a non-empty JSON array")
    return data


def run_retrieval(question: str) -> tuple[str, list[str]]:
    """Run the full RAG pipeline for a single question."""
    import asyncio
    import src.rag.retriever as r_module
    from src.rag.generator import generate_answer
    from src.rag.retriever import nodes_to_chunks

    retriever = getattr(r_module, "_active_retriever", None)
    if retriever is None:
        return "", []

    if hasattr(retriever, "aretrieve"):
        nodes = asyncio.get_event_loop().run_until_complete(retriever.aretrieve(question))
    else:
        nodes = retriever.retrieve(question)

    chunks = nodes_to_chunks(nodes)
    contexts = [c.text for c in chunks]

    result = generate_answer(question, chunks)
    return result["answer"], contexts


def evaluate(test_items: list[dict]) -> dict:
    """
    Run RAGAS evaluation on test set.
    Returns metric dict: faithfulness, answer_relevancy, context_recall, context_precision.
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
    except ImportError as exc:
        raise RuntimeError("Install ragas: pip install ragas") from exc

    questions, answers, contexts, ground_truths = [], [], [], []

    logger.info("Running RAG pipeline on {} questions...", len(test_items))
    for i, item in enumerate(test_items, 1):
        q = item["question"]
        gt = item.get("ground_truth", "")

        t0 = time.time()
        answer, ctx = run_retrieval(q)
        elapsed = int((time.time() - t0) * 1000)

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx or item.get("contexts", []))
        ground_truths.append(gt)

        logger.info("[{}/{}] {} ms | Q: {}", i, len(test_items), elapsed, q[:60])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    logger.info("Running RAGAS metrics...")
    result = ragas_evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    return result.to_pandas().mean(numeric_only=True).to_dict()


def save_report(metrics: dict, output: Path, test_set_path: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_set": str(test_set_path),
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "targets": {
            "faithfulness": 0.80,
            "answer_relevancy": 0.75,
            "context_recall": 0.70,
        },
        "pass": all([
            metrics.get("faithfulness", 0) >= 0.80,
            metrics.get("answer_relevancy", 0) >= 0.75,
            metrics.get("context_recall", 0) >= 0.70,
        ]),
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("RAGAS report saved: {}", output)
    return report


def _init_rag() -> None:
    """Initialize RAG stack before eval."""
    from src.rag.embedder import get_chroma_collection, get_embedder
    from src.rag.retriever import build_hybrid_retriever
    from llama_index.core import Settings as LlamaSettings, VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import StorageContext
    import src.rag.retriever as r_module

    embedder = get_embedder()
    LlamaSettings.embed_model = embedder
    LlamaSettings.llm = None

    _, chroma_col = get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=chroma_col)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_ctx,
        embed_model=embedder,
    )
    r_module._active_index = index
    r_module._active_retriever = build_hybrid_retriever(index, nodes=[], rerank=False)
    logger.info("RAG stack ready for evaluation")


def main() -> None:
    parser = argparse.ArgumentParser(description="DocuMind AI — RAGAS Evaluation")
    parser.add_argument(
        "--test-set",
        type=Path,
        default=Path("data/eval/test_questions.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/ragas_report.json"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit test items for quick run")
    args = parser.parse_args()

    _init_rag()

    test_items = load_test_set(args.test_set)
    if args.limit:
        test_items = test_items[: args.limit]

    metrics = evaluate(test_items)
    report = save_report(metrics, args.output, args.test_set)

    print("\n" + "=" * 50)
    print("RAGAS Evaluation Results")
    print("=" * 50)
    for k, v in report["metrics"].items():
        target = report["targets"].get(k)
        status = "✅" if target is None or v >= target else "❌"
        print(f"  {status} {k}: {v:.4f}" + (f"  (target: {target})" if target else ""))
    print("=" * 50)
    print(f"Overall: {'✅ PASS' if report['pass'] else '❌ FAIL'}")


if __name__ == "__main__":
    main()
