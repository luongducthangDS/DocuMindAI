"""
RAG Retrieval Strategy Benchmark — DocuMind AI
================================================
Full ablation across four retrieval configurations:

  ① BM25-only     — Okapi BM25 exact-match only, no dense vector, no rerank
  ② Dense-only    — ChromaDB vector similarity, top-5, no BM25, no rerank
  ③ Hybrid        — BM25 + dense, RRF fusion, top-20, no reranker
  ④ Hybrid+Rerank — BM25 + dense, RRF fusion, top-20, cross-encoder reranker → top-8

Metrics per strategy:
  • RAGAS: faithfulness, answer_relevancy, context_recall, context_precision
  • Latency: mean, p50, p95 (ms)
  • Retrieval: avg chunks returned

Usage:
  # Quick smoke test (first 5 questions only)
  python eval/run_evals.py --limit 5 --output reports/benchmark_quick.json

  # Full evaluation (all 4 strategies)
  python eval/run_evals.py \\
    --test-set data/eval/test_questions.json \\
    --output reports/benchmark_results.json

  # Run only specific strategies
  python eval/run_evals.py --strategies dense rerank

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
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Force HF cache to local path before any sentence_transformers / transformers import.
# System HF_HOME may point to G:\My Drive\HF_Cache_Models (Google Drive, often offline)
# which causes a native-level crash (0xC0000005) on Windows when Drive is unmounted.
_local_hf = str(Path(__file__).resolve().parents[1] / "data" / "hf_cache")
for _k in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
    os.environ[_k] = _local_hf

# Windows console defaults to cp1252 which can't encode Vietnamese / box-drawing chars.
# Reconfigure stdout/stderr to UTF-8 so print() doesn't crash on non-ASCII output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from loguru import logger

import src.logger  # noqa: F401 — init loguru sinks
from src.config import get_settings


# ── RAG initialisation ────────────────────────────────────────────────────────

def _init_rag_shared() -> tuple:
    """
    Bootstrap shared RAG components used by all strategies.
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

    result = collection.get(include=["documents", "metadatas"])
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    all_nodes = [
        TextNode(text=d, metadata=m or {})
        for d, m in zip(docs, metas) if d
    ]
    logger.info("Shared RAG init: {} nodes in corpus", len(all_nodes))
    return index, collection, all_nodes, embedder


# ── Strategy ①: BM25-only ────────────────────────────────────────────────────

class BM25OnlyRAG:
    """
    Sparse-only baseline: Okapi BM25 exact-match retrieval, no dense vectors.
    Establishes the floor for sparse retrieval — strong on exact legal terms
    ("Điều 48"), weak on paraphrase and semantic queries.
    """

    name = "bm25_only"

    def __init__(self, all_nodes):
        self._nodes = all_nodes

    def retrieve_and_answer(self, question: str, retrieve_only: bool = False) -> tuple[str, list[str]]:
        from llama_index.retrievers.bm25 import BM25Retriever
        from src.rag.generator import generate_answer
        from src.rag.retriever import nodes_to_chunks

        try:
            retriever = BM25Retriever.from_defaults(
                nodes=self._nodes,
                similarity_top_k=5,
            )
            nodes = retriever.retrieve(question)
        except Exception as exc:
            logger.warning("BM25-only retrieve failed: {}", exc)
            return "Tôi không tìm thấy thông tin liên quan.", []

        chunks = nodes_to_chunks(nodes)
        contexts = [c.text for c in chunks]

        if not contexts:
            return "Tôi không tìm thấy thông tin liên quan.", []

        if retrieve_only:
            return "", contexts
        # BM25 raw scores aren't on the cross-encoder 0.05 scale → disable abstain gate
        result = generate_answer(question, chunks, min_score=0.0)
        return result["answer"], contexts


# ── Strategy ②: Dense-only ───────────────────────────────────────────────────

class DenseOnlyRAG:
    """
    Baseline: direct ChromaDB vector similarity search, top-5.
    No BM25, no RRF fusion, no reranker.
    """

    name = "dense_only"

    def __init__(self, collection, embedder):
        self._collection = collection
        self._embedder = embedder

    def retrieve_and_answer(self, question: str, retrieve_only: bool = False) -> tuple[str, list[str]]:
        query_vec = self._embedder.get_query_embedding(question)

        results = self._collection.query(
            query_embeddings=[query_vec],
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        contexts = [d for d in docs if d]

        if not contexts:
            return "Tôi không tìm thấy thông tin liên quan.", []

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
        if retrieve_only:
            return "", contexts
        # Dense cosine sim is 0-1; keep a light floor but not the cross-encoder gate
        result = generate_answer(question, chunks, min_score=0.0)
        return result["answer"], contexts


# ── Strategy ②: Hybrid (no reranker) ─────────────────────────────────────────

class HybridRAG:
    """
    BM25 + dense vector, RRF fusion, top-20.
    No cross-encoder reranker — isolates the contribution of hybrid retrieval alone.
    """

    name = "hybrid"

    def __init__(self, index, all_nodes):
        from src.rag.retriever import build_hybrid_retriever
        self._retriever = build_hybrid_retriever(index, nodes=all_nodes, rerank=False)

    def retrieve_and_answer(self, question: str, retrieve_only: bool = False) -> tuple[str, list[str]]:
        from src.rag.generator import generate_answer
        from src.rag.retriever import nodes_to_chunks

        nodes = self._retriever.retrieve(question)
        chunks = nodes_to_chunks(nodes)
        contexts = [c.text for c in chunks]

        if retrieve_only:
            return "", contexts
        # RRF fusion scores are ~0.016 (always < 0.05) → disable cross-encoder gate
        result = generate_answer(question, chunks, min_score=0.0)
        return result["answer"], contexts


# ── Strategy ③: Hybrid + Reranker ────────────────────────────────────────────

class HybridRerankRAG:
    """
    Production config: BM25 + dense, RRF fusion top-20, cross-encoder reranker → top-8.
    ms-marco-MiniLM-L-6-v2 reranker runs locally, no API call.
    """

    name = "hybrid_rerank"

    def __init__(self, index, all_nodes):
        from src.rag.retriever import build_hybrid_retriever
        self._retriever = build_hybrid_retriever(index, nodes=all_nodes, rerank=True)

    def retrieve_and_answer(self, question: str, retrieve_only: bool = False) -> tuple[str, list[str]]:
        from src.rag.generator import generate_answer
        from src.rag.retriever import nodes_to_chunks

        nodes = self._retriever.retrieve(question)
        chunks = nodes_to_chunks(nodes)
        contexts = [c.text for c in chunks]

        if retrieve_only:
            return "", contexts
        result = generate_answer(question, chunks)
        return result["answer"], contexts


# ── RAGAS evaluation ──────────────────────────────────────────────────────────

class _STEmbeddings:
    """
    LangChain Embeddings shim that delegates to the project's existing LlamaIndex embedder.

    Reuses get_embedder() (llama_index HuggingFaceEmbedding, lru_cache) instead of loading
    a separate SentenceTransformer or HuggingFaceEmbeddings instance. This avoids hf_xet
    which tries to write logs to HF_HOME and fails when HF_HOME points to an unmounted
    network drive (e.g. Google Drive mapped as G:\).
    """

    def __init__(self):
        from src.rag.embedder import get_embedder
        self._embedder = get_embedder()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embedder.get_text_embedding(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.get_query_embedding(text)


class _DirectGeminiRagasLLM:
    """
    Custom RAGAS LLM: calls google.generativeai directly, rotates across all
    (api_key × model) pairs to maximise free-tier throughput.

    Avoids `generate_content() got an unexpected keyword argument 'temperature'`
    (langchain-google-genai 1.0.x + google-genai 2.x incompatibility).

    Rate limit: 3 calls / 61 s per pair ≈ 1 call/20 s — safe under the 5 RPM
    free-tier cap.  With 3 keys × 5 models = 15 pairs → ~45 req/min total,
    400 RAGAS jobs complete in ~9 minutes.
    """

    _LOCK: "asyncio.Lock | None" = None
    _LOCK_LOOP = None            # event loop the current _LOCK is bound to (RAGAS uses a new loop per batch)
    _MAX_PER_WINDOW: int = 10     # 10 calls/61s per pair — safe under gemini-3.1-flash-lite's 15 RPM
    _WINDOW_SEC: float = 61.0
    _PAIR_WINDOWS: "dict" = {}    # (key, model) → [monotonic_timestamps]
    _PAIR_IDX: int = 0            # round-robin cursor across pairs
    _DEAD_PAIRS: "set" = set()    # permanently skip — 404 model or RPD-exhausted
    _PAIR_FAIL: "dict" = {}       # consecutive 429 count per pair; ≥3 → dead
    _MAX_CONSEC_FAIL: int = 3

    def __init__(self, model_name: str = ""):
        import google.generativeai as genai
        settings = get_settings()
        keys = [k for k in [
            settings.google_api_key,
            settings.google_api_key_2,
            settings.google_api_key_3,
        ] if k]
        if not keys:
            raise ValueError("No GOOGLE_API_KEY configured")
        models = [m.strip() for m in settings.gemini_judge_models.split(",") if m.strip()]
        # model-major: consecutive pairs rotate KEYS first (k1,m1),(k2,m1),(k3,m1),(k1,m2)...
        # so the round-robin spreads load across all 3 keys instead of draining key #1 first.
        self._pairs = [(k, m) for m in models for k in keys]
        for p in self._pairs:
            _DirectGeminiRagasLLM._PAIR_WINDOWS.setdefault(p, [])
            _DirectGeminiRagasLLM._PAIR_FAIL.setdefault(p, 0)
        self._genai = genai
        self.run_config = None

    def set_run_config(self, run_config) -> None:
        self.run_config = run_config

    async def generate(self, prompt, n: int = 1, temperature=None, stop=None, **kwargs):
        import asyncio
        import time
        from langchain_core.outputs import LLMResult, ChatGeneration
        from langchain_core.messages import AIMessage
        from google.api_core.exceptions import ResourceExhausted, NotFound

        # RAGAS runs each batch in a NEW event loop; an asyncio.Lock is bound to the
        # loop it was created in. Recreate the lock whenever the running loop changes,
        # otherwise batch 2+ crash with "Lock is bound to a different event loop".
        try:
            _cur_loop = asyncio.get_running_loop()
        except RuntimeError:
            _cur_loop = None
        if _DirectGeminiRagasLLM._LOCK is None or _DirectGeminiRagasLLM._LOCK_LOOP is not _cur_loop:
            _DirectGeminiRagasLLM._LOCK = asyncio.Lock()
            _DirectGeminiRagasLLM._LOCK_LOOP = _cur_loop

        text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)

        # Pick a pair with capacity in its window (skip dead pairs)
        async with _DirectGeminiRagasLLM._LOCK:
            live_pairs = [p for p in self._pairs if p not in _DirectGeminiRagasLLM._DEAD_PAIRS]
            if not live_pairs:
                raise RuntimeError("All Gemini (key, model) pairs are dead — check API keys and model IDs")
            n_pairs = len(live_pairs)
            selected = None
            for offset in range(n_pairs):
                idx = (_DirectGeminiRagasLLM._PAIR_IDX + offset) % n_pairs
                pair = live_pairs[idx]
                now = time.monotonic()
                cutoff = now - _DirectGeminiRagasLLM._WINDOW_SEC
                _DirectGeminiRagasLLM._PAIR_WINDOWS[pair] = [
                    t for t in _DirectGeminiRagasLLM._PAIR_WINDOWS[pair] if t > cutoff
                ]
                if len(_DirectGeminiRagasLLM._PAIR_WINDOWS[pair]) < _DirectGeminiRagasLLM._MAX_PER_WINDOW:
                    _DirectGeminiRagasLLM._PAIR_WINDOWS[pair].append(time.monotonic())
                    _DirectGeminiRagasLLM._PAIR_IDX = (idx + 1) % n_pairs
                    selected = pair
                    break

            if selected is None:
                # All live pairs full: sleep until earliest slot frees up
                earliest = min(
                    _DirectGeminiRagasLLM._PAIR_WINDOWS[p][0]
                    for p in live_pairs
                    if _DirectGeminiRagasLLM._PAIR_WINDOWS[p]
                )
                sleep_for = earliest + _DirectGeminiRagasLLM._WINDOW_SEC - time.monotonic() + 0.2
                await asyncio.sleep(max(sleep_for, 0.5))
                return await self.generate(prompt, n=n, **kwargs)

        key, model_name = selected

        def _sync_call():
            self._genai.configure(api_key=key)
            return self._genai.GenerativeModel(model_name).generate_content(text).text

        loop = asyncio.get_event_loop()
        try:
            result_text = await loop.run_in_executor(None, _sync_call)
            # Success — reset consecutive failure count
            async with _DirectGeminiRagasLLM._LOCK:
                _DirectGeminiRagasLLM._PAIR_FAIL[selected] = 0
        except NotFound:
            # Model ID doesn't exist — permanently remove this pair
            async with _DirectGeminiRagasLLM._LOCK:
                _DirectGeminiRagasLLM._DEAD_PAIRS.add(selected)
                live = len(self._pairs) - len(_DirectGeminiRagasLLM._DEAD_PAIRS)
                logger.warning(f"Model '{model_name}' not found — dead ({live} pairs remaining)")
            return await self.generate(prompt, n=n, **kwargs)
        except ResourceExhausted:
            async with _DirectGeminiRagasLLM._LOCK:
                _DirectGeminiRagasLLM._PAIR_FAIL[selected] = _DirectGeminiRagasLLM._PAIR_FAIL.get(selected, 0) + 1
                fail_count = _DirectGeminiRagasLLM._PAIR_FAIL[selected]
                if fail_count >= _DirectGeminiRagasLLM._MAX_CONSEC_FAIL:
                    # RPD exhausted — permanently kill this pair
                    _DirectGeminiRagasLLM._DEAD_PAIRS.add(selected)
                    live = len(self._pairs) - len(_DirectGeminiRagasLLM._DEAD_PAIRS)
                    logger.warning(f"Pair ({model_name} key…{key[-4:]}) RPD exhausted after {fail_count} retries — dead ({live} pairs remaining)")
                else:
                    logger.warning(f"Pair ({model_name} key…{key[-4:]}) hit quota ({fail_count}/{_DirectGeminiRagasLLM._MAX_CONSEC_FAIL}) — rotating")
            return await self.generate(prompt, n=n, **kwargs)

        generations = [[ChatGeneration(message=AIMessage(content=result_text))]]
        return LLMResult(generations=generations)


def _configure_ragas_llm() -> tuple:
    """
    Build and return (ragas_llm_wrapper, ragas_emb_wrapper).
    Caller passes these explicitly to ragas_evaluate — avoids relying on
    global ragas.llm which is not picked up reliably in 0.1.21.
    Returns (None, None) on failure.
    """
    try:
        settings = get_settings()

        # Use direct Gemini wrapper (bypasses LangChain) to avoid temperature kwarg
        # incompatibility between langchain-google-genai 1.0.x and google-genai 2.x.
        if settings.google_api_key:
            ragas_llm = _DirectGeminiRagasLLM()
            n_keys = sum(1 for k in [settings.google_api_key, settings.google_api_key_2, settings.google_api_key_3] if k)
            n_models = len([m for m in settings.gemini_judge_models.split(",") if m.strip()])
            n_pairs = n_keys * n_models
            logger.info(f"RAGAS judge: {n_pairs} pairs ({n_keys} keys × {n_models} models), ~{n_pairs * _DirectGeminiRagasLLM._MAX_PER_WINDOW} req/min")
        elif settings.groq_api_key:
            from langchain_groq import ChatGroq
            from ragas.llms import LangchainLLMWrapper
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=settings.groq_api_key,
                temperature=0,
            )
            ragas_llm = LangchainLLMWrapper(llm)
            logger.info("RAGAS judge LLM: Groq llama-3.3-70b (fallback)")
        else:
            raise RuntimeError("No GOOGLE_API_KEY or GROQ_API_KEY for RAGAS judge")

        from ragas.embeddings import LangchainEmbeddingsWrapper

        # _STEmbeddings reuses the project's existing LlamaIndex embedder (lru_cache).
        # Avoids HuggingFaceEmbeddings / hf_xet which fails when HF_HOME points to
        # an unmounted drive (e.g. Google Drive mapped as G:\).
        ragas_emb = LangchainEmbeddingsWrapper(_STEmbeddings())

        return ragas_llm, ragas_emb

    except Exception as exc:
        logger.warning("RAGAS LLM config failed — RAGAS metrics will be skipped: {}", exc)
        return None, None


def run_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
    ragas_llm=None,
    ragas_emb=None,
    checkpoint_path: "Path | None" = None,
    batch_size: int = 10,
) -> dict[str, float]:
    """
    Run RAGAS on a (question, answer, contexts, ground_truth) dataset.
    Runs in batches of batch_size, saving a per-question checkpoint after each batch.
    On restart, loads the checkpoint and skips already-evaluated questions.
    Returns metric dict (nanmean across all scored questions).
    """
    import math, json as _json
    import numpy as np

    if ragas_llm is None:
        logger.warning("No RAGAS judge LLM — skipping RAGAS metrics")
        return {}

    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.run_config import RunConfig
    except ImportError:
        logger.error("RAGAS not installed. Run: pip install 'ragas==0.1.21' datasets>=2.14.0")
        return {}

    filtered = [
        (q, a, c, g)
        for q, a, c, g in zip(questions, answers, contexts, ground_truths)
        if c
    ]
    if not filtered:
        logger.warning("All test items had empty contexts — RAGAS skipped")
        return {}

    METRICS = [faithfulness, answer_relevancy, context_recall, context_precision]
    METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    run_cfg = RunConfig(timeout=180, max_retries=5, max_wait=10, max_workers=1)

    # Load checkpoint: {question_text: {metric: score|null}}
    completed: dict[str, dict] = {}
    if checkpoint_path and checkpoint_path.exists():
        try:
            completed = _json.loads(checkpoint_path.read_text(encoding="utf-8"))
            logger.info("RAGAS checkpoint loaded: {}/{} already scored", len(completed), len(filtered))
        except Exception:
            completed = {}

    remaining = [(q, a, c, g) for q, a, c, g in filtered if q not in completed]
    logger.info("Running RAGAS on {} items ({} remaining after checkpoint)...", len(filtered), len(remaining))

    n_batches = math.ceil(len(remaining) / batch_size) if remaining else 0
    for b, batch_start in enumerate(range(0, len(remaining), batch_size)):
        batch = remaining[batch_start: batch_start + batch_size]
        qs, ans, ctx, gts = zip(*batch)
        dataset = Dataset.from_dict({
            "question": list(qs), "answer": list(ans),
            "contexts": list(ctx), "ground_truth": list(gts),
        })
        try:
            result = ragas_evaluate(dataset=dataset, metrics=METRICS,
                                    llm=ragas_llm, embeddings=ragas_emb, run_config=run_cfg)
            df = result.to_pandas()
            for i, row in df.iterrows():
                q_key = qs[i]
                completed[q_key] = {
                    m: (None if math.isnan(float(row[m])) else round(float(row[m]), 4))
                    for m in METRIC_NAMES if m in row
                }
            logger.info("RAGAS batch {}/{} done — {}/{} total scored",
                        b + 1, n_batches, len(completed), len(filtered))
        except Exception as exc:
            logger.error("RAGAS batch {}/{} failed: {}", b + 1, n_batches, exc)

        # Save checkpoint after every batch (even on partial failure)
        if checkpoint_path:
            try:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_text(_json.dumps(completed, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as e:
                logger.warning("Could not save RAGAS checkpoint: {}", e)

    if not completed:
        return {}

    # Aggregate: nanmean across all scored questions
    agg = {}
    for m in METRIC_NAMES:
        vals = [v[m] for v in completed.values() if v.get(m) is not None]
        agg[m] = round(float(np.mean(vals)), 4) if vals else float("nan")
    return agg


# ── Strategy runner ───────────────────────────────────────────────────────────

def _cache_path(cache_dir: Path, strategy_name: str, n_items: int) -> Path:
    return cache_dir / f"{strategy_name}_{n_items}q.json"


def _load_cache(cache_dir: Path | None, strategy_name: str, questions: list[str]) -> dict | None:
    if not cache_dir:
        return None
    p = _cache_path(cache_dir, strategy_name, len(questions))
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("questions") == questions:
            logger.info("  ↩  Loaded {} cached answers for {}", len(questions), strategy_name)
            return data
        logger.warning("  Cache question mismatch for {} — regenerating", strategy_name)
    except Exception as exc:
        logger.warning("  Cache load failed: {} — regenerating", exc)
    return None


def _save_cache(
    cache_dir: Path | None,
    strategy_name: str,
    questions: list[str],
    answers: list[str],
    contexts_list: list,
    latencies_ms: list[float],
) -> None:
    if not cache_dir:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = _cache_path(cache_dir, strategy_name, len(questions))
    p.write_text(
        json.dumps({
            "strategy": strategy_name,
            "questions": questions,
            "answers": answers,
            "contexts_list": contexts_list,
            "latencies_ms": latencies_ms,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("  ✓  Cached answers → {}", p)


def run_strategy(
    strategy,
    test_items: list[dict],
    embedder=None,
    ragas_llm=None,
    ragas_emb=None,
    skip_ragas: bool = False,
    cache_dir: Path | None = None,
    retrieve_only: bool = False,
) -> dict[str, Any]:
    """
    Run one strategy on all test items.
    Collects:
      • latency stats
      • RAGAS metrics (faithfulness, answer_relevancy, context_recall, context_precision)
      • local metrics (hit_rate, mrr, answer_correctness, citation_rate, ooc_refusal_rate)

    Answer caching: if cache_dir is set, generated answers are saved to
    ``cache_dir/{strategy}_{N}q.json`` and reused on subsequent runs, so
    RAGAS can be re-run without calling the generation LLM again.
    """
    from eval.metrics import compute_all

    if retrieve_only:
        # Pure retrieval pass: no answer generation → no LLM, no quota, no cache.
        skip_ragas = True
        cache_dir = None  # answers are empty here — must never poison the answer cache

    ground_truths = [item.get("ground_truth", "") for item in test_items]
    all_questions = [item["question"] for item in test_items]

    logger.info("▶  Running strategy: {}", strategy.name)

    # ── Try loading cached answers ────────────────────────────────────────────
    cached = _load_cache(cache_dir, strategy.name, all_questions)
    if cached:
        questions = cached["questions"]
        answers = cached["answers"]
        contexts_list = cached["contexts_list"]
        latencies_ms = cached["latencies_ms"]
    else:
        latencies_ms: list[float] = []
        questions, answers, contexts_list = [], [], []

        for i, item in enumerate(test_items, 1):
            q = item["question"]

            t0 = time.perf_counter()
            try:
                answer, ctx = strategy.retrieve_and_answer(q, retrieve_only=retrieve_only)
            except Exception as exc:
                logger.warning("[{}/{}] {} failed: {}", i, len(test_items), strategy.name, exc)
                answer, ctx = "", []

            latency = (time.perf_counter() - t0) * 1000
            latencies_ms.append(latency)
            questions.append(q)
            answers.append(answer)
            contexts_list.append(ctx)

            logger.info("  [{}/{}] {:.0f}ms | Q: {}", i, len(test_items), latency, q[:55])

        _save_cache(cache_dir, strategy.name, questions, answers, contexts_list, latencies_ms)

    lat_sorted = sorted(latencies_ms)
    p95_idx = int(len(lat_sorted) * 0.95)
    latency_stats = {
        "mean_ms": round(statistics.mean(latencies_ms), 1),
        "median_ms": round(statistics.median(latencies_ms), 1),
        "p95_ms": round(lat_sorted[min(p95_idx, len(lat_sorted) - 1)], 1),
        "min_ms": round(min(latencies_ms), 1),
        "max_ms": round(max(latencies_ms), 1),
    }

    # Per-question checkpoint: reports/ragas_checkpoints/<strategy>_<n>q.json
    ragas_ckpt = None
    if not skip_ragas and cache_dir:
        ragas_ckpt = cache_dir.parent / "ragas_checkpoints" / f"{strategy.name}_{len(questions)}q.json"
    ragas_metrics = {} if skip_ragas else run_ragas(
        questions, answers, contexts_list, ground_truths,
        ragas_llm=ragas_llm, ragas_emb=ragas_emb,
        checkpoint_path=ragas_ckpt,
    )

    local_metrics = compute_all(
        test_items=test_items,
        answers=answers,
        contexts_list=contexts_list,
        ground_truths=ground_truths,
        embedder=embedder,
    )

    chunk_counts = [len(c) for c in contexts_list]

    return {
        "strategy": strategy.name,
        "n_items": len(test_items),
        "latency": latency_stats,
        "ragas": ragas_metrics,
        "retrieval_local": {
            "hit_rate": local_metrics["retrieval"]["hit_rate"],
            "mrr": local_metrics["retrieval"]["mrr"],
            "avg_chunks": round(statistics.mean(chunk_counts), 1) if chunk_counts else 0,
            "items_with_context": sum(1 for c in contexts_list if c),
        },
        "generation_local": {} if retrieve_only else local_metrics["generation"],
        "domain": {} if retrieve_only else local_metrics["domain"],
    }


# ── Report ────────────────────────────────────────────────────────────────────

def save_report(results: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": get_settings().embedding_model,
        "corpus_chunks": None,
        "results": {r["strategy"]: r for r in results},
        "summary": _build_summary(results),
    }

    try:
        from src.rag.embedder import get_chroma_collection
        _, col = get_chroma_collection()
        report["corpus_chunks"] = col.count()
    except Exception:
        pass

    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success("Benchmark report saved → {}", output)


def _build_summary(results: list[dict]) -> dict:
    """Compute step-wise deltas across the full ablation chain."""
    by_name = {r["strategy"]: r for r in results}
    bm25 = by_name.get("bm25_only", {})
    dense = by_name.get("dense_only", {})
    hybrid = by_name.get("hybrid", {})
    rerank = by_name.get("hybrid_rerank", {})

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
        "dense_vs_bm25": {
            "faithfulness_delta_pct": ragas_delta(bm25, dense, "faithfulness"),
            "context_recall_delta_pct": ragas_delta(bm25, dense, "context_recall"),
        },
        "hybrid_vs_dense": {
            "faithfulness_delta_pct": ragas_delta(dense, hybrid, "faithfulness"),
            "answer_relevancy_delta_pct": ragas_delta(dense, hybrid, "answer_relevancy"),
            "context_recall_delta_pct": ragas_delta(dense, hybrid, "context_recall"),
            "latency_overhead_pct": pct_delta(
                dense.get("latency", {}).get("mean_ms"),
                hybrid.get("latency", {}).get("mean_ms"),
            ),
        },
        "rerank_vs_hybrid": {
            "faithfulness_delta_pct": ragas_delta(hybrid, rerank, "faithfulness"),
            "answer_relevancy_delta_pct": ragas_delta(hybrid, rerank, "answer_relevancy"),
            "context_recall_delta_pct": ragas_delta(hybrid, rerank, "context_recall"),
            "latency_overhead_pct": pct_delta(
                hybrid.get("latency", {}).get("mean_ms"),
                rerank.get("latency", {}).get("mean_ms"),
            ),
        },
        "rerank_vs_dense": {
            "faithfulness_delta_pct": ragas_delta(dense, rerank, "faithfulness"),
            "context_recall_delta_pct": ragas_delta(dense, rerank, "context_recall"),
        },
    }


def print_table(results: list[dict]) -> None:
    strategies = ["bm25_only", "dense_only", "hybrid", "hybrid_rerank"]
    headers = ["BM25-only", "Dense-only", "Hybrid", "Hybrid+Rerank"]
    by_name = {r["strategy"]: r for r in results}
    w = 84

    def _row(label: str, getter) -> str:
        row = f"  {label:<28}"
        for strat in strategies:
            val = getter(by_name.get(strat, {}))
            if val is None:
                row += f"  {'N/A':>12}"
            elif isinstance(val, float):
                row += f"  {val:>12.4f}"
            else:
                row += f"  {val:>12}"
        return row

    # ── RAGAS metrics (LLM judge) ─────────────────────────────────────
    print("\n" + "=" * w)
    print("  [RAGAS — LLM judge] faithfulness / relevancy / recall / precision")
    print("=" * w)
    header_row = f"  {'Metric':<28}" + "".join(f"  {h:>12}" for h in headers)
    print(header_row)
    print("-" * w)
    for m in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        print(_row(m, lambda r, _m=m: r.get("ragas", {}).get(_m)))
    print("-" * w)
    print(_row("Avg Latency (ms)", lambda r: r.get("latency", {}).get("mean_ms")))
    print(_row("P95 Latency (ms)", lambda r: r.get("latency", {}).get("p95_ms")))

    # ── Local retrieval metrics (no LLM) ──────────────────────────────
    print("=" * w)
    print("  [Retrieval — local, no LLM] hit_rate / MRR / avg chunks")
    print("-" * w)
    print(_row("hit_rate@K", lambda r: r.get("retrieval_local", {}).get("hit_rate")))
    print(_row("mrr", lambda r: r.get("retrieval_local", {}).get("mrr")))
    print(_row("avg_chunks_returned", lambda r: r.get("retrieval_local", {}).get("avg_chunks")))

    # ── Local generation metrics (no LLM) ────────────────────────────
    print("=" * w)
    print("  [Generation — local, no LLM] answer_correctness vs ground truth")
    print("-" * w)
    print(_row("correctness_semantic", lambda r: r.get("generation_local", {}).get("answer_correctness_semantic")))
    print(_row("correctness_token_f1", lambda r: r.get("generation_local", {}).get("answer_correctness_f1")))

    # ── Domain metrics ────────────────────────────────────────────────
    print("=" * w)
    print("  [Domain — legal] citation_rate / OOC refusal_rate")
    print("-" * w)
    print(_row("citation_rate", lambda r: r.get("domain", {}).get("citation_rate")))
    print(_row("ooc_refusal_rate", lambda r: r.get("domain", {}).get("ooc_refusal_rate")))
    print("=" * w + "\n")

    # ── Deltas ────────────────────────────────────────────────────────
    summary = _build_summary(results)
    print("  Step-wise deltas (faithfulness | recall | latency):")
    steps = [
        ("BM25 -> Dense", "dense_vs_bm25"),
        ("Dense -> Hybrid", "hybrid_vs_dense"),
        ("Hybrid -> Hybrid+Rerank", "rerank_vs_hybrid"),
        ("Dense -> Rerank (headline)", "rerank_vs_dense"),
    ]
    for label, key in steps:
        d = summary.get(key, {})
        parts = []
        for metric_key, short in [
            ("faithfulness_delta_pct", "faithfulness"),
            ("context_recall_delta_pct", "recall"),
            ("latency_overhead_pct", "latency"),
        ]:
            val = d.get(metric_key)
            if val is not None:
                sign = "+" if val > 0 else ""
                parts.append(f"{short} {sign}{val:.1f}%")
        if parts:
            print(f"    {label:<30} {' | '.join(parts)}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DocuMind AI — Retrieval Strategy Benchmark (dense vs hybrid vs hybrid+rerank)"
    )
    parser.add_argument("--test-set", type=Path, default=Path("data/eval/test_questions.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark_results.json"))
    parser.add_argument("--limit", type=int, default=None, help="Limit items for quick smoke test")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["bm25", "dense", "hybrid", "rerank"],
        default=["bm25", "dense", "hybrid", "rerank"],
        help="Which strategies to benchmark (default: all four)",
    )
    parser.add_argument("--skip-ragas", action="store_true", help="Skip RAGAS metrics (latency only)")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("reports/answers_cache"),
        help="Directory for cached answers (default: reports/answers_cache). "
             "Caching lets you re-run RAGAS without re-calling the generation LLM.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable answer caching (always regenerate)")
    parser.add_argument("--fresh", action="store_true",
                        help="Xóa answers_cache + ragas_checkpoints trước khi chạy "
                             "(đo lại từ đầu sau khi đổi code/model)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="CHỈ đo retrieval (hit_rate/MRR/avg_chunks) — KHÔNG gọi LLM "
                             "generation, không quota, chạy vài giây cho cả 4 strategy")
    args = parser.parse_args()

    if args.retrieval_only:
        args.skip_ragas = True  # retrieval-only → không sinh câu trả lời → không có gì để judge

    if args.fresh:
        for d in (args.cache_dir, args.cache_dir.parent / "ragas_checkpoints"):
            removed = 0
            if d.exists():
                for f in d.glob("*.json"):
                    f.unlink()
                    removed += 1
            logger.info("--fresh: xóa {} file trong {}", removed, d)

    if not args.test_set.exists():
        raise FileNotFoundError(f"Test set not found: {args.test_set}")
    test_items: list[dict] = json.loads(args.test_set.read_text(encoding="utf-8"))
    if args.limit:
        test_items = test_items[: args.limit]
    logger.info("Test set: {} items from {}", len(test_items), args.test_set)

    ragas_llm, ragas_emb = (None, None)
    if not args.skip_ragas:
        ragas_llm, ragas_emb = _configure_ragas_llm()

    logger.info("Initialising shared RAG components...")
    index, collection, all_nodes, embedder = _init_rag_shared()
    logger.info("Corpus: {} chunks ready", len(all_nodes))

    strategy_map = {}
    if "bm25" in args.strategies:
        strategy_map["bm25"] = BM25OnlyRAG(all_nodes)
    if "dense" in args.strategies:
        strategy_map["dense"] = DenseOnlyRAG(collection, embedder)
    if "hybrid" in args.strategies:
        strategy_map["hybrid"] = HybridRAG(index, all_nodes)
    if "rerank" in args.strategies:
        strategy_map["rerank"] = HybridRerankRAG(index, all_nodes)

    cache_dir = None if args.no_cache else args.cache_dir

    results = []
    for name, strategy in strategy_map.items():
        res = run_strategy(
            strategy, test_items,
            embedder=embedder,
            ragas_llm=ragas_llm, ragas_emb=ragas_emb,
            skip_ragas=args.skip_ragas,
            cache_dir=cache_dir,
            retrieve_only=args.retrieval_only,
        )
        results.append(res)

    print_table(results)
    save_report(results, args.output)
    print(f"Full report: {args.output}\n")


if __name__ == "__main__":
    main()
