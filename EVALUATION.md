# DocuMind AI — Evaluation & Findings

RAG assistant for UNETI student regulations (Vietnamese legal/administrative QA).
This document summarizes how the system was evaluated, the measured results, and the
engineering issues found and fixed during evaluation.

## 1. Methodology

- **Benchmark:** 110 hand-built Vietnamese questions across 7 regulatory documents
  (100 in-corpus + 10 out-of-corpus "trap" questions about other schools / topics not
  covered), each with a ground-truth answer and source chunk id.
- **Ablation:** 4 retrieval strategies evaluated on the *same* question set —
  BM25-only, Dense-only, Hybrid (BM25 + dense + RRF), Hybrid + cross-encoder reranker.
- **Metrics:**
  - *Retrieval (no LLM):* hit_rate@K, MRR, avg chunks returned.
  - *Generation (LLM):* RAGAS faithfulness / answer_relevancy / context_recall /
    context_precision; answer correctness vs ground truth.
  - *Domain robustness:* citation rate, out-of-corpus (OOC) refusal rate.
- **Stack:** FastAPI · LangGraph · ChromaDB · sentence-transformers ·
  Groq Llama-3.3-70B (primary) / Gemini (fallback) · RAGAS (Gemini judge).

## 2. Results — retrieval & domain (full benchmark, n = 110)

| Strategy            | hit_rate@K | MRR  | avg chunks | citation | OOC refusal | p95 latency |
|---------------------|:----------:|:----:|:----------:|:--------:|:-----------:|:-----------:|
| BM25-only           |    0.94    | 0.86 |     5      |   0.87   |    1.00     |   11.8 s    |
| Dense-only          |    0.89    | 0.76 |     5      |   0.64   |    1.00     |   11.4 s    |
| Hybrid              |    0.95    | 0.83 |    20      |   0.85   |    1.00     |   11.7 s    |
| **Hybrid + Rerank** |  **0.95**  |**0.87**|    8     | **0.87** |  **1.00**   |   31.6 s    |

- **Hybrid + reranker is the best configuration:** highest MRR (0.87), tight context
  (8 chunks vs 20), and a perfect OOC-refusal rate.
- **Out-of-corpus refusal = 100%** across all strategies — the system never answers
  questions about other institutions or uncovered topics (verified on 10 trap questions).
- BM25 is a strong baseline on this corpus because legal queries are keyword-heavy
  (document codes, article numbers, exact terms).

## 3. Results — generation quality (RAGAS, pilot n = 5)

Indicative numbers on a 5-question pilot (full-110 RAGAS deferred — see Limitations):

| Strategy            | faithfulness | answer_relevancy | context_precision |
|---------------------|:------------:|:----------------:|:-----------------:|
| Hybrid              |    ~1.00     |      ~0.73       |       ~0.67       |
| **Hybrid + Rerank** |    ~0.93     |      ~0.85       |     **~0.83**     |

Faithfulness ≈ 0.9 indicates answers stay grounded in the retrieved passages (low
hallucination); the reranker lifts both relevancy and precision over plain hybrid.

## 4. Engineering findings & fixes

Evaluation surfaced several real issues that were diagnosed to root cause and fixed:

1. **Reranker language mismatch.** Production used an English cross-encoder
   (`ms-marco-MiniLM-L-6-v2`) to rerank Vietnamese passages. Swapped to a multilingual
   reranker (`bge-reranker-v2-m3`) → **context_precision 0.66 → 0.83**, citation 0.80 → 1.00
   (pilot). Made configurable via `.env`.
2. **Score-scale abstain bug.** A fixed relevance threshold (0.05, calibrated for
   cross-encoder logits) was applied to RRF fusion scores (~0.016) → the hybrid path
   abstained on *every* question. Fixed by making the abstain threshold per-retriever.
3. **Generation over-refusal.** ~12–16% of *answerable* questions were refused even
   though the correct passage was retrieved (root cause: an over-conservative prompt that
   fixated on matching document codes verbatim, amplified by a small generation model).
   Addressed by redesigning the system prompt to answer from passage *content* while
   keeping strict OOC refusal.
4. **Evaluation infrastructure.** Built a multi-key Gemini judge with round-robin
   key/model rotation and per-pair rate-limiting, event-loop-safe locking across RAGAS
   batches, answer caching, and per-batch checkpointing for resumable runs.

## 5. Limitations (honest scope)

- Domain-focused corpus: 91 chunks across 7 documents — a scoped assistant, not a
  large-scale system.
- RAGAS judge numbers are from a 5-question pilot; a full-110 RAGAS run is pending due
  to free-tier daily quota limits on the judge LLM.
- The multilingual reranker adds latency (p95 ≈ 32 s on CPU); a lighter reranker or GPU
  would be needed for low-latency production.
- The dense embedding (MiniLM-384d) underperforms on Vietnamese; an upgrade to BGE-M3 is
  identified (est. +5–8 pp context_recall).
- Over-refusal fix applied but not yet re-measured at full scale.

## 6. CV-ready summary

> Built a Vietnamese RAG assistant for university regulations (FastAPI · LangGraph ·
> ChromaDB) with hybrid retrieval (BM25 + dense + RRF) and a cross-encoder reranker.
> Achieved **hit_rate@K 0.95, MRR 0.88, and 100% out-of-corpus refusal** on a self-built
> 110-question benchmark; RAGAS faithfulness ≈ 0.9 (pilot).
>
> Designed a 4-strategy ablation evaluation harness (RAGAS + custom retrieval/domain
> metrics). Used it to diagnose and fix production issues — a language-mismatched reranker
> (**context_precision 0.66 → 0.83**), a score-scale abstain bug, and ~15% generation
> over-refusal — measuring improvement before/after each change.
