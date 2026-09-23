<!--
SYNC IMPACT REPORT (scratch — remove before commit)
Version change: (none) → 1.0.0
Bump rationale: initial ratification of an existing project; no prior constitution file existed.
Principles added:
  I. Grounded Answers or Refusal (NON-NEGOTIABLE)
  II. Corpus Boundary Is the Product
  III. Evidence-Based Evaluation
  IV. Configuration Over Hard-Coding
  V. Green Test Suite Before Claim
Sections added: Technology Constraints; Development Workflow & Quality Gates; Governance
Sections removed: none
Deferred TODOs: none
Templates requiring no change: plan-template.md, spec-template.md, tasks-template.md
-->

# DocuMind AI Constitution

## Core Principles

### I. Grounded Answers or Refusal (NON-NEGOTIABLE)

Every user-facing answer MUST be traceable to retrieved corpus text. Concretely:

- Each substantive claim carries an inline citation `[N]` resolving to a chunk with document
  identity (văn bản, Điều/Khoản) in its metadata.
- When retrieval returns nothing above the relevance floor (`score < 0.05` filter in
  `generator.py`), the system MUST return the "không tìm thấy" response WITHOUT calling the LLM
  and WITHOUT showing sources. Answering from model parametric knowledge is a defect, not a
  fallback.
- Generation temperature stays at `0.0` so citations are reproducible across runs.

Rationale: the product is consulted for statutory/regulatory decisions. An unsourced sentence
that happens to be right is indistinguishable, to the user, from one that is wrong.

### II. Corpus Boundary Is the Product

The system MUST make its own scope legible and MUST enforce it.

- The published corpus manifest (count of VERIFIED documents, coverage date, and an explicit
  "ngoài phạm vi" list) is part of the user interface, not internal documentation.
- `scope_gate` MUST reject out-of-scope questions with (a) the corpus scope, (b) the kinds of
  work not handled, and (c) up to 3 nearby answerable questions.
- `scope_gate` MUST be fail-closed: if the classifier errors, report a classification failure —
  never silently treat the question as out of scope, and never fall through to answering.

Rationale: the differentiator over a generic assistant is knowing where knowledge ends.

### III. Evidence-Based Evaluation

Reported numbers MUST come from a reproducible run over a written-first gold set.

- Gold answers are authored BEFORE the system is run against them.
- The gold set covers scenario questions, out-of-scope questions, and temporal questions.
- Retrieval is scored against stable identifiers (`clause_uid`, `version_id`), not token overlap.
- Any metric that appears in README.md or EVALUATION.md MUST name the run that produced it and
  the date. Unsourced or stale numbers MUST be removed rather than rounded or re-used.
- Changing the embedding model requires an A/B run on the gold set before adoption
  (`eval/embedding_ab.py`), with results committed under `reports/`.

Rationale: this repository is read as evidence of engineering judgement; a benchmark number
without provenance destroys the credibility of every other number next to it.

### IV. Configuration Over Hard-Coding

Model identities, ports, providers, and paths MUST have exactly one source of truth in `.env`
plus `src/config.py`.

- `EMBEDDING_MODEL` is the only place the embedder is named; the vector collection stores
  `embedding_model` / `embedding_dim` metadata and MUST raise `EmbeddingModelMismatch` on drift.
- Vector store access goes through `src/rag/vector_backend.py`; swapping
  `VECTOR_STORE_PROVIDER` between `chroma` and `qdrant` MUST NOT require edits at any call site.
- HuggingFace cache setup happens only via `use_local_hf_cache()` in `src/hf_env.py`.
- Unknown `.env` keys are rejected (pydantic `extra_forbidden`); config drift fails loudly at
  startup rather than silently at inference time.

Rationale: duplicated configuration has already caused real failures in this repo (eight copies
of HF cache setup in four variants); one source of truth is cheaper than the incident.

### V. Green Test Suite Before Claim

`pytest -q` MUST pass in full before any change is described as done, and before any status
claim is written into README.md, CLAUDE.md, or EVALUATION.md.

- The stated pass count in documentation MUST match the last actual run, with its date.
- A change to retrieval, chunking, or the embedding model additionally requires the temporal
  eval (`eval/temporal_eval.py`) to run.
- Skipping or deleting a test to reach green is a violation, not a workaround.

Rationale: "84/84" and "166/166" have both appeared in this repo's documentation; only one of
them could be current, and a reader cannot tell which.

## Technology Constraints

- Python 3.10+ (3.11 target), FastAPI + uvicorn, LangGraph agent, ChromaDB (default) or Qdrant
  Cloud, React 19 + Vite 6 frontend.
- LLM chain is ordered and MUST degrade rather than fail: Groq llama-3.3-70b → Gemini
  flash-lite → OpenAI-compatible → extractive (no LLM).
- Retrieval is hybrid by construction: BM25 + dense + RRF fusion + cross-encoder rerank.
- No new runtime dependency may be added to satisfy a feature that existing components
  (LLM client, embedder, config) can serve.
- Production deployment targets exactly two services: frontend on Vercel, backend on Render.
  No other deployment target may be reintroduced without an amendment.
- Persistence is files and vector store only — no relational database.

## Development Workflow & Quality Gates

- Significant changes are specified before implementation; module specs live in `docs/spec/` and
  decisions in `docs/decisions/` using the existing `TEMPLATE.md`.
- A decision record is REQUIRED when a change alters retrieval behaviour, evaluation
  methodology, corpus scope, or the deployment surface.
- Gates before a change is considered complete, in order: full `pytest -q` green → relevant eval
  run → documentation updated to match reality → diff reviewed.
- Documentation and code MUST NOT describe different systems. If README.md, CLAUDE.md, and
  `docs/spec/` disagree about the product vertical, corpus size, or metrics, resolving the
  contradiction is a blocking task, not a cleanup task.

## Governance

This constitution supersedes ad-hoc practice for the DocuMind AI repository. Where an existing
document conflicts with it, this file wins and the other document is corrected.

- **Amendments**: proposed as a decision record in `docs/decisions/`, stating what changes, why,
  and the migration required. An amendment takes effect when merged and reflected here.
- **Versioning**: semantic. MAJOR for removing or redefining a principle in a backward
  incompatible way; MINOR for adding a principle or materially expanding one; PATCH for
  clarification and wording.
- **Compliance review**: every review verifies the five principles above. Added complexity must
  be justified against Principle IV and the no-new-dependency constraint.
- **Runtime guidance**: `CLAUDE.md` carries day-to-day operating instructions and must remain
  consistent with this constitution.

**Version**: 1.0.0 | **Ratified**: 2026-05-21 | **Last Amended**: 2026-09-18
