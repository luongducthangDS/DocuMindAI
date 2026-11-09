# DocuMind as an enterprise RAG and agent platform

This page maps each platform requirement to where it lives in the code and how it is measured.
It covers the four additions on branch `feat/platform-upgrade` plus what DocuMind already had.

## Request flow

```
client ──(x-api-key)──► principal.resolve_context ──► RetrievalContext(tenant, ACL labels, as_of)
                                    │                     (labels come from data/tenants, never the request)
                                    ▼
          LangGraph: contextualize → router ─┬─ simple_qa → hybrid retrieve (BM25 + dense) → rerank
                                             │                → temporal filter → grade → answer
                                             ├─ compare / summarize / report / compliance_check
                                             └─ action → propose()  ──► agent_actions: pending
                                                                         │
           admin (X-Admin-Key, X-Approver) ── POST /actions/{id}/approve ┘──► execute once ──► hr_requests
every LLM call ──► record_generation ──► Langfuse span   +   usage meter ──► usage_log ──► /api/v1/metrics
```

## Requirement → implementation → measurement

| Requirement | Implementation | Measured by |
|---|---|---|
| Ingest PDF, DOCX, XLSX | `ingestion/loader.py`: `load_pdf`, `load_docx` (paragraphs and tables in order), `load_xlsx` (each row tagged with its column headers). Magic bytes, size limit, zip-bomb guard (uncompressed total ≤ 100 MB) | `tests/test_office_ingest.py` |
| Hybrid retrieval + reranking | BM25 + dense (Chroma locally, Qdrant in production), FPT or cross-encoder reranker (existing) | `eval/temporal_eval.py`: Recall@1/3/5/8, MRR, **nDCG@3/5/8** (new), retrieve p50/p95 |
| Department access control | `acl_label` per chunk (department), held labels resolved server-side per tenant. Enforced on 3 paths: Chroma `where`, Qdrant filter, BM25 `allows()` | `eval/acl_eval.py`: **Access Control Test Pass Rate** over 6 principals × 44 documents × 2 dates on each path. Currently 528/528 per path, 0 leaks. `tests/test_acl_eval.py` also checks that an injected leak is caught |
| Agent calls business tools | Read tools (search, summarize, compare, compliance, report) plus a write action `create_hr_request` | `GET /api/v1/actions/stats`: **tool_success_rate** = executed / (executed + failed) |
| Human approval for controlled actions | `agent/actions.py`: the agent only proposes. Approve/reject via `/api/v1/actions` with the admin key; state is in SQLite, so approval survives restarts. Conditional `UPDATE … WHERE status=…` means exactly one execution | `tests/test_actions.py` (8 concurrent approvers → 1 execution; handler failure recorded; anonymous users cannot propose) |
| Quality monitoring | Gold set of 208 questions, answer accuracy, faithfulness judge, CI gate against a baseline (existing) | `eval/temporal_eval.py`, `eval/ci_gate.py` in CI |
| Cost and latency monitoring | `src/usage.py`: per-request meter fed by `record_generation` (every LLM call), stored in `usage_log` for REST and WebSocket. Prices are configurable (`LLM_PRICES_JSON`), and unpriced models are reported, not costed at 0 | `GET /api/v1/metrics?hours=24`: requests, **P50/P95 latency**, error rate, LLM calls/failures, **tokens, cost, cost per request**, per intent |
| Tracing | Langfuse via OTLP (existing): one trace per request, spans for retrieval and generations | Langfuse dashboard |
| Regression tests, CI/CD | GitHub Actions: pytest (703 tests), gold-set validation, retrieval eval gated against `reports/eval_baseline.json` | `.github/workflows/ci.yml` |

## Design decisions

- **Approval is a database state machine, not a LangGraph `interrupt`.** An approval can come hours
  later, from another person, after a deploy. An in-process checkpoint would lose it; a row does not.
  The graph run ends with "proposal created", and execution happens in the approve request.
- **Action arguments come from the user's own words, not from an LLM.** `proposal_from_query` copies
  the request text and picks the department by keyword. Nothing can be hallucinated into a ticket,
  and the approver sees exactly what was asked.
- **The ACL metric has an oracle written from the policy**, not from `RetrievalContext`, and it runs
  the real stores (ephemeral Chroma, in-memory Qdrant). A test that compared the code with itself
  would always pass.
- **Accounting never breaks an answer.** A failed `usage_log` write is logged and dropped.

## Limits

- `create_hr_request` writes to a local `hr_requests` table that stands in for a real HRM or ticket
  system. Connecting a real system means a new handler with the same signature.
- The approver is identified by a header next to the shared admin key, which gives an audit trail
  but not per-person authorization. Per-approver keys or SSO are the next step.
- A WebSocket turn that fails mid-stream is not written to `usage_log`.
- List prices in `usage.DEFAULT_PRICES` must be checked against the provider's current pricing. FPT
  models have no default price and show up under `unpriced_models`.
