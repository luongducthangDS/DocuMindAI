# DocuMind AI 🏦

> **Enterprise Agentic RAG & Automated Compliance Platform for Vietnamese Banking Regulations**  
> *Production-Grade AI Portfolio Project — Senior / Staff AI Engineer Showcase*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-1C3C3C.svg?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6-FF6F00.svg)](https://www.trychroma.com)
[![React 19](https://img.shields.io/badge/React-19.0-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-6.0-646CFF.svg?logo=vite&logoColor=white)](https://vitejs.dev)
[![Test Suite](https://img.shields.io/badge/Tests-84%2F84%20Passing%20(100%25)-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 🌟 Executive Overview

**DocuMind AI** is an enterprise-grade AI assistant and regulatory compliance auditor tailored for the Vietnamese commercial banking and financial services sector. Built to solve high-stakes compliance and information retrieval challenges, DocuMind AI provides precise, statutory-grounded answers with paragraph-level legal citations, rejecting questions outside its regulatory domain and deterministically auditing credit conditions against State Bank of Vietnam (SBV) regulations.

### Key Capabilities

1. **State-Machine Agentic Workflow (LangGraph)**:
   - Dynamic intent routing (`simple_qa`, `compare`, `summarize`, `compliance_check`).
   - Deterministic multi-turn state preservation and session memory.
2. **Dual-Pass Hybrid Retrieval & Neural Reranking**:
   - **Sparse Retrieval**: Okapi BM25 for exact statutory term matching (e.g., *"Điều 14 Thông tư 18/2024"*).
   - **Dense Retrieval**: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) for semantic paraphrase understanding.
   - **Fusion & Reranking**: Reciprocal Rank Fusion (RRF) pool (top-20) re-scored by cross-encoder (`BAAI/bge-reranker-v2-m3`) to select the top-8 highest-precision chunks.
3. **Automated Statutory Compliance Engine (`compliance_check`)**:
   - Hybrid regex and LLM parameter extraction for quantifiable audit thresholds.
   - Verifiable pass/fail evaluations against curated regulatory standards:
     - **Debt-To-Income (DTI)** ceiling: $\le 60\%$ (Circular 39/2016/TT-NHNN).
     - **Unsecured Credit Card limit**: $\le 100$ million VNĐ (Circular 18/2024/TT-NHNN).
     - **Demand Deposit Rate cap**: $\le 0.5\%/\text{year}$ (Decision 1124/QĐ-NHNN).
     - **Short-Term Deposit Rate cap (1 to <6 months)**: $\le 4.75\%/\text{year}$ (Decision 1124/QĐ-NHNN).
     - **Unsecured Consumer Loan minimum income**: $\ge 5.0$ million VNĐ/month (Regulation QC CV-05/2023/NH).
4. **Resilient Multi-LLM Routing**:
   - **Primary**: Ultra-low latency Groq LLaMA-3.3-70B Versatile (~300 tokens/sec).
   - **Automatic Failover**: Google Gemini 2.0 Flash Lite when rate limits, quotas, or timeouts occur.
5. **Strict Grounding & Hallucination Prevention**:
   - Compulsory inline statutory citations `[N]` referencing Article, Circular number, and issuing institution.
   - Confident abstention: automatic domain boundary check and refusal when queries lack documentary support.
6. **Modern Full-Stack Experience**:
   - Reactive Dark-Mode React/Vite interface featuring WebSocket streaming, interactive citation drawers, compliance testing sandbox, and document explorer.

---

## 🏛️ System Architecture

### 1. Request Pipeline

```
User Query (HTTP / WebSocket)
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  [1] FastAPI Gateway                                                    │
│      Rate limiting (10 req/min) · CORS allowlist · GZip compression     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  [2] LangGraph Intent Router Node                                       │
│      LLM Classification (temp=0) → simple_qa │ compare │ summarize      │
│                                    │ compliance_check                  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
┌───────────────────────────────────┐   ┌─────────────────────────────────┐
│  [3A] Hybrid Retrieval Pipeline   │   │  [3B] Compliance Engine         │
│  ┌──────────────┐ ┌─────────────┐ │   │  • Regex/LLM parameter extract  │
│  │ Okapi BM25   │ │ Dense Vector│ │   │  • Threshold condition evaluate │
│  │ (Exact Lexical)│ (MiniLM-L12)│ │   │    (e.g., DTI <= 60%, cap <= 0.5)│
│  └──────┬───────┘ └──────┬──────┘ │   │  • Verified statutory citation  │
│         └─────── RRF ────┘        │   │    (pass / fail / missing data) │
│            Pool: top-20           │   └────────────────┬────────────────┘
│                 │                 │                    │
│                 ▼                 │                    │
│  ┌──────────────────────────────┐ │                    │
│  │ Cross-Encoder Neural Rerank  │ │                    │
│  │ BAAI/bge-reranker-v2-m3      │ │                    │
│  │ Select: top-8 high-precision │ │                    │
│  └──────────────┬───────────────┘ │                    │
└─────────────────┼─────────────────┘                    │
                  ▼                                      │
┌──────────────────────────────────────────────────┐     │
│  [4] Generator Node + Citation Verification      │     │
│      Primary: Groq LLaMA-3.3-70B (~300 tok/s)    │     │
│      Fallback: Gemini 2.0 Flash Lite (auto)      │     │
│      Grounding prompt: mandatory [N] citation    │     │
└─────────────────┬────────────────────────────────┘     │
                  │                                      │
                  └──────────────────┬───────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  [5] Delivery & Streaming Layer                                         │
│      Async WebSocket Generator (token-by-token) or REST JSON Response   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2. LangGraph State Machine Architecture

```mermaid
graph TD
    Start([User Input]) --> RouterNode[Router Node: Classify Intent]
    
    RouterNode -->|compliance_check| ComplianceNode[Compliance Engine: Rule Check]
    RouterNode -->|simple_qa / compare / summarize| RetrieveNode[Retrieve Node: Hybrid + Rerank]
    
    RetrieveNode --> GraderNode{Grader: Relevance Check}
    GraderNode -->|Relevant| GenerateNode[Generate Node: LLaMA-3.3-70B]
    GraderNode -->|Irrelevant / Out-of-Corpus| RefusalNode[Abstain Node: Domain Refusal]
    
    GenerateNode --> FailoverCheck{LLM Success?}
    FailoverCheck -->|Success| FormatNode[Format & Citations Node]
    FailoverCheck -->|Timeout / Quota| FallbackNode[Fallback Node: Gemini 2.0 Flash Lite]
    FallbackNode --> FormatNode
    
    ComplianceNode --> PersistNode[Memory Persist: SQLite Session Log]
    FormatNode --> PersistNode
    RefusalNode --> PersistNode
    PersistNode --> End([Client Response])
```

---

## 📚 Vietnamese Banking Regulatory Corpus

The bundled dataset consists of official statutory regulations and internal credit policies processed by our specialized legal chunker (`src/ingestion/chunker.py`), dividing documents at exact `Điều` (Article) and `Khoản` (Clause) boundaries:

| STT | Văn bản / Quyết định | Số hiệu | Cơ quan ban hành | Nội dung chính |
|:---:|---|---|---|---|
| 1 | **Quy định cho vay TCTD** | `39/2016/TT-NHNN` | Ngân hàng Nhà nước VN | Điều kiện cấp tín dụng, phương thức cho vay, lãi suất nợ quá hạn (tối đa 150%), lãi chậm trả (tối đa 10%). |
| 2 | **Quy định hoạt động thẻ ngân hàng** | `18/2024/TT-NHNN` | Ngân hàng Nhà nước VN | Điều kiện mở thẻ, phân loại thẻ, trần hạn mức thẻ tín dụng tín chấp (100 triệu VNĐ), hạn mức rút tiền mặt (tối đa 50%). |
| 3 | **Quy định tiền gửi tiết kiệm** | `48/2018/TT-NHNN` | Ngân hàng Nhà nước VN | Đối tượng gửi tiền, quy trình gửi/rút, quy tắc rút trước hạn một phần hưởng lãi suất không kỳ hạn, cơ chế tái tục tự động. |
| 4 | **Trần lãi suất tiền gửi & cho vay ưu tiên** | `1124/QĐ-NHNN` | Ngân hàng Nhà nước VN | Trần lãi suất VND không kỳ hạn & dưới 1 tháng (0.5%/năm), 1 đến dưới 6 tháng (4.75%/năm), trần cho vay ngắn hạn ưu tiên (4.0%/năm). |
| 5 | **Quy chế cho vay tiêu dùng tín chấp** | `CV-05/2023/NH` | Ngân hàng Thương mại | Điều kiện khách hàng cá nhân (20-60 tuổi), thu nhập tối thiểu 5 triệu VNĐ/tháng, trần tỷ lệ DTI $\le 60\%$, phê duyệt 24-48h. |
| 6 | **Biểu phí tài khoản, thẻ & ngân hàng số** | `88/2024/QĐ-NH` | Ngân hàng Thương mại | Miễn phí chuyển tiền 24/7, điều kiện miễn phí quản lý tài khoản (số dư $\ge 2$ triệu), chính sách miễn lãi thẻ tín dụng lên đến 45 ngày. |

---

## ⚖️ Automated Compliance Engine

Unlike standard RAG systems that rely solely on probabilistic generation, DocuMind AI features a **hybrid deterministic-symbolic compliance auditor**:

```python
# Example compliance check verification
from src.rag.compliance import check_compliance

# Case 1: Income verification for unsecured consumer loan
result = check_compliance("Khách hàng có mức thu nhập 8 triệu đồng/tháng có đủ điều kiện vay tín chấp không?")
# Output:
# {
#   "matched": True,
#   "criterion_id": "vay_tin_chap_thu_nhap",
#   "verdict": "pass",
#   "extracted_value": 8.0,
#   "explanation": "Khách hàng có mức thu nhập đạt điều kiện tối thiểu để xét duyệt hồ sơ vay tín chấp tiêu dùng theo Điều 6 Quy chế CV-05/2023/NH.",
#   "citation": {"so_hieu": "Quy chế CV-05/2023/NH", "dieu_khoan": "Điều 6"}
# }

# Case 2: Interest rate ceiling violation check
result = check_compliance("Áp dụng lãi suất 0.8% cho tiền gửi không kỳ hạn có hợp lệ không?")
# Output: verdict = "fail" (Exceeds SBV ceiling 0.5%/year according to Điều 1 Quyết định 1124/QĐ-NHNN)
```

---

## 📊 Evaluation

The evaluation harness ([`eval/`](eval/)) runs a 4-strategy retrieval ablation
(BM25 · dense · hybrid+RRF · hybrid+reranker) plus RAGAS generation metrics against a
hand-built question set with ground-truth answers and source chunk ids.

**Retrieval benchmark for the banking corpus is being rebuilt.** The current banking
question set ([`data/eval/test_questions.json`](data/eval/test_questions.json), 25 Q on
Thông tư 39/2016/TT-NHNN) was run against a 36-chunk pilot corpus where *every* strategy
saturates at hit-rate 1.0 / MRR 1.0 ([`reports/benchmark_results.json`](reports/benchmark_results.json))
— too small to discriminate configurations, so those figures are **not** reported as
evidence. A larger corpus and a harder question set (adversarial paraphrases, near-miss
negatives) are in progress.

**Historical benchmark (previous corpus).** The methodology and engineering findings
carry over — see [`EVALUATION.md`](EVALUATION.md). On a self-built 110-question set
(prior domain: university regulations): hybrid+reranker reached **hit_rate@K 0.95,
MRR 0.87, 100% out-of-corpus refusal**; RAGAS faithfulness ≈ 0.9 (5-question pilot).
That eval also drove real fixes — a language-mismatched reranker
(context_precision 0.66 → 0.83), a score-scale abstain bug, and ~15% generation
over-refusal.

| Stage | Mechanism |
|---|---|
| Sparse retrieval | Okapi BM25 — exact statutory term / article-ID matching |
| Dense retrieval | `paraphrase-multilingual-MiniLM-L12-v2` — semantic paraphrase matching |
| Fusion | Reciprocal Rank Fusion over both candidate lists (top-20) |
| Reranking | `BAAI/bge-reranker-v2-m3` cross-encoder → top-8 |
| Generation guardrails | out-of-corpus refusal · paragraph-level citation enforcement |

---

## 🛠️ Technology Stack

| Layer | Technology | Justification |
|---|---|---|
| **Agent Orchestration** | LangGraph 0.2 + LlamaIndex 0.14 | Explicit typed state transitions, modular unit-testability of nodes, and deterministic (non-LLM) intent routing. |
| **Primary LLM** | Groq (Llama 3.3 70B Versatile) | ~300 tokens/second generation speed on LPUs; ideal for responsive real-time streaming. |
| **Fallback LLM** | Google Gemini 2.0 Flash Lite | High concurrency, large context window, zero cold-start fallback when Groq hits TPM/RPM ceilings. |
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` | Compact (120MB), CPU-optimized, high multilingual semantic fidelity for Vietnamese text. |
| **Vector Store** | ChromaDB (Local Persistent) / Qdrant | Pluggable backend via `VECTOR_STORE_PROVIDER` without rewriting ingestion or retrieval queries. |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | State-of-the-art multilingual cross-encoder reranker for high-precision legal clause ranking. |
| **Backend Web API** | FastAPI + WebSockets + Pydantic v2 | Full async I/O, bidirectional streaming, automatic OpenAPI schema generation. |
| **Frontend UI** | React 19 + TypeScript + Vite | Dark-mode banking console, source preview drawer, compliance testing dashboard. |
| **Testing** | Pytest + Pytest-Cov + Pytest-Asyncio | 84/84 tests passing (verified offline) across units, integrations, and guardrails. |

---

## 🚀 Quickstart Guide

### Prerequisites
- Python 3.10 or 3.11 installed.
- Node.js 18+ (for frontend dev server).
- API Keys: At least one of `GROQ_API_KEY` or `GOOGLE_API_KEY`.

### 1. Environment Setup
Clone the repository and install backend dependencies:
```powershell
git clone https://github.com/luongducthangDS/DocuMindAI.git
cd DocuMindAI

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create your `.env` file:
```ini
GROQ_API_KEY=gsk_your_groq_key_here
GOOGLE_API_KEY=AIza_your_google_gemini_key_here
GENERATOR_PROVIDER=groq
EMBEDDING_PROVIDER=local
VECTOR_STORE_PROVIDER=chroma
```

### 2. Ingest Banking Corpus
Populate the ChromaDB vector database with the curated banking documents:
```powershell
python scripts/ingest_documents.py --source-dir data/raw/banking_docs --manifest data/raw/manifest.json --reset
```

### 3. Run the Application

#### Option A: One-Click PowerShell Launcher
```powershell
.\start.ps1
```
This launches both FastAPI on `http://localhost:8081` and Vite frontend on `http://localhost:5174`.

#### Option B: Manual Launch
```powershell
# Terminal 1 — Backend
uvicorn src.api.main:app --host 0.0.0.0 --port 8081 --reload

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
```

Visit the interactive web console at **`http://localhost:5174`**.  
Interactive API Swagger Docs: **`http://localhost:8081/docs`**.

---

## 🧪 Testing & Verification

Run the full automated test suite:
```powershell
# Run all tests with coverage report
pytest -v

# Run compliance-specific unit tests
pytest tests/test_compliance.py -v

# Run retrieval benchmark evaluation (smoke test)
python eval/run_evals.py --strategies dense rerank --retrieval-only --limit 5
```

---

## 📐 Architecture Decision Records (ADRs)

- **ADR-001: LangGraph State Machine over Chain-based Orchestrators**:
  - *Context*: Financial regulations require deterministic error recovery and auditable branching between informational QA and compliance audits.
  - *Decision*: Adopted LangGraph `StateGraph` with explicit typed state (`AgentState`).
  - *Outcome*: Enables granular node-level unit testing and transparent LangSmith trace logging.
- **ADR-002: Reciprocal Rank Fusion (RRF) Hybrid Retrieval**:
  - *Context*: User queries oscillate between natural language questions (*"vay tiền mua xe cần thu nhập bao nhiêu"*) and exact statutory lookups (*"Khoản 2 Điều 13 TT 39"*).
  - *Decision*: Implemented dual-pass Okapi BM25 + dense vector retrieval fused with reciprocal rank fusion prior to cross-encoder reranking.
  - *Outcome*: Recovers both exact statutory lookups and paraphrased natural-language queries in one path; the reranker then promotes the exact article toward rank 1. Retrieval ablation numbers: see [`EVALUATION.md`](EVALUATION.md).
- **ADR-003: Multi-Provider LLM Automatic Failover**:
  - *Context*: Free/standard tier LLM APIs occasionally return 429 rate limits or network timeouts.
  - *Decision*: Implemented primary execution on Groq LLaMA-3.3-70B with automatic fallback to Google Gemini 2.0 Flash Lite on 429/timeout.
  - *Outcome*: Rate-limit and transient-failure errors are absorbed by the fallback path instead of surfacing to the user.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
