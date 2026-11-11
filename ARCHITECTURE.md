# DocuMind AI — Architecture Overview

> Hệ thống tra cứu pháp luật lao động & BHXH Việt Nam, xây dựng trên Agentic RAG với LangGraph.

---

## 1. Tổng quan hệ thống

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                   │
│   POST /api/v1/query  │  WebSocket /api/v1/ws/{id}      │
│   Guardrails (injection check, PII redaction)           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│               LangGraph Agent (graph.py)                │
│                                                         │
│  contextualize → router → retrieve → temporal_filter   │
│       → research → grade → [reformulate] → answer      │
└─────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  RAG Pipeline                           │
│   Hybrid Search (Dense + BM25 + RRF) → Reranker        │
│   Vector Store: ChromaDB / Qdrant                       │
│   Embedding: Gemini Embedding API                       │
└─────────────────────────────────────────────────────────┘
```

---

## 2. LangGraph Flow — Chi tiết từng node

```
                    [START]
                       │
              contextualize_node
          (rewrite follow-up questions,
           restore Vietnamese diacritics)
                       │
                 router_node
          (classify intent: simple_qa /
           compare / summarize / report /
           compliance_check / unknown)
                       │
                retrieve_node
          (Hybrid Search: Dense + BM25,
           fused via RRF, top-20 candidates)
                       │
           temporal_filter_node
          (keep only clauses in force
           at the queried date)
                       │
              research_node  ◄─────────────────────┐
          (Agentic loop: LLM decides               │
           enough / fetch / search /               │
           versions / overview)                    │
                       │                           │
               grade_node                          │
          (LLM-as-judge: are retrieved             │
           chunks relevant to the query?)          │
                       │                           │
              relevant? ──No──► reformulate_node ──┘
                       │        (rewrite query,
                       │         max 1 retry)
                       Yes
                       │
           ┌───────────┴────────────┐
        simple_qa              compare / summarize
        answer_node            / report / compliance
           │                   specialized nodes
           └───────────┬────────────┘
                 persist_node
              (save to memory)
                  [END]
```

---

## 3. Agentic RAG Loop (research.py) — Trái tim của hệ thống

Đây là điểm khác biệt chính so với RAG thông thường.

**RAG thông thường:**
```
query → retrieve → answer
```

**DocuMind Agentic RAG:**
```
query → retrieve → LLM đánh giá → đủ? → answer
                        │
                        └─ chưa đủ → quyết định:
                               ├── fetch: lấy điều luật được dẫn chiếu trực tiếp
                               ├── search: tìm kiếm thêm với sub-query
                               ├── versions: lấy phiên bản cũ/mới để so sánh
                               └── overview: GraphRAG cho câu hỏi tổng quát
```

**Tại sao cần Agentic RAG?**

Pháp luật VN có cấu trúc dẫn chiếu chéo — Điều 23 nói "theo Điều 67", Điều 67 mới có con số thực. RAG 1 lượt chỉ lấy được Điều 23 nhưng không có câu trả lời. Agent đọc nội dung đã có, thấy dẫn chiếu, tự quyết định lấy thêm Điều 67.

**Kết quả đo được:** Multi-hop accuracy: 0.913 → 1.000 sau khi thêm agentic loop.

---

## 4. Hybrid Search — Cách kết hợp Dense + BM25

```
Query
  │
  ├──► Dense (Gemini Embedding) ──► Top-20 chunks (cosine similarity)
  │                                          │
  └──► BM25 (keyword matching)  ──► Top-20 chunks (BM25 score)
                                             │
                              RRF Fusion (Reciprocal Rank Fusion)
                              score = Σ 1/(k + rank_i), k=60
                                             │
                              Reranker (cross-encoder FPT/BGE)
                              → Final top-8 chunks
```

**Tại sao RRF thay vì weighted sum?**
- Dense tốt cho câu hỏi ngữ nghĩa ("nghỉ đẻ" → "chế độ thai sản")
- BM25 tốt cho exact match (số hiệu văn bản "45/2019/QH14", tên điều cụ thể)
- RRF không cần tune trọng số — rank-based nên robust với scale khác nhau giữa 2 nhánh

---

## 5. Self-Correction Loop (grader.py)

```
retrieved_chunks
       │
       ▼
  Heuristic check
  (best_score > threshold × 2.0?)
       │
   Yes ─► relevant=True (skip LLM, save cost)
       │
   No  ─► LLM-as-judge
          (Gemini: "chunks có đủ trả lời query không?")
               │
           relevant? ──No──► reformulate_node
                              (rewrite query differently,
                               retry retrieval once)
```

**Two-tier grading để tối ưu chi phí:**
- Chunk score cao rõ ràng → skip LLM judge hoàn toàn
- Score mơ hồ → mới dùng LLM phán xét
- Lỗi LLM judge → fail open (coi là relevant) để không bao giờ block câu trả lời

---

## 6. Temporal Filtering — Truy xuất đúng phiên bản luật

Pháp luật VN thay đổi liên tục — Luật BHXH 2014 bị thay bởi Luật 41/2024. Nếu không filter, hệ thống có thể trả lời bằng luật đã hết hiệu lực.

**Giải pháp:**
```
Mỗi chunk metadata có:
  effective_from_i: 20210101  (YYYYMMDD integer)
  effective_to_i:   99991231  (open-ended = còn hiệu lực)
  status: "in_force" | "superseded" | "not_yet_in_force"

Filter tại vector store (pre-filter, không phải post-filter):
  effective_from_i <= as_of_date < effective_to_i
```

**Tại sao pre-filter quan trọng?**
Post-filter = reranker tốn 5/8 slot cho điều đã hết hiệu lực → gold chunk bị đẩy ra khỏi top-8. Đo được: avg_chunks trong prompt = 2.87 với post-filter, vs 7+ với pre-filter.

---

## 7. Evaluation Results

| Metric | Before | After |
|--------|--------|-------|
| Recall@1 (legal_qa_200) | 0.585 | 0.790 |
| MRR (legal_qa_200) | 0.725 | 0.857 |
| Multi-hop accuracy | 0.913 | 1.000 |
| End-to-end accuracy | ~0.85 | ~0.99 |

**Các cải tiến chính tạo ra sự thay đổi:**
1. Thêm Reranker → Recall@1: 0.585 → 0.720
2. Chunk heading prefix (Chương/Mục) → +0.04
3. Agentic loop → multi-hop: 0.913 → 1.000
4. Graph expansion → +0.028 end-to-end

---

## 8. Kiến trúc Access Control (RBAC)

```
Request header: X-API-Key
        │
        ▼
  principal.py: resolve_context()
        │
        ▼
  RetrievalContext {
    tenant_id: "company_abc"
    acl_labels: {"hr_policy", "public"}
    as_of_date: "2024-07-01"
  }
        │
        ▼
  Vector store pre-filter:
    tenant_id IN ["public", "company_abc"]
    acl_label IN ["hr_policy", "public"]
```

Mỗi chunk được gán `tenant_id` + `acl_label` lúc ingest. Filter xảy ra trước retrieval — chunk không được phép không bao giờ chạm đến reranker hay LLM.

---

## 9. Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Agent Orchestration | LangGraph | State machine cho multi-step reasoning, dễ debug |
| LLM (Primary) | FPT AI (Qwen3, DeepSeek) | Vietnamese performance, cost |
| LLM (Fallback) | Gemini (3.8-flash, 3.5-flash-lite) | Reliability backup |
| Embedding | Gemini Embedding API | Best Vietnamese semantic search |
| Reranker | BGE-reranker-v2-m3 (local) / FPT Rerank API | Cross-encoder precision |
| Vector Store | ChromaDB (dev) / Qdrant (prod) | Swap via config, no code change |
| Sparse Search | BM25 (llama-index) | Exact match for legal doc numbers |
| Observability | Langfuse (custom OTLP) | Trace every agent step |
| API | FastAPI + WebSocket | REST + streaming |
| Frontend | React (TypeScript) | |

---

## 10. Cấu trúc thư mục

```
src/
├── agent/
│   ├── graph.py        # LangGraph workflow — entry point chính
│   ├── research.py     # Agentic RAG loop (enough/fetch/search/versions)
│   ├── tools.py        # 6 LangChain tools cho LLM
│   └── memory.py       # Short-term + long-term memory
├── rag/
│   ├── retriever.py    # Hybrid search: Dense + BM25 + RRF + Reranker
│   ├── generator.py    # LLM generation với mandatory citations
│   ├── grader.py       # Self-correction: relevance grading
│   ├── context.py      # RetrievalContext: tenant + ACL + temporal
│   └── temporal.py     # Point-in-time filtering logic
├── ingestion/
│   ├── chunker.py      # Legal document chunking (by Điều/article)
│   ├── manifest.py     # Document registry + amendment tracking
│   └── versions.py     # Clause version management
├── api/
│   ├── main.py         # FastAPI app + lifespan
│   ├── routes/query.py # REST endpoint + WebSocket handler
│   └── principal.py    # API key → RetrievalContext resolver
├── config.py           # Settings (pydantic-settings)
├── guardrails.py       # Prompt injection detection + PII redaction
└── langfuse_otel.py    # Custom observability (OTLP/HTTP)
```

---

## 11. Câu hỏi thường gặp trong interview

**Q: Tại sao chunking theo Điều thay vì fixed-size?**
A: Pháp luật VN có cấu trúc rõ ràng theo Điều/Khoản. Fixed-size chunk có thể cắt giữa một điều luật, mất ngữ cảnh. Mỗi Điều là một đơn vị ngữ nghĩa hoàn chỉnh — "Điều 54. Điều kiện hưởng lương hưu" nên là 1 chunk.

**Q: Tại sao cần reformulate_node?**
A: Khi grader đánh giá chunk không relevant, có 2 khả năng: (1) thông tin không có trong corpus, (2) query diễn đạt sai thuật ngữ. Reformulate rewrite query với phrasing khác, retry 1 lần. Giúp bắt được câu hỏi dùng từ dân dã ("rút BHXH") mà retrieval lần đầu trả về chunk lệch.

**Q: Làm sao biết Recall@1 tăng từ 0.585 lên 0.790?**
A: Xây dựng Golden Test Set 208 câu hỏi-đáp từ corpus thực. Mỗi câu có ground truth là chunk chứa câu trả lời. Recall@1 = tỷ lệ câu có ground truth chunk ở vị trí 1 trong kết quả retrieve. Chạy evaluation trước và sau mỗi thay đổi (reranker, chunk size, heading prefix...).

**Q: Tại sao không dùng langfuse-python SDK?**
A: Dependency conflict — langfuse >=4.x cần opentelemetry-api >=1.33.1, nhưng chromadb cần opentelemetry-exporter-otlp-proto-grpc==1.27.0. Thay vì downgrade ChromaDB hoặc chấp nhận conflict, tự implement OTLP/HTTP client bằng `requests` — đã có sẵn, không thêm dependency.

**Q: Tại sao BM25 cần _ScreenedRetriever?**
A: BM25 giữ một index trong memory cho toàn bộ corpus — không có khái niệm filter như vector store. Nếu không screen, BM25 trả về chunk của tenant khác rồi mới bị drop sau fusion. `_ScreenedRetriever` wrap BM25, drop chunk không được phép trước khi vào RRF — đảm bảo ACL không bao giờ bị bypass.
