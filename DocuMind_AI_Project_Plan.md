# DocuMind AI — Kế hoạch Project End-to-End

> Hệ thống RAG + Agentic AI chuyên về văn bản pháp luật Việt Nam  
> **Mục tiêu:** Portfolio project production-grade để apply vị trí AI Engineer

---

## 1. Tổng quan

### Mô tả

DocuMind AI là hệ thống hỏi đáp thông minh trên văn bản pháp luật Việt Nam, kết hợp RAG (Retrieval-Augmented Generation) và Multi-step Agent. Người dùng upload hoặc truy vấn các văn bản như Luật, Nghị định, Thông tư — hệ thống trả lời chính xác kèm trích dẫn nguồn, tóm tắt tự động và xuất báo cáo PDF.

### Tagline

*"Hỏi bất kỳ điều gì về pháp luật Việt Nam — nhận câu trả lời trong vài giây, có nguồn, có điều khoản cụ thể."*

### Lý do chọn domain pháp luật

- Dữ liệu 100% công khai và hợp lệ (vbpl.vn — Bộ Tư pháp)
- Dataset sẵn có trên Hugging Face (`th1nhng0/vietnamese_legal_corpus`, 6GB)
- Pain point thực tế: doanh nghiệp, kế toán, HR đều cần tra cứu luật nhanh
- Văn bản luật dài, có cấu trúc rõ (Điều, Khoản, Mục) → ideal cho RAG demo

---

## 2. Tech Stack

| Layer | Công nghệ | Lý do chọn |
|---|---|---|
| Core RAG | LlamaIndex | Tốt hơn LangChain cho RAG thuần, hybrid search built-in |
| Agent / Orchestration | LangGraph | Được nhắc nhiều nhất trong JD 2026, multi-step reasoning |
| LLM chính | Groq + Llama 3.1 70B | Miễn phí, nhanh nhất hiện tại (~300 token/s) |
| LLM dự phòng | Gemini 1.5 Flash | Rẻ, context window lớn (1M token) |
| Embedding | bge-m3 | Tốt nhất cho tiếng Việt, multilingual |
| Vector DB | ChromaDB → pgvector | Chroma cho dev, pgvector khi production |
| Backend | FastAPI + WebSocket | Async, streaming response, standard trong JD |
| Frontend | Streamlit (MVP) | Nhanh nhất để demo, không cần frontend skill |
| Deploy | Docker + Railway | Free tier đủ dùng, CI/CD tự động qua GitHub |
| Observability | LangSmith (free) | Trace mọi LLM call, đo latency, cost |
| Eval | RAGAS | Framework chuẩn đo chất lượng RAG |
| PDF export | ReportLab | Sinh báo cáo PDF đẹp, pure Python |

### Chi phí ước tính

```
Groq API:     Miễn phí (60 req/phút)
Gemini Flash: ~$0.075/1M token
Railway:      $5 credit/tháng (đủ cho demo)
ChromaDB:     Miễn phí (self-hosted)
LangSmith:    Miễn phí (developer tier)
────────────────────────────────────
Tổng:         ~$0–5/tháng khi development
```

---

## 3. Cấu trúc thư mục

```
documind-ai/
├── data/
│   ├── raw/                    # văn bản crawl về (JSON)
│   ├── processed/              # sau khi clean + chunk
│   └── eval/                   # test set 50 câu hỏi
├── src/
│   ├── ingestion/
│   │   ├── crawler.py          # crawl vbpl.vn
│   │   ├── loader.py           # load HF dataset + PDF
│   │   ├── cleaner.py          # làm sạch văn bản
│   │   └── chunker.py          # chunk theo Điều/Khoản
│   ├── rag/
│   │   ├── embedder.py         # bge-m3 embedding
│   │   ├── retriever.py        # hybrid search + reranker
│   │   └── generator.py        # LLM generation + citation
│   ├── agent/
│   │   ├── graph.py            # LangGraph workflow
│   │   ├── tools.py            # search, summarize, compare, export
│   │   └── memory.py           # conversation + long-term memory
│   ├── report/
│   │   └── generator.py        # ReportLab PDF export
│   └── api/
│       ├── main.py             # FastAPI app
│       ├── routes/
│       └── schemas.py
├── frontend/
│   └── app.py                  # Streamlit UI
├── tests/
│   ├── test_ingestion.py
│   ├── test_rag.py
│   └── test_agent.py
├── eval/
│   └── ragas_eval.py           # RAGAS evaluation script
├── .github/
│   └── workflows/
│       └── deploy.yml          # CI/CD GitHub Actions
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 4. Nguồn dữ liệu

### Nguồn chính — Hugging Face (dùng ngay, tuần 1)

```python
from datasets import load_dataset

ds = load_dataset("th1nhng0/vietnamese_legal_corpus")
# 6GB raw text, structured JSON
# Bao gồm: Luật, Nghị định, Thông tư, Quyết định
```

**Tại sao dùng trước:** Có ngay data trong 30 phút, tập trung build pipeline thay vì tốn thời gian crawl.

### Nguồn bổ sung — vbpl.vn (crawl tuần 2)

Website của Bộ Tư pháp, dữ liệu công khai 100%, không có điều khoản cấm crawl phi thương mại.

```python
# crawl_vbpl.py
import requests, time, json
from bs4 import BeautifulSoup
from pathlib import Path

BASE = "https://vbpl.vn"
HEADERS = {"User-Agent": "DocuMind-Research-Bot/1.0 (academic, non-commercial)"}
DELAY = 2  # giây — lịch sự với server

def get_doc_list(van_ban_type="luat", page=1):
    url = f"{BASE}/TW/Pages/vbpq-toanvan.aspx?type={van_ban_type}&page={page}"
    r = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    return [a["href"] for a in soup.select("div.title a")]

def get_doc_content(url):
    r = requests.get(BASE + url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    return {
        "title":   soup.select_one("h1.doc-title").get_text(strip=True),
        "content": soup.select_one("div.toanvan").get_text(separator="\n", strip=True),
        "url":     BASE + url,
        "source":  "vbpl.vn"
    }

def crawl(van_ban_type, max_pages=10):
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    for page in range(1, max_pages + 1):
        for href in get_doc_list(van_ban_type, page):
            doc = get_doc_content(href)
            fname = f"data/raw/{doc['title'][:60].replace('/', '_')}.json"
            Path(fname).write_text(json.dumps(doc, ensure_ascii=False))
            time.sleep(DELAY)
```

**Loại văn bản ưu tiên crawl:**
- Luật Doanh nghiệp 2020 + sửa đổi 2025
- Luật Thuế Thu nhập Doanh nghiệp
- Luật Lao động + Nghị định hướng dẫn
- Luật Bảo vệ Dữ liệu Cá nhân (PDPA)
- Nghị định về chuyển đổi số, AI

---

## 5. Chunking Strategy

Văn bản luật có cấu trúc phân cấp rõ ràng — **chunk theo cấu trúc pháp lý, không theo số ký tự**.

```
Văn bản
└── Chương I, II, III...
    └── Mục 1, 2, 3...
        └── Điều 1, 2, 3...   ← đơn vị chunk chính
            └── Khoản 1, 2...
                └── Điểm a, b, c...
```

```python
# chunker.py
import re

def chunk_by_dieu(text: str, doc_meta: dict) -> list[dict]:
    """
    Chunk văn bản luật theo từng Điều.
    Mỗi chunk giữ nguyên ngữ nghĩa pháp lý đầy đủ.
    """
    dieu_pattern = re.compile(
        r'(Điều\s+\d+[a-z]?\.\s+.*?)(?=Điều\s+\d+[a-z]?\.|$)',
        re.DOTALL
    )
    chunks = []
    for match in dieu_pattern.finditer(text):
        chunk_text = match.group(1).strip()
        if len(chunk_text) < 50:
            continue
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "source_url":  doc_meta["url"],
                "title":       doc_meta["title"],
                "doc_type":    doc_meta.get("type", "unknown"),
                "dieu_header": chunk_text.split("\n")[0][:80],
                "char_count":  len(chunk_text),
            }
        })
    return chunks
```

---

## 6. RAG Pipeline

### Retrieval: Hybrid Search + Reranker

```python
# retriever.py
from llama_index.core import VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

def build_hybrid_retriever(index, nodes, top_k=10):
    vector_retriever = index.as_retriever(similarity_top_k=top_k)
    bm25_retriever   = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)

    # Kết hợp dense (semantic) + sparse (keyword) search
    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=5,
        num_queries=1,
        mode="reciprocal_rerank",  # RRF — tốt hơn simple merge
    )
```

### Generation: Citation bắt buộc

Mỗi câu trả lời phải kèm trích dẫn cụ thể:

```
Câu trả lời: Theo Điều 5, Khoản 2 Luật Doanh nghiệp 2020...

Nguồn:
[1] Luật Doanh nghiệp 2020 — Điều 5. Quyền của doanh nghiệp
    https://vbpl.vn/...
[2] Nghị định 01/2021/NĐ-CP — Điều 12. Đăng ký doanh nghiệp
    https://vbpl.vn/...
```

---

## 7. LangGraph Agent

### Cấu trúc Graph

```
User Query
    │
    ▼
Router Node ──────────────────────────────────┐
    │                                          │
    ├─ simple_qa ──► Retriever ──► Generator   │
    │                                          │
    ├─ compare ───► Multi-retrieve ──► Compare │
    │                                          │
    ├─ summarize ─► Summarize Agent            │
    │                                          │
    └─ report ────► Report Generator ──► PDF   │
                                               │
                         ◄─────────────────────┘
                         Response + Sources
```

### Tools của Agent

```python
# tools.py
tools = [
    search_legal_docs,      # tìm kiếm trong vector DB
    summarize_document,     # tóm tắt 1 văn bản dài
    compare_documents,      # so sánh 2+ văn bản
    extract_articles,       # trích xuất các Điều liên quan
    generate_pdf_report,    # tạo báo cáo PDF
    get_document_metadata,  # lấy thông tin văn bản (ngày ban hành, cơ quan...)
]
```

### Memory

```python
# memory.py
# Short-term: lưu trong session (in-memory)
# Long-term:  lưu summary vào SQLite, tái sử dụng qua session
```

---

## 8. API Endpoints

```
POST /api/v1/query          # hỏi đáp (streaming WebSocket)
POST /api/v1/upload         # upload văn bản mới
GET  /api/v1/documents      # danh sách văn bản đã index
POST /api/v1/report/create  # tạo báo cáo PDF tự động
GET  /api/v1/health         # health check (dùng cho monitoring)
GET  /api/v1/metrics        # latency, query count, error rate
```

---

## 9. Kế hoạch 7 tuần

### Tuần 1 — Nền tảng + Ingestion Pipeline

**Mục tiêu:** App chạy được local, query được HF dataset

**Tasks:**
- [ ] Tạo GitHub repo, setup virtual env, cấu trúc folder
- [ ] Load `th1nhng0/vietnamese_legal_corpus` từ Hugging Face
- [ ] Implement `chunker.py` — chunk theo Điều/Khoản
- [ ] Setup ChromaDB persistent, embed bằng bge-m3
- [ ] FastAPI endpoint: `POST /upload`, `GET /documents`
- [ ] Unit test cho ingestion pipeline (`pytest tests/test_ingestion.py`)
- [ ] `.env.example` với tất cả keys cần thiết

**Deliverable:** `python ingest.py` chạy được, 100+ văn bản đã index vào Chroma

---

### Tuần 2 — RAG Core + Citation

**Mục tiêu:** Hỏi đáp cơ bản có trích dẫn nguồn chính xác

**Tasks:**
- [ ] Hybrid search: vector + BM25 kết hợp RRF
- [ ] Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- [ ] Query expansion: LLM viết lại câu hỏi trước khi retrieve
- [ ] Citation formatter: mỗi answer kèm Điều + URL nguồn
- [ ] Streaming response qua FastAPI StreamingResponse
- [ ] RAGAS baseline: chạy eval lần đầu, ghi lại số
- [ ] Crawl thêm văn bản từ vbpl.vn (50–100 văn bản)

**Deliverable:** Query "Điều kiện thành lập công ty TNHH?" → trả lời + nguồn Điều X

---

### Tuần 3 — LangGraph Agent

**Mục tiêu:** Agent multi-step, xử lý câu hỏi phức tạp

**Tasks:**
- [ ] Thiết kế graph: Router → Retriever → Reasoner → Formatter
- [ ] Implement 6 tools (search, summarize, compare, extract, pdf, metadata)
- [ ] Conversation memory (in-session)
- [ ] Long-term memory (SQLite, cross-session summary)
- [ ] Streaming token-by-token qua WebSocket
- [ ] Test: "So sánh quy định về thuế TNDN trong Luật 2008 và sửa đổi 2025"
- [ ] LangSmith: bật trace, xem latency từng step

**Deliverable:** Agent xử lý được câu hỏi 3+ bước, stream response ra UI

---

### Tuần 4 — Auto Report Generator

**Mục tiêu:** Tự động sinh báo cáo PDF đẹp

**Tasks:**
- [ ] Template báo cáo: ReportLab (logo, header, section, bảng, footnote)
- [ ] Report Agent: tự động tóm tắt + so sánh nhiều văn bản
- [ ] Scheduled report: APScheduler chạy hàng ngày/tuần
- [ ] Export: PDF + Markdown
- [ ] Email delivery: SMTP gửi báo cáo (dùng Gmail SMTP miễn phí)
- [ ] Demo use case: upload Luật Lao động + Nghị định → báo cáo tóm tắt điểm mới

**Deliverable:** Click "Tạo báo cáo" → PDF đẹp download được trong 30 giây

---

### Tuần 5 — Production: Docker + Deploy + Monitoring

**Mục tiêu:** App chạy public, có monitoring thật — đây là điểm phân biệt fresher vs mid-level

**Tasks:**
- [ ] `Dockerfile` + `docker-compose.yml` (app + chromadb + redis)
- [ ] GitHub Actions: push main → auto deploy lên Railway
- [ ] LangSmith dashboard: trace mọi LLM call, alert khi error rate > 5%
- [ ] Structured logging: `loguru` → log file với timestamp, user_id, query, latency
- [ ] Rate limiting: `slowapi` (10 req/phút free tier)
- [ ] Error handling: retry logic, fallback LLM (Gemini nếu Groq timeout)
- [ ] Health check: `GET /health` trả về status từng service
- [ ] Environment: staging + production tách biệt

**Deliverable:** `https://documind.up.railway.app` accessible, LangSmith có data thật

---

### Tuần 6 — Eval Harness + Optimization

**Mục tiêu:** Đo được chất lượng RAG bằng số, optimize bottleneck

**Tasks:**
- [ ] Tạo test set: 50 câu hỏi pháp lý + expected answers (tự viết tay)
- [ ] RAGAS metrics: faithfulness, answer_relevancy, context_recall, context_precision
- [ ] Target: faithfulness ≥ 0.80, answer_relevancy ≥ 0.75
- [ ] Regression test: PR phải pass eval threshold mới được merge
- [ ] Redis cache: cache query giống nhau → giảm latency + cost 60%
- [ ] Async processing: upload lớn xử lý background (FastAPI BackgroundTasks)
- [ ] Optimize chunk size: thử 256/512/1024, so sánh RAGAS score

**Deliverable:** `ragas_eval.py` chạy tự động, report với số liệu thật

---

### Tuần 7 — Portfolio + README + Demo

**Mục tiêu:** Sẵn sàng để interviewer xem và tự thử

**Tasks:**
- [ ] README: architecture diagram, tech choices + lý do, benchmark numbers, screenshots
- [ ] Demo video 2–3 phút: upload → query → agent reason → cite → export PDF
- [ ] Hugging Face Space: deploy public demo (interviewer tự thử không cần account)
- [ ] LinkedIn post: "Tôi build RAG + Agent như thế nào — lessons learned"
- [ ] Số liệu để trình bày:
  - Latency trung bình: X ms/query
  - RAGAS faithfulness: X.XX
  - Cost per query: ~$0.00X
  - Corpus size: X văn bản, X chunks

**Deliverable:** GitHub repo public, HF Space chạy được, README có benchmark

---

## 10. Evaluation Framework

### Metrics mục tiêu

| Metric | Mục tiêu | Ý nghĩa |
|---|---|---|
| Faithfulness | ≥ 0.80 | Answer có căn cứ từ context, không hallucinate |
| Answer Relevancy | ≥ 0.75 | Answer đúng với câu hỏi |
| Context Recall | ≥ 0.70 | Retrieve đủ context cần thiết |
| Latency (P95) | ≤ 5s | Tốc độ chấp nhận được |
| Cost per query | ≤ $0.01 | Sustainable khi scale |

### Chạy eval

```bash
python eval/ragas_eval.py \
  --test-set data/eval/test_questions.json \
  --output reports/ragas_report.json
```

---

## 11. Deliverables cuối cùng

Đây là những thứ bạn show khi phỏng vấn:

| # | Deliverable | Mô tả |
|---|---|---|
| 1 | GitHub repo public | README đẹp, architecture diagram, RAGAS benchmark |
| 2 | App deploy Railway | URL public, interviewer tự thử được |
| 3 | LangSmith dashboard | Show trace thật, latency, cost mỗi query |
| 4 | RAGAS eval report | Số liệu chất lượng RAG (faithfulness, relevancy) |
| 5 | Docker image | `docker pull` là chạy được ngay |
| 6 | Demo video | 2–3 phút, upload → chat → export PDF |
| 7 | HF Space | Demo public không cần tài khoản |

---

## 12. Câu hỏi phỏng vấn bạn sẽ trả lời được

Sau khi hoàn thành project này, bạn tự tin trả lời:

**Về RAG:**
- *"Tại sao dùng hybrid search thay vì chỉ vector search?"* → BM25 tốt với keyword exact match (tên Điều, số Nghị định), vector search tốt với semantic. Kết hợp qua RRF cho kết quả tốt hơn cả hai.
- *"Chunking strategy của bạn là gì?"* → Chunk theo Điều/Khoản thay vì số ký tự cố định, vì văn bản luật có cấu trúc ngữ nghĩa rõ ràng theo từng Điều.
- *"RAG của bạn chính xác cỡ nào?"* → RAGAS faithfulness 0.83, answer relevancy 0.79, đây là test set 50 câu tự tạo.

**Về Agent:**
- *"Khi nào dùng LangGraph thay vì LangChain?"* → LangGraph cho phép định nghĩa graph rõ ràng với state machine, dễ debug và có thể loop back. Phù hợp khi agent cần nhiều bước quyết định.

**Về Production:**
- *"Làm sao bạn monitor hệ thống?"* → LangSmith trace mọi LLM call, structured logging với loguru, health check endpoint, alert khi error rate vượt ngưỡng.
- *"Bạn handle lỗi LLM timeout như thế nào?"* → Retry 2 lần, sau đó fallback sang Gemini Flash. Tất cả được log lại.

---

## 13. References

- **Dataset:** [th1nhng0/vietnamese_legal_corpus](https://huggingface.co/datasets/th1nhng0/vietnamese_legal_corpus)
- **Nguồn crawl:** [vbpl.vn](https://vbpl.vn) — Cơ sở dữ liệu quốc gia văn bản pháp luật
- **Embedding:** [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- **Eval:** [RAGAS](https://docs.ragas.io)
- **Observability:** [LangSmith](https://smith.langchain.com)
- **Deploy:** [Railway](https://railway.app)

---

*Cập nhật lần cuối: Tháng 5/2026*
