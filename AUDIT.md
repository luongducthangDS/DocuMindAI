# AUDIT.md — DocuMind AI

> **Viết cho:** Ted (chủ dự án), người duyệt kế hoạch trước khi sửa code.
> **Ngày audit:** 2026-09-17 · **Nhánh:** `feature/labor-pivot` @ `a7e37e1`
> **Nguyên tắc:** mọi con số trong tài liệu này hoặc do em chạy lại tại chỗ, hoặc trích từ
> file report có sẵn trong repo (ghi rõ đường dẫn). Số nào không kiểm chứng được thì nói rõ
> là không kiểm chứng được.

---

## 1. Phạm vi và phương pháp

Đã đọc: toàn bộ `src/` (12.7k dòng Python), `eval/`, `tests/`, `scripts/`, `docs/spec/`,
`docs/decisions/`, `tasks/todo.md`, `Dockerfile`, `render.yaml`, `frontend/src/main.tsx`,
cấu hình `.env`/`.env.example`, và lịch sử git 15 commit gần nhất.

Đã chạy để kiểm chứng (không lấy số từ tài liệu):

| Kiểm chứng | Lệnh | Kết quả thực |
|---|---|---|
| Test suite | `pytest -q --no-cov` | **166 passed / 166**, 48.2s |
| Corpus thực trong index | `chromadb.PersistentClient('data/chroma_db')` | **1146 chunks**, 19 văn bản, 1024-dim, nhãn `AITeamVN/Vietnamese_Embedding` |
| Schema metadata thực | `col.get(include=['metadatas'])` | 23 khoá phủ 1146/1146 chunk (xem §3.3) |
| Lint | `ruff check src tests eval scripts` | **334 finding** (src 116, tests 53, eval 69, scripts 96) |
| CI | `ls .github/workflows` | **không tồn tại** |
| Gold set | `data/eval/*.json` | 25 câu **ngân hàng** + 30 câu **lao động (temporal)** |

---

## 2. Đính chính premise trước khi audit

Đề bài mô tả hệ thống bằng 4 dữ kiện không còn đúng với repo hiện tại. Cần thống nhất lại
trước khi đo, vì cả 5 phase đều neo vào những con số này:

| Đề bài nói | Thực tế trong repo | Bằng chứng |
|---|---|---|
| "RAGAS faithfulness 0.87, failure rate 12.9% trên 50 câu" | **Không tái lập được.** Report sinh ra số này đã bị xoá vì đo trên corpus 36 chunk ngân hàng (chính Ted xoá và ghi lý do trong commit message) | commit `a7e37e1` |
| "Docker trên Railway" | Railway đã gỡ sạch khỏi repo; deploy hiện tại là Render (backend) + Vercel (frontend) | `render.yaml`, `frontend/vercel.json`, `CLAUDE.md` |
| "5 intent routes" | 6 intent (`simple_qa`, `compare`, `summarize`, `report`, `compliance_check`, `unknown`) — nhưng `unknown` không có nhánh riêng, nó cũng đi vào retrieval | `src/agent/graph.py:745-753` |
| "LangGraph StateGraph" | Đúng, nhưng graph có **12 node** (nhiều hơn mô tả), gồm `do_contextualize`, `do_temporal_filter`, `do_grade`, `do_reformulate` | `src/agent/graph.py:770-804` |

**Hệ quả quan trọng nhất:** hệ thống hiện **không có baseline đo được**. Phase 1 không phải
"mở rộng bộ đo", mà là "dựng baseline lần đầu cho corpus lao động".

---

## 3. Kiến trúc thực tế (as-built)

### 3.1 Đường đi một request `POST /api/v1/query`

```
Client ──► FastAPI (src/api/main.py)
             │  CORS allowlist · GZip · log middleware
             │  ⚠ rate limiter KHỞI TẠO nhưng KHÔNG ĐƯỢC GẮN (xem R4)
             ▼
        routes/query.py
             │ ① check_prompt_injection()  ← regex, fail-open
             │ ② ensure_rag_initialized()  ← lazy, có lock
             ▼
        run_agent()  (src/agent/graph.py)
             │
        ┌────▼──────────── LangGraph StateGraph ─────────────────────────┐
        │ do_contextualize   viết lại câu hỏi follow-up (Groq→Gemini)    │
        │        ▼                                                       │
        │ router             keyword fast-path → LLM nếu câu dài > 6 từ  │
        │        ├─ simple_qa / unknown ─► do_retrieve                   │
        │        ├─ compare  ─► do_compare  ─┐                           │
        │        ├─ summarize─► do_summarize ├─ fallback ─► do_retrieve  │
        │        ├─ report   ─► do_report   ─┤   (+answer)               │
        │        └─ compliance─► do_compliance┘                          │
        │                                                                │
        │ do_retrieve  ─► do_temporal_filter ─► do_grade                 │
        │                                          │                     │
        │                     ┌── irrelevant ──────┤ (tối đa 2 vòng)     │
        │                     ▼                    ▼                     │
        │              do_reformulate          do_answer                 │
        │                     └────► do_retrieve   │                     │
        │                                          ▼                     │
        │                                     do_persist ─► END          │
        └────────────────────────────────────────────────────────────────┘
             │ ③ validate_citations()  ← cắt [N] vượt chunk_count
             ▼
        QueryResponse {answer, sources, steps[], used_llm, latency_ms, ...}
```

### 3.2 Tầng retrieval (`src/rag/retriever.py`)

```
query ─┬─► Dense  (Vietnamese_Embedding 1024d, Chroma/Qdrant)  top_k=20 ─┐
       └─► BM25   (llama-index BM25Retriever, corpus nạp từ store)  20 ──┤
                                                                         ▼
                                        QueryFusionRetriever (RRF, k=60)
                                                   ▼
                        Cross-encoder BAAI/bge-reranker-v2-m3 → top_n=8
                                                   ▼
                        _reranker_active=True ⇒ ngưỡng abstain 0.05
                        _reranker_active=False ⇒ ngưỡng 0.0 (TẮT GATE)
```

Ba lớp fallback lồng nhau: reranker fail → `_TruncatedRetriever` (cắt rank RRF);
retriever fail → `retrieve_direct_chroma()` (dense thuần, top-5); LLM fail →
Groq → Gemini (3 key × 3 model, round-robin) → OpenAI-compatible → extractive (không LLM).

### 3.3 Tầng dữ liệu — điểm mạnh nhất của dự án

Collection `documind_legal`: **1146 chunk / 19 văn bản**, chunk theo **Điều**, metadata
**cấp điều/khoản** đầy đủ 23 khoá trên 100% chunk:

```
clause_uid        45-2019-QH14__d1              ← khoá định danh điều/khoản
version_id        45-2019-QH14__d1__v2021-01-01 ← khoá định danh BẢN của điều đó
effective_from    2021-01-01
effective_to      9999-12-31   ← sentinel mở (Chroma không chứa None)
status            in_force | repealed
superseded_by     ""           ← trỏ tới version thay thế
amended_by_doc    ""           ← văn bản sửa đổi
so_hieu / doc_id / title / doc_type / dieu / dieu_header / dieu_tieu_de / khoan / diem
consolidated_from / ngay_ban_hanh / source / source_url / verify_status / char_count
```

Phân bố: BLLĐ 45/2019 (227), NĐ 145/2020 (109), Luật BHXH 41/2024 (146), Luật BHXH 58/2014
(125), NĐ 374/2025 (103), Luật Việc làm 38/2013 (62) & 74/2025 (55), NĐ 158/2025 (60),
TT 59/2015 (58), NĐ 28/2015 (44), NĐ 115/2015 (37), TT 12/2025 (24), TT 10/2020 (22),
NĐ 159/2025 (20), NĐ 134/2015 (18), NĐ 293/2025 (11), NĐ 135/2020 (10), NĐ 74/2024 (8),
TT 11/2025 (7).

Nguồn sự thật: `docs/corpus/corpus_manifest.yaml` → `src/ingestion/manifest.py` → chunker →
index. Chỉ document `verify_status=VERIFIED*` mới được vào index (CORPUS_SPEC §2).

### 3.4 Số liệu thật đang có trong repo

`reports/temporal_eval.json` — 30 câu nhạy thời điểm, 3 nhánh dùng chung một lần retrieval:

| Metric | no_temporal | prompt_only | **temporal_filter** |
|---|---|---|---|
| context_gold | 73.3% | 73.3% | **80.0%** |
| context_distractor | 76.7% | 76.7% | **0.0%** |
| context_clean | 20.0% | 20.0% | **80.0%** |
| **answer_accuracy** | 36.7% | 46.7% | **83.3%** |

`reports/embedding_ab.json` — A/B model embedding trên cùng 30 câu, chỉ đổi 1 biến:
`paraphrase-multilingual-MiniLM-L12-v2` final@8 = **0.821** → `AITeamVN/Vietnamese_Embedding`
final@8 = **1.000** (1146 chunk, pool 20, rerank bge-v2-m3 → top 8).

Đây là hai phép đo **đạt chuẩn portfolio**: gold viết tay từ toàn văn và commit **trước**
khi chạy hệ thống lần đầu (`5723245` đứng trước `250c8a3`), có nhóm control để bắt lỗi lọc
quá tay, có `failure_analysis.md`. Phần còn lại của dự án chưa đạt mức này.

---

## 4. Điểm yếu và rủi ro

Xếp theo mức độ. Mỗi mục có bằng chứng file:dòng, tác động, và phase xử lý đề xuất.

### R1 — 🔴 Nghiêm trọng: hệ thống vẫn "nghĩ" mình là trợ lý ngân hàng

Corpus đã pivot sang lao động/BHXH từ 15 commit trước, nhưng **4 thành phần chưa pivot**:

| Thành phần | Bằng chứng | Nội dung sai |
|---|---|---|
| System prompt | `src/rag/generator.py:26` | *"Bạn là trợ lý tra cứu tài liệu ngân hàng (quy định, biểu phí, sản phẩm...)"* |
| Compliance engine | `data/compliance/criteria.json` | **5/5 tiêu chí là ngân hàng**: DTI ≤60%, hạn mức thẻ 100tr, trần lãi suất NHNN, thu nhập vay tín chấp |
| Gold set chính | `data/eval/test_questions.json` | **25 câu ngân hàng** (Thông tư 39/2016, 18/2024...) — trong khi `CLAUDE.md` ghi file này "đang rỗng" |
| Thông điệp từ chối | `src/api/routes/query.py:91,232` | *"Vui lòng đặt câu hỏi về tài liệu ngân hàng."* |

**Tác động:** intent `compliance_check` là đường đi thẳng tới **câu trả lời sai có thẩm quyền**:
node trả `✅ Đạt điều kiện` kèm citation "Thông tư 39/2016/TT-NHNN — Điều 7" cho một corpus
**không hề chứa văn bản đó**. Không có LLM nào chặn được — đây là rule engine deterministic
(`src/rag/compliance.py`), verdict đi thẳng ra UI kèm dấu ✅.

Ngoài ra: nếu Phase 1 chạy eval trên `test_questions.json` hiện tại, **mọi số sẽ ≈ 0** và
không phản ánh gì về hệ thống.

**Đã có kế hoạch nhưng chưa làm:** `tasks/todo.md` T21–T23 (criteria lao động), T24 (gold set).

### R2 — 🔴 Nghiêm trọng: không có baseline, và bộ metric hiện tại đo bằng proxy

1. **Không có baseline**: `reports/benchmark_results.json` đã xoá (`a7e37e1`). Con số
   0.87/12.9% trong đề bài thuộc về corpus khác, model khác, prompt khác.
2. **hit_rate/MRR đo bằng token-F1, không dùng gold chunk**: `eval/metrics.py:99-106` coi một
   chunk là "relevant" nếu token-F1 với `ground_truth` ≥ **0.15**. Đây là ngưỡng rất lỏng —
   một chunk cùng chủ đề nhưng **sai bản hiệu lực** vẫn tính là hit. Trong khi đó gold set
   temporal **đã có `source_clause` = `clause_uid` thật** (`74-2024-ND-CP__d3`) nhưng
   `metrics.py` không dùng tới.
3. **Không có citation correctness**: `src/guardrails.py:103` chỉ kiểm `1 ≤ N ≤ chunk_count`.
   Một câu trả lời cite `[3]` trong khi nội dung lấy từ chunk 5 vẫn pass. Không kiểm điều/khoản
   nêu trong câu chữ ("Điều 107") có tồn tại trong context hay không.
4. **Không có over-refusal metric**: `ooc_refusal_rate` chỉ đo "câu OOC có bị từ chối không",
   không đo "câu **trong** phạm vi có bị từ chối oan không" — mà đây mới là failure mode
   nguy hiểm khi bật ngưỡng abstain.
5. **RAGAS không chạy được thường xuyên**: judge là Gemini free tier, 3 req/61s/pair
   (`CLAUDE.md`). 200 câu × 4 strategy × 4 metric là bất khả thi trong CI.

### R3 — 🟠 Cao: WebSocket là một hệ thống thứ hai, đi vòng qua nửa pipeline

`src/api/routes/query.py:196-312` tự cầm retrieval + generation, **không** qua `run_agent`:

| Lớp bảo vệ | REST `/query` | WebSocket `/ws/{id}` |
|---|---|---|
| Router / intent | ✅ | ❌ |
| Lọc hiệu lực `as_of_date` | ✅ | ❌ (không nhận tham số) |
| Grade + retry | ✅ | ❌ |
| Compliance | ✅ | ❌ |
| `validate_citations` | ✅ dùng kết quả | ⚠️ **gọi rồi vứt** (dòng 281: `_, invalid = ...`) — client nhận nguyên text có citation ảo |
| Sources | chỉ nguồn được cite | ❌ 5 chunk đầu bất kể có cite hay không (dòng 293-303) |
| Persist / long-term memory | ✅ | ❌ |

Frontend hiện **không dùng** WS (`frontend/src/main.tsx` chỉ `fetch`), nhưng endpoint vẫn mở
công khai. Mọi cải tiến Phase 2/3/4 làm trong graph sẽ **không** áp dụng cho đường này.

### R4 — 🟠 Cao: rate limiting không có hiệu lực

`src/api/main.py:173-176` tạo `Limiter(default_limits=["10/minute"])`, dòng 211 gán
`app.state.limiter`, nhưng **không có `SlowAPIMiddleware`** và **không route nào** có
decorator `@limiter.limit`. Grep toàn `src/`: chỉ 2 chỗ nhắc tới `limiter`, đều trong
`main.py`. Với slowapi, `default_limits` chỉ áp dụng qua middleware → **giới hạn 10 req/phút
hiện không tồn tại**. README vẫn quảng cáo "Rate limiting (10 req/min)".

Kết hợp R5, endpoint tốn kém nhất (`/query` → 2-4 lời gọi LLM) và endpoint ghi dữ liệu
(`/upload`) đều không giới hạn, không auth.

### R5 — 🟠 Cao: `/upload` ghi thẳng vào collection đã kiểm định

`src/api/routes/documents.py:42-135` + `_index_chunks:166`: PDF bất kỳ, không auth, được
`index.insert_nodes()` vào **chính** collection `documind_legal` đang phục vụ.

Hai hệ quả:
1. Phá nguyên tắc CORPUS_SPEC §2 ("chỉ VERIFIED mới vào index") — kỷ luật corpus là điểm
   mạnh của dự án, và đây là cửa hậu phá nó.
2. Chunk upload **không có** `effective_from`/`status`/`clause_uid` → theo
   `src/rag/temporal.py:182` ("chunk không có ngày thì giữ lại"), chúng **luôn lọt** bộ lọc
   hiệu lực ở mọi `as_of_date`, kể cả mốc quá khứ.

### R6 — 🟠 Cao: image production build sai model và không có corpus

`Dockerfile:56-60` prebake **`paraphrase-multilingual-MiniLM-L12-v2`** (384-dim, model cũ),
trong khi `render.yaml:40` set `EMBEDDING_MODEL=AITeamVN/Vietnamese_Embedding` (1024-dim).
Đồng thời Dockerfile chỉ `COPY src/ scripts/ ingest.py` — **không copy `data/`**, tức không
có ChromaDB trong image.

Comment trong `render.yaml:9-11` ghi *"Corpus gốc (91 chunks, baked vào image lúc build)"* —
sai cả con số (thực tế 1146) lẫn cơ chế (không hề được bake). Kết luận: **deploy Render hiện
tại gần như chắc chắn khởi động với vector store rỗng**, hoặc lỗi model mismatch. Em chưa
verify được trên môi trường Render thật (không có quyền truy cập) — đây là suy luận từ file.

### R7 — 🟡 Trung bình-cao: observability rỗng đúng chỗ Phase 3 cần

| Cần có | Hiện trạng | Bằng chứng |
|---|---|---|
| Request ID xuyên suốt | ❌ không tồn tại | grep `request_id` trong `src/`: 0 hit (chỉ 1 comment) |
| Latency theo node p50/p95 | ⚠️ `steps[].ms` có đo nhưng **chỉ trả về response**, không log, không tổng hợp | `graph.py` mọi node |
| Token in/out | ❌ | grep `prompt_tokens|usage`: 0 hit |
| Chi phí ước tính | ❌ | — |
| Số lần fallback | ❌ chỉ có `used_llm` của lần cuối, không đếm | `generator.py:386-410` |
| `/metrics` | ⚠️ có nhưng nghèo: `total_queries`, `avg_latency_ms`, `error_rate(60')`, `active_sessions`, `corpus_chunks` — **không p50/p95** | `routes/health.py:126-160` |
| Structured logging | ⚠️ `logs/chat_history.jsonl` có JSON/dòng, nhưng log ứng dụng là text loguru | `routes/query.py:34` |

### R8 — 🟡 Trung bình: ngưỡng abstain tự tắt trên production

`generator.py:71-88`: ngưỡng 0.05 chỉ áp dụng khi cross-encoder **thực sự load được**;
nếu không → trả 0.0 (tắt gate). `render.yaml:51` đặt `ENABLE_RERANKER=false` (free tier 512MB).

Chuỗi hệ quả trên production:
1. Gate abstain tắt → mọi chunk đều "đủ điểm" → khả năng trả lời từ context rác tăng.
2. `grader.py:398` mất đường tắt heuristic → **mọi** query đều phải gọi LLM judge → +1 lời gọi
   LLM/câu, tăng latency và chi phí.
3. Chất lượng OOC-refusal rơi hoàn toàn vào prompt.

Tức là **cấu hình production khác cấu hình đo benchmark** ở đúng biến quan trọng nhất.
Mọi số trong `reports/` đều đo với reranker BẬT.

### R9 — 🟡 Trung bình: guardrails input mỏng và lệch ngôn ngữ

`src/guardrails.py`: 13 pattern injection, **10 tiếng Anh / 3 tiếng Việt**, trong khi 100%
người dùng thật hỏi tiếng Việt. 3 regex tiếng Việt viết bằng character-class thủ công
(`b[oỏ]\s+qua\s+(h[uướ]?[ơờ]?ng\s+d[aẫ][ấ]n...)`) — chưa có một test nào cho module này
(grep `check_prompt_injection` trong `tests/`: **0 hit**).

Thiếu hoàn toàn: **PII detection** (CCCD, số BHXH, số điện thoại, MST — rất dễ xuất hiện
trong câu hỏi lao động dạng "tôi đóng BHXH từ 2015, CCCD 0012...") và **scope gate**
(`tasks/todo.md` T15-T18 chưa làm). Hiện `unknown` vẫn route vào `do_retrieve`
(`graph.py:751`) — không có cổng chặn thật, chỉ có prompt + ngưỡng score đã tắt ở R8.

### R10 — 🟡 Trung bình: không có CI, lint không sạch

`.github/` đã bị xoá cùng Railway. Không có gate nào trên PR. `ruff check src tests` cho
**169 finding** (E501 ×127 toàn bộ, I001 ×39, F401 ×34) dù `tasks/todo.md` checkpoint C9
yêu cầu "ruff xanh". Không có secret scanning, dù `.env` chứa 3 Gemini key thật
(`.gitignore` đã chặn đúng — đây là rủi ro quy trình, không phải rò rỉ hiện tại).

### R11 — 🟡 Trung bình: tài liệu lệch thực tế ở mức gây hiểu sai

Với dự án portfolio, README **là** sản phẩm. Hiện tại:

| File | Sai gì |
|---|---|
| `README.md` | Toàn bộ định vị là "Vietnamese Banking Regulations"; badge "84/84 tests" (thực 166); mô tả dense = MiniLM 384-dim (thực Vietnamese_Embedding 1024); liệt kê 5 tiêu chí compliance ngân hàng như tính năng; **không nhắc temporal retrieval** — thứ duy nhất thật sự khác biệt |
| `EVALUATION.md` | Số liệu corpus UNETI (110 câu), đã tự dán nhãn "historical" nhưng vẫn là file EVALUATION duy nhất |
| `CLAUDE.md` | Ghi "1146 chunks" đúng, nhưng phần trên vẫn mô tả corpus ngân hàng 6 văn bản là "trạng thái hiện tại"; ghi `test_questions.json` "đang rỗng" (thực tế 25 câu) |
| `render.yaml` | "91 chunks baked vào image" — sai (R6) |
| `src/rag/retriever.py:2` | Docstring ghi model MiniLM |

### R12 — 🟢 Thấp: tính năng khác biệt nhất không demo được qua UI

`as_of_date` có ở API (`schemas.py:22`) và graph, đo được 36.7% → 83.3%, nhưng
`frontend/src/main.tsx` **không gửi** trường này (grep: 0 hit). Nhà tuyển dụng mở demo sẽ
không thấy gì. `tasks/todo.md` T29 đã lên kế hoạch, chưa làm.

### R13 — 🟢 Thấp: state in-process, không scale ngang

`_sessions: dict` (`query.py:52`), `_active_retriever` module-global, `ShortTermMemory` trong
RAM. Với `--workers 1` hiện tại thì đúng; thêm worker là hỏng session affinity. Cần biết
trước khi Phase 5 load test.

---

## 5. Đối chiếu 5 phase của Ted với trạng thái thực

| Phase | Trạng thái thực | Ghi chú |
|---|---|---|
| **1 — Đánh giá** | ~15%. Có harness temporal 30 câu rất tốt; thiếu gold chính, thiếu metric theo `clause_uid`, thiếu citation-correctness, thiếu CI | Phải làm trước, và phải **sau** khi chốt R1 |
| **2 — Temporal retrieval** | **~70% đã xong**. Schema cấp điều/khoản đã thiết kế, đã index đủ 1146/1146 chunk, đã có node `do_temporal_filter`, đã đo A/B, đã có `DEC-0001` | **Không cần re-index.** Việc còn lại: mở rộng gold, UI (R12), `as_of` cho compliance, link thay thế cấp văn bản |
| **3 — Observability & chi phí** | ~10%. Chỉ có `steps[].ms` và `/metrics` nghèo | Làm từ gần như đầu |
| **4 — Guardrails** | ~30%. Có injection regex + citation index check | Thiếu PII, scope gate, citation existence, và **toàn bộ phép đo FP/FN** |
| **5 — Tải & độ bền** | 0%. Thêm: rate limiting đang **không chạy** (R4) | Cần chốt môi trường đo — Render free 512MB đo ra con số khác hẳn local |

**Điều chỉnh đề xuất:** đề bài đặt Phase 2 là "điểm khác biệt chính cần xây". Thực tế nó đã
xây xong và đã có số. Việc đáng làm hơn là **Phase 0 mới** — dọn R1 (pivot prompt + compliance
+ gold) — vì nếu không, Phase 1 đo ra một baseline vô nghĩa và mọi phase sau so với nó.

---

## 6. Cần Ted chốt trước khi em viết code

1. **`docs/adr/` hay giữ `docs/decisions/DEC-*`?** Repo đã có convention DEC riêng, có
   TEMPLATE, có README giải thích "luật đảo thứ tự" (Spec → Decision → Code → Số đo), và
   2 DEC đã viết. Convention này **chặt hơn** ADR tiêu chuẩn (bắt buộc có số đo). Em đề xuất
   **giữ `docs/decisions/`** và thêm `docs/adr/README.md` trỏ sang, thay vì đổi tên 2 file
   đang được tham chiếu. Quy tắc của Ted: không đổi naming convention khi chưa hỏi.

2. **25 câu ngân hàng trong `data/eval/test_questions.json`** — xoá, hay chuyển sang
   `data/eval/_archive/`? Đây là ghi đè/xoá dữ liệu gốc, em không tự quyết.

3. **Cách dựng 200 câu gold.** Chất lượng gold quyết định mọi số về sau. Hai phương án:
   (a) **viết tay 100% từ toàn văn** như 30 câu temporal đã làm — chậm (~3-4 buổi), nhưng
   giữ nguyên chuẩn "gold-first" đang là điểm mạnh; (b) **LLM sinh nháp + em đối chiếu từng
   câu với toàn văn** rồi commit trước khi chạy — nhanh hơn ~3×, rủi ro là câu hỏi mang
   "hơi văn" của chính model đang được đo. Em đề xuất **(b) cho nhóm dễ, (a) cho nhóm
   temporal/multi-clause/out-of-scope**, và ghi rõ phương pháp từng nhóm trong gold file.

4. **Nhánh git.** Repo **không có nhánh `develop`** (chỉ `main`, `feature/labor-pivot`,
   `feature/llm-openai-backup`), trong khi `gitflow.md` yêu cầu feature branch tạo từ
   `develop`. Nhánh `feature/labor-pivot` hiện có 15 commit chưa merge. Chọn: (a) tạo
   `develop` từ `main` rồi merge `labor-pivot` vào trước; (b) tạm coi `feature/labor-pivot`
   là nhánh tích hợp và nhánh con tạo từ nó. Em nghiêng về (a).

5. **CI chạy gì.** RAGAS cần Gemini key + rate limit 3 req/61s → không thể chạy mỗi PR.
   Em đề xuất tách: mỗi PR chạy **pytest + ruff + eval retrieval deterministic** (không cần
   API key, ~2 phút); RAGAS chạy `workflow_dispatch` + nightly với secrets. Cần Ted xác nhận
   có sẵn sàng đặt GROQ/GOOGLE key làm GitHub secret không.

---

## 7. Kế hoạch chi tiết Phase 1

**Nhánh:** `feature/eval-harness-v2` · **PR:** 1 PR duy nhất, rebase squash 1 commit
**Điều kiện tiên quyết:** câu hỏi 2, 3, 4, 5 ở §6 đã được chốt.

### P1-T0 — Dọn domain mismatch (bắt buộc, làm trước)

Không thể đo baseline khi system prompt và compliance engine nói về ngân hàng.

| Việc | File | Acceptance |
|---|---|---|
| Viết lại `_SYSTEM_PROMPT` sang vertical lao động/BHXH | `src/rag/generator.py:26` | `grep -i "ngân hàng\|biểu phí" src/` rỗng |
| Đổi thông điệp từ chối | `src/api/routes/query.py:91,232` | như trên |
| Vô hiệu hoá `compliance_check` **tạm thời** (route về `do_retrieve`) cho tới khi có criteria lao động (T21-T23) | `src/agent/graph.py:745` | Không còn verdict ✅/❌ trích dẫn văn bản ngoài corpus |
| DEC ghi lại quyết định tắt tạm compliance | `docs/decisions/DEC-0003-*.md` | Có mục "Bằng chứng" |

> Em **không** tự viết `criteria.json` lao động trong Phase 1 — đó là T21-T23, phạm vi riêng,
> cần đối chiếu toàn văn. Phase 1 chỉ tắt đường sinh câu trả lời sai.

### P1-T1 — SPEC + DEC cho bộ đo v2

`docs/spec/SPEC-eval-v2.md`: định nghĩa chính xác từng metric, cách chấm, cái gì **không** đo.
`docs/decisions/DEC-0004-metric-theo-clause-uid.md`: vì sao bỏ token-F1 ≥0.15 sang so khớp
`clause_uid` thật (bằng chứng: R2.2 — chunk sai bản hiệu lực vẫn tính hit).

### P1-T2 — Schema gold set v2 (thống nhất 1 file)

```jsonc
{
  "id": "ld_q001",
  "question": "...",
  "intent": "simple_qa | compare | summarize | compliance_check | out_of_scope",
  "difficulty": "easy | medium | hard",
  "question_kind": "single_clause | multi_clause | temporal | out_of_scope | out_of_range",
  "as_of_date": "2025-03-01",                // null = hiện hành
  "gold_clause_uids": ["74-2024-ND-CP__d3"], // >=1, phải tồn tại thật trong index
  "gold_doc_ids": ["74-2024-ND-CP"],
  "ground_truth": "...",                     // trích từ toàn văn, có dẫn Điều
  "expect_contains": [["4.960.000"]],        // nhóm AND-of-OR, như gold temporal
  "expect_absent": ["5.310.000"],
  "distractor_docs": ["293-2025-ND-CP"],
  "expected_behavior": "answer | refuse_out_of_scope | refuse_out_of_range",
  "authored_by": "human | llm_drafted_human_verified",
  "source_span": "data/raw/lao_dong/74-2024-ND-CP.md#L120-L140"
}
```

30 câu temporal hiện có **tương thích ngược** (chỉ thiếu `intent`/`difficulty`/`gold_clause_uids`
— map từ `source_clause`). Migration script + test đảm bảo không mất câu nào.

**Test chặn gold rác** (`tests/test_eval_goldset.py` mở rộng): mọi `gold_clause_uids` phải tồn
tại trong collection; mọi `as_of_date` hợp lệ ISO; mọi `expect_absent` không nằm trong
`expect_contains`; không trùng `id`.

### P1-T3 — Dựng ≥200 câu, commit **trước** khi chạy lần đầu

Phân bổ đề xuất (tổng 205):

| Nhóm | n | Nguồn |
|---|---|---|
| `single_clause` (1 điều) | 70 | BLLĐ, NĐ 145/2020, Luật BHXH 41/2024 |
| `multi_clause` (cần ≥2 điều, có thể khác văn bản) | 40 | VD: trợ cấp thôi việc = Điều 46 BLLĐ + Điều 8 NĐ 145 |
| `temporal` | 45 | 30 câu sẵn + 15 câu mới (BHXH 2014↔2024, Việc làm 2013↔2025) |
| `out_of_scope` (phải từ chối) | 30 | thuế, hình sự, đất đai, doanh nghiệp, y tế |
| `out_of_range` (mốc < 2015-01-01) | 5 | mở rộng từ 2 câu sẵn có |
| `compliance/formula` | 15 | làm thêm giờ, tuổi hưu, tỷ lệ đóng — dùng cho T21-T23 sau |

Quy tắc bất di bất dịch: **commit gold trước commit chạy eval**, như `5723245` → `250c8a3`.

### P1-T4 — `eval/run_evals.py` v2 — một lệnh

```powershell
python eval/run_evals.py --suite full --output reports/eval/2026-09-XX/
```

| Nhóm metric | Chỉ số | Cần LLM? |
|---|---|---|
| Retrieval | `recall@1/5/8/20`, `MRR`, `nDCG@8` — **so khớp `clause_uid`**, không token-F1 | ❌ |
| Temporal | `context_gold`, `context_distractor`, `context_clean`, `answer_accuracy` (giữ nguyên harness đã có) | một phần |
| Citation | `citation_rate`, **`citation_validity`** (mọi `[N]` ≤ chunk_count), **`citation_groundedness`** (clause được cite ∈ `gold_clause_uids` ∪ context) | ❌ |
| Từ chối | `refusal_accuracy` (OOC bị từ chối), **`over_refusal_rate`** (câu in-scope bị từ chối oan) | ❌ |
| RAGAS | faithfulness, answer_relevancy, context_precision, context_recall | ✅ (flag `--ragas`) |
| Latency | mean/p50/p95 theo strategy | ❌ |

Chia 2 chế độ: `--retrieval-only` (deterministic, 0 API call, chạy được trong CI) và
`--full` (có generation + RAGAS).

### P1-T5 — Báo cáo + so sánh baseline

- `reports/eval/<ts>/report.md` — bảng số + failure list theo từng câu sai + phân loại nguyên nhân.
- `reports/eval/<ts>/metrics.csv` — 1 dòng/strategy/metric, để vẽ biểu đồ.
- `reports/eval/baseline.json` — snapshot được "đóng băng", mọi phase sau diff với file này.
- `eval/compare_runs.py` — in bảng Δ giữa 2 run, đánh dấu regression.

### P1-T6 — CI

`.github/workflows/ci.yml` (mọi PR, không cần secret): `ruff check src tests` → `pytest`
→ `python eval/run_evals.py --retrieval-only --suite full` → so với `baseline.json`,
**fail nếu `recall@8` giảm >2 điểm tuyệt đối**.
`.github/workflows/eval-full.yml` (`workflow_dispatch` + nightly, dùng secrets): bản đầy đủ + RAGAS.

Kèm việc: đưa `ruff check src tests` về xanh (169 finding, phần lớn E501/I001/F401 — sửa cơ học).

### P1-T7 — DEC tổng kết

`DEC-0005`: vì sao CI chỉ chạy retrieval metric; `DEC-0006`: ngưỡng regression gate chọn 2 điểm
dựa trên độ nhiễu đo được khi chạy lại cùng cấu hình 3 lần.

### Bàn giao Phase 1

- [ ] Gold ≥200 câu, 100% `gold_clause_uids` tồn tại thật, commit riêng trước khi chạy
- [ ] `python eval/run_evals.py` chạy một lệnh ra đủ 6 nhóm metric
- [ ] `reports/eval/baseline.json` + `report.md` + `metrics.csv` có số thật
- [ ] CI xanh trên PR; regression gate chặn được (test bằng cách cố tình hạ `top_k`)
- [ ] 3-4 DEC mới có mục "Bằng chứng" điền số
- [ ] `pytest` + `ruff check src tests` xanh

### Ước lượng & rủi ro

| Hạng mục | Ước lượng |
|---|---|
| P1-T0 dọn domain | 0.5 buổi |
| P1-T2/T4/T5 code harness | 1.5 buổi |
| P1-T3 dựng 200 câu gold | 2-4 buổi (phụ thuộc câu trả lời §6.3) |
| P1-T6 CI + dọn lint | 0.5 buổi |

**Rủi ro lớn nhất:** gold set. 200 câu viết ẩu sẽ cho ra baseline đẹp mà vô nghĩa — đúng
cái bẫy mà chính Ted đã tránh khi xoá `benchmark_results.json` cũ. Em đề xuất chốt §6.3
trước khi bắt đầu.
