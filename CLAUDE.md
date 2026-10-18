# DocuMind AI — CLAUDE.md

Project hướng dẫn cho Claude Code. Đọc file này trước khi làm bất cứ gì.

**Sai lầm & bài học:** khi phát hiện một **quyết định kỹ thuật sai** (chọn công nghệ, kiến
trúc, ngưỡng, dữ liệu) hoặc một **lần xử lý sai** (kể cả của Claude: đoán thay vì đo, báo
"đã verify" khi chưa) — thêm một mục vào `docs/LESSONS.md` **trong cùng commit** với bản
sửa. Không ghi bug lặt vặt. Định dạng và luật ghi ở đầu file đó.

## Project overview

AI agent RAG tra cứu **pháp luật lao động và bảo hiểm xã hội Việt Nam** — Bộ luật Lao động,
Luật BHXH, Luật Việc làm cùng nghị định/thông tư hướng dẫn. Trả lời kèm trích dẫn điều khoản,
từ chối khi câu hỏi ngoài phạm vi tài liệu đã nạp, và tra cứu được **theo thời điểm**
(`as_of_date`) vì nhiều văn bản trong corpus đã bị thay thế.

> **Phạm vi sản phẩm là nguồn sự thật, không phải văn bản tự do.** Khai báo ở
> `src/config.py` (`DOMAIN_NAME` / `DOMAIN_SCOPE` / `DOMAIN_TOPICS`) và
> `frontend/src/main.tsx` (`PRODUCT`). Mọi prompt, message từ chối, nhãn UI phải lấy từ đó —
> đừng viết lại chuỗi mô tả phạm vi ở chỗ khác.

**Trạng thái hiện tại:**
- Corpus lao động/BHXH: 20 văn bản tại `data/raw/lao_dong/` (+ `_versions/` cho bản sửa đổi
  cấp khoản). Index có **19** `doc_id`: `18-VBHN-VPQH` là bản hợp nhất, nạp làm nội dung của `45-2019-QH14`
  (`consolidated_from`) — `/documents` hiện 19 là đúng. Corpus ngân hàng cũ đã bỏ — `data/raw/banking_docs/` không còn tồn tại.
- ChromaDB `data/chroma_db/`: local dùng **`documind_legal_fpt`** (FPT `Vietnamese_Embedding`, 1024-dim,
  **1249 chunks**, từ 2026-10-17 — `.env` CHROMA_COLLECTION/EMBEDDING_MODEL); `documind_legal` (gemini-embedding-2,
  1247, **chưa có** bản sửa khoản hết hiệu lực bên dưới) và `documind_legal__backup_gemini-embedding-001`
  (1151, corpus cũ) còn giữ. Ingest lại 2026-10-17 sau khi bù 145/2020 Điều 102–115 + phụ lục, sửa
  bảng/công thức. 2026-10-18: khoản đi sửa một luật đã bị thay (`loai_sua: het_hieu_luc` — BLLĐ Điều 219
  khoản 1 sửa Luật BHXH 58/2014, Luật BHXH 41/2024 Điều 139 khoản 2 sửa Luật Việc làm 38/2013) tách ra
  và đóng hiệu lực cùng luật đó → 1249; vá tại chỗ cả Chroma lẫn Qdrant, đối chiếu khớp bản ingest mới.
  Qdrant Cloud (production đọc từ đây): collection **`documind_legal_fpt`** (1249 point, FPT
  `Vietnamese_Embedding` 1024-dim, migrate + verify 2026-10-18); env Render + CI đặt
  `QDRANT_COLLECTION=documind_legal_fpt`, `EMBEDDING_MODEL=Vietnamese_Embedding`. Collection cũ
  `documind_legal` (gemini-embedding-001, 1151, corpus cũ) giữ nguyên làm đường quay lui — đổi lại
  2 biến env là về bản cũ. `scripts/restore_chroma_from_qdrant.py` phải chạy với cùng cặp env đó.
  **Danh tính index = hash nội dung, không phải số chunk** (`vector_backend.corpus_fingerprint`):
  app log `Index fingerprint …`, `meta.index`/trace có đuôi hash. 2026-10-18 Qdrant = Chroma =
  `ingest_documents.py --dry-run` = `5ce81d96ccb9`. Index có khớp code không → so hash dry-run với hash
  app log, đừng suy từ git log. `.env` local KHÔNG đặt `QDRANT_COLLECTION` — chạy gì trên Qdrant từ
  máy local phải truyền `QDRANT_COLLECTION=documind_legal_fpt`, không thì rơi về `documind_legal` (3072-dim).
- `data/compliance/criteria.json`: 6 tiêu chí định lượng lao động (trần làm thêm giờ
  năm/tháng, thời gian thử việc, lương thử việc 85%, nghỉ hằng năm, lương tối thiểu vùng I).
  Mọi tiêu chí phải trích dẫn văn bản CÓ trong `data/raw/lao_dong/` — có test chặn
  (`tests/test_compliance.py::TestCriteriaDataIntegrity`).
- Gold set: `data/eval/temporal_questions.json` (30 câu, viết tay trước khi chạy hệ thống).
  Bộ 25 câu ngân hàng tiền-pivot đã archive sang `data/eval/_archive/`.
- Bộ câu khó: `data/eval/hard_questions.json` (19 câu, 6 loại: ghép nhiều điều khoản, ngày
  giao thời, ngoài phạm vi, tiền đề sai, tình huống tuân thủ, tra bảng). Claude soạn từ
  nguyên văn corpus, `reviewed: false` — **cần người duyệt**. Bộ 30 câu temporal đã bão hoà
  (1.000), dùng bộ này để đo cải tiến:
  `python eval/temporal_eval.py --gold data/eval/hard_questions.json --output reports/hard_eval.json`
- **Gold set chính: `data/eval/legal_qa_200.json` (208 câu)** = 30 temporal + 19 hard + 150 câu
  Claude soạn (`g2_*`, `reviewed: false`) + 9 câu `g3_cv*` (2026-10-18, `clause_version`, `reviewed: false`):
  corpus chỉ có 6 khoản nhiều phiên bản (12 phiên bản) mà gold cũ phủ 4 — giờ phủ đủ 12, có cặp cùng câu
  hỏi ở hai thời điểm. Phiên bản theo văn bản (357 chunk, 9 văn bản bị thay) là phần dày của corpus.
  Câu Có/Không có thể khai `expect_verdict: "có"|"không"` — câu trả lời phải MỞ ĐẦU bằng kết luận đó (chấm
  chuỗi không bắt được "**Có** … phải đủ 20 năm"); mới có `lh_b02a/b02b`. Kiểm tra với index thật:
  `python eval/validate_gold.py data/eval/legal_qa_200.json`. Metric: recall@1/3/5/8, MRR,
  faithfulness (`--judge <arms>`), p50/p95; `--mlflow <run>` log vào MLflow (`mlflow.db`, gitignored);
  `--embed-cache` chỉ dùng cho CI (latency khi đó không phải số thật). Kết quả: README mục Đánh giá —
  chỉ trích nguyên `summary` của report, không tính tay trên tập con (test chặn); report ghi
  `meta.index` (provider/collection/model/số chunk/hash nội dung) + `meta.git_sha`.
- Nguyên liệu câu hỏi tổng hợp (CHƯA DUYỆT, không phải gold): `data/eval/synthetic/doc2query_questions.jsonl` —
  4.992 câu đời thường FPT sinh cho 1.249 đoạn (2026-10-18), mỗi dòng mang `text_sha1` của đoạn để biết dòng nào cũ.
  Sinh tiếp cho đoạn mới/đổi: `python scripts/generate_doc2query.py`. Xem README trong thư mục trước khi dùng.
- CI: `.github/workflows/ci.yml` — pytest + retrieval eval trên Qdrant `documind_legal_fpt`, gate bằng
  `eval/ci_gate.py` so với `reports/eval_baseline.json` (FAIL cả khi report có câu đo thiếu nhánh dense —
  `meta.n_degraded`). Reranker TẮT trong CI như production.
  Cần secrets `FPT_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`.
- **Chroma Cloud** (database `DocuMind`, biến `CHROMA_CLOUD_*`): chỉ là bản sao để xem dashboard
  và so provider — app KHÔNG đọc từ đó. Đồng bộ: `python scripts/copy_chroma_to_cloud.py`; so
  sánh: `python eval/provider_bench.py` (kết quả README mục Đánh giá). Gọi qua REST
  (`src/rag/chroma_cloud.py`) vì `chromadb` 0.6.3 không tương thích server Cloud 1.x.
- **Reranker local đang KHÔNG chạy** (2026-09-23): Windows Smart App Control chặn DLL `pyarrow`
  → `sentence_transformers` không import được → retriever log "Reranker unavailable" và bỏ qua.
  Report nào ghi rerank phải xem `meta.reranker_active`, không tin `ENABLE_RERANKER`.
  torch/sentence-transformers/transformers **không** còn trong `requirements.txt` (2026-09-27,
  image production không cài ~1.3GB này) — muốn chạy reranker: `pip install -r requirements-rerank.txt`.
- Test suite: 687 passed (đo 2026-10-18). Test không gửi trace ra Langfuse/LangSmith
  (`conftest.py::patch_settings` xoá key).

**Stack:**
- Backend: FastAPI + LangGraph agent + vector store qua `VECTOR_STORE_PROVIDER`
  (`chroma` mặc định local, hoặc `qdrant` cho Qdrant Cloud — xem `src/rag/vector_backend.py`)
- Embedding: local `Vietnamese_Embedding` (1024-dim) qua FPT `/v1/embeddings` (từ 2026-10-17; production còn
  ghim `gemini-embedding-001`) — **không nạp model nào vào RAM**. Ngưỡng cosine grader theo model
  (`grader._COSINE_GATES`): model chưa đo thì luôn hỏi judge. Model đọc từ `EMBEDDING_MODEL`, không hard-code. Lịch sử: chạy
  model local `AITeamVN/Vietnamese_Embedding` (1024-dim) tới 2026-09-19; `gemini-embedding-001` tới 2026-10.
- Retriever: Hybrid BM25 + dense vector + RRF fusion + cross-encoder reranker
  (`bge-reranker-v2-m3` qua FPT `/v1/rerank` khi có FPT_API_KEY, 10 ứng viên — `_RERANK_POOL`; không key thì
  local). Sau fusion,
  `_apply_query_priors` (retriever.py) hạ 3 bậc đoạn thuộc chương/văn bản BHXH tự nguyện khi câu
  hỏi không nói "tự nguyện", và đoạn phụ lục khi câu hỏi không hỏi danh mục/biểu mẫu/năm sinh/địa
  bàn — chỉ đổi thứ tự, không đổi điểm. Đo 2026-10-17 (legal_qa_200, recall@1 câu một điều):
  FPT embedding 0.828 → luật 0.834 → + rerank pool 10 **0.882** (`reports/temporal_eval_vnembed_rerank10.json`,
  rerank API p50 250ms); Gemini-2 0.72 → 0.78 với luật. Đọc theo cặp, không theo tỷ lệ (n=169, CI95 ±0.05):
  rerank thắng 13/thua 5 ở hạng 1 (p≈0.10), không tăng top-8 → production TẮT rerank. Eval đầu cuối
  2026-10-18 (`reports/e2e_k_rerank.json`, k = số đoạn vào LLM ∈ {3,5,8} × rerank) giữ quyết định đó:
  k=8 không rerank accuracy 0.950 (lúc đó câu < 150 ký tự thực ra chỉ nhận 5 đoạn — xem dưới); rerank ở k=8 thắng 3/thua 1 câu — trong mức
  dao động 2–3 câu giữa hai lượt cùng cấu hình (FPT temperature 0 vẫn không tất định) — mà p95 +0.7s và
  /v1/rerank 429 khi tải (chung hạn mức key với LLM). k mới là đòn bẩy: k=8 so k=3 thắng 9/thua 2.
  Faithfulness judge ≈0.99 ở cả 6 cấu hình và cho 1.0 cả câu sai bám đúng một đoạn đã hết hiệu lực →
  chưa dùng làm thước đo an toàn được; chấm chuỗi cũng báo sai câu đúng. Đo lại sau chunk H1 + gold đã duyệt
  (`reports/retrieval_experiments_2026-10-18.json`): rerank r@1 câu một điều 0.860 → 0.899 nhưng r@8 câu một điều 0.989 → 0.983,
  +0.76 s, 29/199 câu vẫn 429 ở 50 lượt/phút — vẫn TẮT (LLM nhận đủ 8 đoạn nên thứ tự trong top-8 ít đổi câu trả
  lời). doc2query (FPT sinh 4 câu hỏi/đoạn, nhánh RRF thứ ba) đã thử và LOẠI: cứu 4–7/29 câu người duyệt xác nhận
  truy hồi sai nhưng r@1 0.860 → 0.788, độ phủ câu nhiều điều 0.969 → 0.938. Hiệu chỉnh cả hai bằng người
  chấm: bảng duyệt https://claude.ai/artifact/GwULtyAAPsxS5LewLzjcdn (36 câu trượt hạng 1 + 30 câu trả
  lời k=8; nhãn ở collection `reviews` của artifact, Claude đọc bằng ArtifactData). Lượt đầu (32/36 `gold_sai`)
  chấm trên văn bản bị trang cắt ở 700–900 ký tự → bỏ. Lượt duyệt lại trên toàn văn (2026-10-18, 36/36): truy hồi
  sai 26, nhiều điều đều đúng 5, diễn đạt xa 3, gold sai 2 → đã áp vào `legal_qa_200.json` (`acceptable_clauses`,
  sửa g2_b033/g2_b036, mỗi câu ghi `clause_review`; `reviewed` vẫn false — người duyệt chấm nhãn điều khoản, chưa
  duyệt cả câu). Phần B (30 câu trả lời để hiệu chỉnh judge) chưa ai chấm. `ci_gate` in bảng thắng/thua theo câu;
  câu nhiều điều khoản đo bằng `multi_coverage_at_k` (recall@1 của chúng luôn = 0 theo định nghĩa).
- LLM: **chỉ FPT AI Marketplace** (2026-10-18): `GEMINI_GENERATION_MODELS`, `GEMINI_GENERATION_MODELS_COMPLEX`,
  `GEMINI_JUDGE_MODELS` để TRỐNG ở `.env` và Render ⇒ không còn cặp Gemini; FPT hỏng cả hai model thì trích
  nguyên văn nguồn (`used_llm=extractive_fallback`). Bật lại Gemini = điền lại 3 biến (code giữ nguyên;
  test ép default qua `conftest.py`). Mô tả cơ chế khi còn Gemini (2026-10-17): `_gemini_pairs` đặt các
  cặp `("fpt", model)` từ `FPT_MODELS` lên đầu vòng xoay, nên mọi call site (`gemini_generate`,
  `stream_answer`, contextualize/restore-diacritics trong graph) thử FPT trước; lỗi FPT nào
  (kể cả 401) cũng cooldown model đó rồi rơi về Gemini (`gemini-3.8-flash`, phân loại
  `gemini-3.5-flash-lite`, 3 key × `GEMINI_GENERATION_MODELS`); hết mọi cặp thì trích nguyên
  văn nguồn. FPT gọi bằng `requests` tới `https://token-api.fpt.ai/v1/chat/completions`
  (OpenAI-compatible; `mkp-api.fptcloud.com` trả 401 với key token-api), không cài
  `openai`/`litellm`. Luôn gửi `chat_template_kwargs.enable_thinking=false`: bật suy luận thì
  Qwen3.8-27B mất 23s cho câu 1 dòng (quá timeout 20s của router). `used_llm` = `fpt`/`gemini`
  theo nhà cung cấp thật sự trả lời. Lượt gọi Qwen đầu tiên sau lúc nhàn rỗi có thể treo tới
  timeout (đo 2026-10-17, phía FPT) → cooldown 120s, model kế tiếp gánh.
  Round-robin cursor chỉ xoay phần Gemini. Đã gỡ Groq, embedding local và HF Inference API.
  UI web đi WebSocket → `stream_answer`, eval đi `generate_answer`: từ 2026-10-18 hai đường cùng gắn
  `_as_of_block` (trước đó đường stream không có ngày tra cứu). Sửa prompt thì sửa `_SYSTEM_PROMPT` —
  dùng chung cho cả hai.
  **Câu Có/Không — phân tích trước, trả lời sau** (`_analysis_block`, 2026-10-18): câu khớp `_YES_NO_RE`
  (30/208 câu gold) gọi thêm một lượt `analyze-conditions` (trích điều kiện kèm [N] → áp vào dữ kiện →
  kết luận) rồi nối phân tích vào ngữ cảnh của lượt trả lời; cả `generate_answer` lẫn `stream_answer`
  (stream vẫn stream — phân tích xong mới bắt đầu). Lỗi thì bỏ qua. Lý do: rule 8 BLUF + suy luận tắt ⇒
  Qwen chốt Có/Không theo luật cũ nó nhớ — `lh_m05b` (16 năm đóng, Luật 41/2024 chỉ cần 15) đúng 10/24 →
  12/12; `lh_c02b`, `g2_d028` 6/6; +2.0s p50, max 5.6s. Đã đo và loại: rule "trí nhớ có thể là luật cũ"
  5/12; BLUF "điều kiện trước" 5/12, 4/12; bỏ BLUF 11/12 nhưng 208 câu thêm 2 lỗi thật; DeepSeek trước Qwen
  thua 6 (`lh_o02` trả lời câu ngoài phạm vi); `enable_thinking` cho câu phức tạp p95 sinh 61s; kiểm SAU
  bản nháp (verifier) 10/12 — model tự phân tích đúng nhưng so với bản nháp lại ghi "GIỮ". So phương án đổi
  prompt/model thì đọc từng câu đổi kết quả — chấm chuỗi thiên vị cấu hình đang chạy (LESSONS).
  Rule 8 (BLUF) chỉ áp khi các đoạn CÓ nội dung được hỏi; không có thì không nêu con số, dùng khuôn hỏi lại
  của rule 4 (2026-10-18 — DeepSeek nêu "1,5%" BHYT theo trí nhớ cho câu ngoài phạm vi `lh_o02` 4/12 → 0/12).
  **Model dự phòng phải eval như model chính:** DeepSeek gánh câu đầu sau mỗi lần Qwen treo/cooldown.
  `generate_answer` trả `model`, report eval ghi `model` từng câu; DeepSeek trả lời cả 208 câu:
  `reports/eval_e2e_deepseek_first.json` (0.981, ngoài phạm vi 9/9). Đo DeepSeek: `FPT_MODELS=DeepSeek-V4-Flash,Qwen3.8-27B`.
  Chạy 2 eval đầu cuối song song trên cùng key FPT = 429 embedding — chạy tuần tự.
- Frontend: React + Vite (port 5174), proxies `/api` → backend port 8081

**Deploy (2026-07, theo default-tech-stack skill):**
- Local dev: như cũ, không đổi gì (`.\start.ps1`)
- Production: **chỉ Render, một service** (2026-10-02): `https://documindai-pd0r.onrender.com`
  (service `DocuMindAI`, `srv-daron9h7lnhs73e89nkg`) phục vụ cả UI (Dockerfile build React vào
  `dist/`) lẫn API, cùng domain — không cần `VITE_API_URL`/`ALLOWED_ORIGINS`
  cho UI của chính nó. Service này tạo tay, KHÔNG qua Blueprint: env trong `render.yaml` không
  tự áp, phải điền ở dashboard (thiếu `VECTOR_STORE_PROVIDER=qdrant` = Chroma rỗng trong
  container; thiếu `ENVIRONMENT=production` = trace mang nhãn `development`, `/docs` mở).
- Đã xoá CI/CD Railway (`.github/workflows/deploy.yml`, `railway.toml`, `deploy_railway.ps1`,
  `scripts/start_railway.sh`) — không còn dùng Railway ở bất kỳ đâu trong repo
- Không dùng Supabase/Postgres — dự án không có bảng quan hệ nào (chỉ log JSONL)

## Corpus & Compliance

Corpus tại `data/raw/lao_dong/` — 20 văn bản, frontmatter YAML (`doc_id`, `so_hieu`, `ten`,
`ngay_hieu_luc`, `trang_thai`), thân bài chuẩn hoá theo `Điều ...`; `_versions/` chứa bản
sửa đổi ở cấp khoản. Bốn nhóm:
1. **Lao động**: `45-2019-QH14` (Bộ luật Lao động), `18-VBHN-VPQH`, `145-2020-ND-CP`, `10-2020-TT-BLDTBXH`.
2. **Lương tối thiểu**: `293-2025-ND-CP` (hiệu lực 01/01/2026, vùng I 5.310.000đ) thay `74-2024-ND-CP` (4.960.000đ).
3. **BHXH**: `41-2024-QH15` thay `58-2014-QH13`; nghị định/thông tư `115-2015`, `134-2015`, `158-2025`, `159-2025`, `59-2015-TT-BLDTBXH`, `11`/`12-2025-TT-BNV`.
4. **Việc làm & BHTN**: `74-2025-QH15` thay `38-2013-QH13`; `374-2025-ND-CP`, `28-2015-ND-CP`. Thêm `135-2020-ND-CP` (tuổi nghỉ hưu).

Cả bản cũ lẫn bản mới đều nằm trong index — đó là điều làm `as_of_date` có ý nghĩa.

Lệnh ingest tài liệu:
```powershell
# Ingest corpus lao động vào ChromaDB.
# BẮT BUỘC có --manifest: thiếu cờ này script rơi về chế độ quét thư mục, nuốt luôn
# _versions/ thành tài liệu độc lập (chunk rác thay vì 1249).
python scripts/ingest_documents.py --source-dir data/raw/lao_dong `
  --manifest docs/corpus/corpus_manifest.yaml --reset

# Xem trước số chunk mà không ghi vào DB
python scripts/ingest_documents.py --source-dir data/raw/lao_dong `
  --manifest docs/corpus/corpus_manifest.yaml --dry-run
```

**Access control / multi-tenant / point-in-time (DEC-0003):** mọi truy hồi đi qua
`RetrievalContext` (`src/rag/context.py`) — tenant + nhãn ACL + `as_of_date` compile thành
filter đẩy **xuống** vector store trước khi search. Bốn field metadata bắt buộc:
`tenant_id`, `acl_label`, `effective_from_i`, `effective_to_i` (ngày dạng int YYYYMMDD —
Chroma 0.6.3 không so sánh `$lte` trên chuỗi). Chunk thiếu 4 field này **biến mất khỏi mọi
kết quả**, nên sau khi ingest lại corpus cũ phải chạy:

```powershell
python scripts/backfill_access_meta.py          # xem trước
python scripts/backfill_access_meta.py --yes    # ghi
```

Tenant demo khai ở `data/tenants/tenants.json`; client gửi `x-api-key`, **không bao giờ**
tự khai `acl_labels` (server tra ra). Không header = ẩn danh, chỉ thấy `tenant_id=public`.
UI web **không** có ô nhập key (đã gỡ 2026-09-23) — luôn chạy ẩn danh/`public`; tenant
chỉ dùng qua API. WebSocket: trình duyệt không gắn được header lên handshake, nên client
gửi key trong message (`{"query": ..., "api_key": ...}`), server resolve theo từng lượt. Header vẫn
dùng được cho client không phải trình duyệt. `GET /api/v1/whoami` cho biết key ứng với
tenant nào. CORS phải giữ `X-API-Key` trong `allow_headers` (`src/api/main.py`).
`src/rag/temporal.py` vẫn chạy làm lớp lọc thứ hai — đừng gỡ.

`data/compliance/criteria.json` định nghĩa 6 tiêu chí kiểm định tuân thủ định lượng
(pass/fail) cho agent node `compliance_check`. Engine này phát ✅/❌ kèm trích dẫn nên là
đường đi thẳng tới "câu trả lời sai có thẩm quyền" — khi thêm tiêu chí:
- Số liệu phải tra từ chính văn bản trong `data/raw/lao_dong/`, không lấy từ trí nhớ.
- `so_hieu` phải trỏ tới văn bản CÓ trong corpus (test `TestCriteriaDataIntegrity` chặn).
- `anchors` (bắt buộc): cụm CHỦ ĐỀ — câu không chứa cụm nào thì tiêu chí được 0 điểm.
  `keywords` chấm theo tổng ĐỘ DÀI cụm khớp, ngưỡng `_MIN_KEYWORD_SCORE = 6`, chỉ để phân xử
  giữa tiêu chí cùng họ. Khớp theo ranh giới từ ("vùng i" ≠ "vùng iii"). Tránh cụm quá rộng.
- Chỉ khớp bằng từ khoá, trích số chỉ bằng regex, số phải nằm trong 30 ký tự quanh một anchor
  và mang đúng đơn vị/kỳ hạn của tiêu chí; không chắc thì `no_match`/`insufficient_info` → RAG.
  Đã gỡ (2026-10-18) đường khớp embedding (0/5 đúng) và LLM trích số (đoán "10 tháng lương" thành
  10 triệu): bản cũ phát ✅/❌ cho 42/83 câu gold có chữ số, 10 phán quyết sai tới tay người dùng.
  `test_gold_set_gets_verdicts_only_on_compliance_questions` chạy engine trên cả gold set (tất định).
  Eval đầu cuối (`temporal_eval.py`) gọi thẳng `generate_answer`, KHÔNG qua router/node này.
Đánh giá retrieval benchmark:
```powershell
python eval/run_evals.py --strategies dense rerank --retrieval-only --limit 5
```

## Chạy local

**Một lệnh duy nhất (mở 2 cửa sổ):**
```powershell
.\start.ps1
```
Mở 2 terminal riêng — backend + frontend. Chờ backend in `Application startup complete` (~25s) rồi mở UI.

**Thủ công (2 terminal):**
```powershell
# Terminal 1 — Backend (chạy trước)
uvicorn src.api.main:app --host 0.0.0.0 --port 8081 --reload

# Terminal 2 — Frontend (chờ backend sẵn sàng rồi mới chạy)
cd frontend
npm run dev
```

Mở `http://localhost:5174`. API docs: `http://localhost:8081/docs`.

## Cấu trúc thư mục quan trọng

```
src/
  api/          FastAPI app, routes (query, documents, health, reports)
  rag/
    embedder.py       SentenceTransformer embedder + Chroma/Qdrant client factories
    vector_backend.py Provider-agnostic vector store access (count/fetch/query) —
                       đổi VECTOR_STORE_PROVIDER không cần sửa call site nào khác
    retriever.py      Hybrid retriever + reranker, RetrievedChunk dataclass
    generator.py      LLM generation (chỉ Gemini, xoay key×model), score filtering
    compliance.py     Compliance-check engine (pass/fail theo tiêu chí JSON hand-curated)
  agent/
    graph.py      LangGraph agent graph
    memory.py     ShortTermMemory (cache in-process, max 10 turns, persist SQLite
                  qua session_id — sống qua được restart/redeploy)
  config.py       Pydantic settings — đọc từ .env
  logger.py       Loguru setup

frontend/src/
  main.tsx        React SPA — chat, docs list, PDF upload tabs
  styles.css

scripts/
  ingest_documents.py        Ingest thư mục .md bất kỳ (chunk theo Điều/Khoản) vào ChromaDB
  rebuild_chroma_direct.py   Ingest từ data/raw/ (dùng cho JSON pháp luật)
  expand_corpus.py           Crawl thêm văn bản
  migrate_chroma_to_qdrant.py   Migrate corpus ChromaDB local → Qdrant Cloud
                                 (đọc embeddings có sẵn, không re-embed)

eval/
  rag_comparison.py  Benchmark pipeline (RAGAS + local metrics)
  run_evals.py       Entry point eval
  metrics.py         Local metrics: hit_rate, MRR, citation_rate, ooc_refusal_rate
  corpus_llm_audit.py  Model FPT đọc từng đoạn corpus: loại đoạn + lỗi trích xuất (bảng vỡ, câu đứt,
                       lẫn nội dung). Cache theo hash đoạn; mặc định 80 RPM (FPT 100 RPM/model, chung
                       key với production). Kiểm toàn văn với nguồn: `python scripts/acquire_corpus.py`

logs/
  chat_history.jsonl   Log mỗi query/response (append, structured JSON; tự tạo lại khi chạy)
```

## Config quan trọng (.env)

- `FPT_API_KEY`, `FPT_MODELS` — LLM chính (FPT AI Marketplace); key rỗng = chỉ Gemini.
  Test xoá `FPT_API_KEY` (`conftest.py::patch_settings`) để không gọi API thật.
- `GOOGLE_API_KEY`, `GOOGLE_API_KEY_2`, `GOOGLE_API_KEY_3` — 3 Gemini keys
- `GEMINI_JUDGE_MODELS` — danh sách model cho RAGAS eval & classification (mặc định `gemini-3.5-flash-lite`)
- `GEMINI_GENERATION_MODELS` — danh sách model Gemini cho generation (mặc định `gemini-3.8-flash,gemini-3.5-flash-lite`)
- `EMBEDDING_MODEL=Vietnamese_Embedding` (local; tên không bắt đầu bằng `gemini` = đi FPT) — nguồn sự thật duy nhất cho model
  embedding. Đổi giá trị này **bắt buộc** chạy `python scripts/reembed_corpus.py --yes`;
  collection mang nhãn model đã index, lệch nhãn là `EmbeddingModelMismatch` lúc mở store.
- `API_PORT=8081`
- **Không được** thêm `HF_HOME` vào `.env` — pydantic `extra_forbidden` sẽ reject
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` (hoặc `LANGFUSE_HOST`,
  cả hai tên đều được đọc) — tracing tuỳ chọn, rỗng thì tắt hẳn. Xem `src/langfuse_otel.py`:
  gửi OTLP/HTTP thuần qua `requests`, KHÔNG cài SDK `langfuse` chính thức (xung đột version
  `opentelemetry-api/sdk` với `chromadb` trong venv này — đã thử thật, phá 17 test). Một trace
  = một lượt `run_agent()`/1 câu hỏi WS, span con cho retrieve (`retriever`), mọi lời gọi Gemini
  (`generation`, kèm model/token/cost), compliance-check (`tool`).
  Trace ghi *quyết định kèm lý do* (`tests/test_trace_decisions.py`): mọi span có `environment` +
  `release` (git SHA, `-dirty` = code chưa commit), trace có metadata `index`
  (provider/collection/model/số chunk); tag `route:<vì sao ra intent>`, `as_of:`/`as_of_src:`,
  `abstained`/`uncited`/`invalid_citation`; span `retrieve-documents` (từng chunk: văn bản, điều,
  khoản, hiệu lực, điểm RRF/rerank + `cosine` dense + `bm25` thô — không kèm nội dung),
  `temporal-filter` (đoạn bị loại + lý do),
  `compliance-check` (tiêu chí, cụm từ khoá khớp + điểm, số trích được), `post-process`
  (`[n]` → điều nào). Lượt thử Gemini lỗi mà vòng xoay còn đi tiếp = WARNING, hết vòng = ERROR.
  Dựng payload trace không bao giờ được làm hỏng câu trả lời (`langfuse_otel.non_fatal`).
  Gửi: thread daemon, gửi lại 429/5xx/lỗi mạng, 4xx thì bỏ — mọi batch bị bỏ đều log WARNING
  (key sai = `HTTP 401`); lúc thoát `atexit` chờ tổng tối đa 5s rồi log số batch còn dở.
  Đọc trace bằng code: CHỈ `GET /api/public/v2/observations?fromStartTime=&toStartTime=`
  (`fields=core,basic,io,metadata,trace_context`) — org tạo sau 16/09/2026 bị chặn
  `/api/public/traces` (410 `LEGACY_API_UNAVAILABLE_FOR_NEW_ORGANIZATION`).
- `LANGCHAIN_API_KEY` / `LANGCHAIN_TRACING_V2` / `LANGCHAIN_PROJECT` — LangSmith, chạy song song
  Langfuse (bật cả hai = mỗi câu hỏi trace 2 nơi). Kiểm tra e2e (gọi Gemini + LangSmith thật):
  `python scripts/smoke_langsmith_e2e.py --n 15` (câu lấy từ gold set, mỗi chủ đề một câu).
  Trên LangSmith mọi lời gọi LLM là run `gemini-generate` (`@traceable` trên `gemini_generate`),
  mục đích thấy qua node cha. Langfuse xếp generation phẳng nên đặt tên theo mục đích:
  `route-intent`, `contextualize-query`, `restore-diacritics`, `grade-chunks`, `reformulate-query`,
  `generate-answer`(`-stream`), `summarize-document`,
  `compare-documents`, `eval-judge-faithfulness` — lời gọi mới phải truyền `name=`
  (`test_every_llm_call_site_names_its_purpose` chặn).

## Các lưu ý kỹ thuật

**HuggingFace cache:** Model nằm ở `data/hf_cache/`. Hệ thống có `G:\My Drive\HF_Cache_Models` (Google Drive, thường offline) được set làm `HF_HOME` trong system env. Mọi script cần force-set tất cả HF env vars sang local trước khi import `sentence_transformers`:
```python
for k in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
    os.environ[k] = str(local_hf_path)
```

**Chunk (2026-10-18):** một Điều = một đoạn; điều > 4.000 ký tự tách theo khoản, điều có khoản sửa tại chỗ tách
quanh khoản đó (bỏ qua "2.", "3." nằm trong phần trích “…”). Mọi đoạn mang "Chương — Mục"; mảnh không mở đầu bằng
"Điều N." mang thêm "Điều N. Tiêu đề" + câu dẫn ≤ 300 ký tự (`chunker._article_head`). KHÔNG gắn tên văn bản lên mọi
đoạn — đo: recall@1 tăng nhưng độ phủ câu nhiều điều 0.941 → 0.853 (`reports/chunk_context_budget.json`). Chunker
làm rơi ≤ 6 từ của văn bản đã làm sạch (`TestChunksAreComplete`); kiểm lại toàn corpus bằng phép phủ cụm 8 từ.

**Ngữ cảnh LLM:** mọi câu nhận đủ top_n (8) đoạn, nguyên vẹn, trong ngân sách `_MAX_TOTAL_CHARS = 24_000`; đoạn không
vừa thì bỏ nguyên đoạn. Trước 2026-10-18 câu < 150 ký tự bị cắt còn 5 đoạn và mỗi đoạn cắt ở 3.000 ký tự — 3 câu gold
mất đoạn gold; bỏ cắt + header: accuracy 0.950 → 0.975, p50 đầu cuối 1.5 → 2.1 s.

**Score filtering:** `generator.py` lọc chunk có `score < 0.05` trước khi gọi LLM. Nếu tất cả chunk dưới ngưỡng → trả về message "không tìm thấy" + gợi ý upload, không gọi LLM, không show nguồn.

**LLM temperature:** `0.0` (không phải 0.1) để citation ổn định giữa các lần chạy.

**ChromaDB:** Dùng local `PersistentClient` (không cần server). HTTP server ở `localhost:8000` thường không chạy — `get_chroma_collection()` probe TCP 0.5s rồi đi thẳng xuống local, thay vì chờ `heartbeat()` hết 4.2s. Collection `documind_legal` đang có 1247 chunks (3072-dim, `gemini-embedding-2`), kèm metadata `embedding_model`/`embedding_dim` để phát hiện lệch model.

**Pin chromadb là 0.6.3 — chạy đúng `.venv` của repo:** Python global của máy có chromadb 1.5.9; chỉ cần mở `data/chroma_db/` một lần bằng bản đó là store bị migrate tại chỗ (thêm cột `schema_str`, `config_json_str` về `{}`, `index_metadata.pickle` của HNSW thành dict) và 0.6.3 hết mở nổi — mọi truy vấn trả 0 chunk. Khôi phục: `python scripts/repair_chroma_config.py` để vá config, `python scripts/restore_chroma_from_qdrant.py --yes` để dựng lại chunk+vector từ Qdrant (không re-embed, ~15 phút vì Qdrant free tier ở xa — cluster `australia-southeast1` GCP, theo URL trong log 2026-10-18).

**Đổi model embedding:** sửa `EMBEDDING_MODEL` trong `.env` → `python scripts/reembed_corpus.py --yes` (re-embed tại chỗ, giữ nguyên chunk + metadata temporal, tự sao lưu collection cũ sang `documind_legal__backup_<model cũ>`) → `pytest -q` + `python eval/temporal_eval.py`. So sánh ứng viên trước khi đổi: `python eval/embedding_ab.py --models current <model-moi>`.

**Cache HuggingFace:** mọi entrypoint gọi `use_local_hf_cache()` từ `src/hf_env.py` — không tự set biến env HF nữa (trước đây 8 bản sao, 4 biến thể, đã gây lỗi thật). Riêng embedder truyền `cache_folder` thẳng làm tham số vì `huggingface_hub` đóng băng đường dẫn cache ngay lúc import, mọi thao tác env sau đó là quá muộn.

**Vite proxy:** `frontend/vite.config.ts` proxy `/api` → `http://localhost:8081`. Nếu đổi port backend phải cập nhật cả đây. Production một service Render cùng domain nên không cần `VITE_API_URL` — chỉ set khi tách frontend ra domain khác (`frontend/.env.example`).

**Vector store provider:** mặc định `chroma`. Đổi sang Qdrant Cloud: chạy
`python scripts/migrate_chroma_to_qdrant.py --verify`, rồi set `VECTOR_STORE_PROVIDER=qdrant`
+ `QDRANT_URL` + `QDRANT_API_KEY` trong `.env`. Toàn bộ code retrieval (main.py init, BM25
corpus load, health check, direct-query fallback) đi qua `src/rag/vector_backend.py` nên
không cần sửa gì thêm — nhưng cần chạy migrate lại mỗi khi corpus đổi (script
đọc embeddings có sẵn trong Chroma, không re-embed).

## Eval (RAGAS)

Chỉ chạy khi cần benchmark, không phải production. Gold set đang dùng:
`data/eval/temporal_questions.json` (30 câu lao động, có `source_clause`). Bộ 25 câu ngân
hàng tiền-pivot nằm ở `data/eval/_archive/` — không đo được gì trên corpus này:
```powershell
python eval/run_evals.py --strategies dense hybrid --output reports/ragas_50q.json
```

RAGAS dùng Gemini làm judge — 3 keys × 5 models = 15 pairs, rate limit 3 req/61s/pair.
Checkpoint tự động lưu tại `reports/ragas_checkpoints/` sau mỗi 10 câu.
