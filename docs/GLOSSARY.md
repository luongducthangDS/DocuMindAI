# Bảng thuật ngữ DocuMind — định nghĩa để hiểu sâu

Liệt kê **mọi kỹ thuật đang thật sự có trong repo** (không kê thuật ngữ RAG chung chung).
Mỗi mục: định nghĩa → **vì sao dự án này chọn nó** → chỗ nó sống trong code → câu sẽ bị hỏi.

## Cách dùng

Tự chấm từng mục theo thang của [docs/drills/](drills/README.md):

| Mức | Nghĩa |
|---|---|
| **2** | Định nghĩa được + nói được vì sao dự án dùng + nói được phương án bị loại |
| **1** | Định nghĩa được, không nói được vì sao |
| **0** | Phải tra mới nói được |

Mọi mục ≤1 trong danh sách **§0 Ưu tiên** → viết một DEC trong [docs/decisions/](decisions/README.md).
Mục ≤1 ngoài §0 thì để đó, không phải thứ bị hỏi nhiều.

---

## §0. Ưu tiên — 12 thuật ngữ bị hỏi nhiều nhất

Nếu chỉ có một buổi, học đúng 12 mục này.

| # | Thuật ngữ | Vì sao bị hỏi | Tự chấm |
|---|---|---|---|
| 1 | Chunking theo cấu trúc (§2.3) | Quyết định trần chất lượng của mọi thứ phía sau | ☐ |
| 2 | BM25 Okapi — `k1`, `b`, IDF (§4.1) | Câu hỏi số 1 về sparse retrieval | ☐ |
| 3 | Bi-encoder vs Cross-encoder (§3.2, §4.5) | Phân biệt được = hiểu vì sao phải có 2 tầng | ☐ |
| 4 | RRF và hằng số `k=60` (§4.4) | Nơi sinh ra bug ngưỡng 0.05 của chính repo này | ☐ |
| 5 | Thang điểm và hiệu chuẩn ngưỡng (§4.6) | Bug thật, kể được là điểm cộng lớn | ☐ |
| 6 | Hybrid retrieval — vì sao cần cả hai (§4.3) | "Dense đủ rồi, cần BM25 làm gì?" | ☐ |
| 7 | Grounding và citation `[N]` (§6.4) | Đây là lý do tồn tại của sản phẩm | ☐ |
| 8 | Abstain — từ chối trả lời (§6.5) | Phân biệt RAG nghiêm túc với demo | ☐ |
| 9 | Corrective RAG — vòng grade→reformulate (§1.5) | Vòng lặp trong graph, hay bị hỏi điều kiện dừng | ☐ |
| 10 | Temporal filter, hiệu lực cấp điều khoản (§5.1) | Tính năng khác biệt nhất của dự án | ☐ |
| 11 | hit_rate@k / MRR / faithfulness (§8) | "Anh đo chất lượng bằng gì?" | ☐ |
| 12 | LLM-as-judge và giới hạn của nó (§8.6) | Bẫy kinh điển: dùng LLM chấm LLM | ☐ |

---

## §1. Kiến trúc và orchestration

**1.1 RAG (Retrieval-Augmented Generation)** — nạp tri thức vào LLM ngay lúc hỏi bằng cách truy
hồi văn bản liên quan rồi đưa vào prompt, thay vì nhồi tri thức vào trọng số bằng fine-tune.
*Vì sao ở đây:* văn bản pháp luật đổi liên tục và bắt buộc trích dẫn nguồn — fine-tune không cho
cả hai thứ đó.
*Bị hỏi:* khi nào RAG thua fine-tune? → khi cần đổi **văn phong/định dạng đầu ra**, không phải đổi tri thức.

**1.2 Agentic RAG** — RAG có nhánh rẽ và vòng lặp thay vì đường thẳng retrieve→generate.
Ở đây: phân loại ý định → truy hồi → lọc hiệu lực → chấm → (làm lại hoặc trả lời).
[graph.py:770](../src/agent/graph.py#L770)

**1.3 LangGraph / StateGraph** — thư viện dựng đồ thị trạng thái. `node` là hàm nhận state trả về
phần state cần cập nhật; `edge` là luồng cố định; `conditional_edge` rẽ nhánh theo một hàm.
*Vì sao chọn thay vì chain tuyến tính:* cần **vòng lặp có điều kiện dừng** (§1.5) — chain không
diễn đạt được vòng lặp.
[graph.py:770-803](../src/agent/graph.py#L770)

**1.4 AgentState (TypedDict)** — kiểu dữ liệu chảy qua mọi node: `query`, `retrieved_chunks`,
`answer`, `sources`, `grade`, `retry_count`, `tried_queries`, `as_of_date`, `time_out_of_range`,
`steps`. Mỗi node trả về **dict con**, LangGraph merge vào state.
*Bị hỏi:* vì sao trả dict con chứ không sửa state tại chỗ? → node thuần thì test được và việc
merge nằm ở một chỗ kiểm soát được.
[graph.py:51](../src/agent/graph.py#L51)

**1.5 Corrective RAG (CRAG) / self-correction loop** — sau truy hồi thì **chấm** context; nếu
không liên quan thì **viết lại câu hỏi** và truy hồi lại, tối đa `MAX_RETRIES = 2` lần, hết lượt vẫn
đi tiếp để generator tự abstain.
*Vì sao phải có trần cứng:* vòng lặp không trần là cách nhanh nhất đốt quota và treo request.
[graph.py:756](../src/agent/graph.py#L756)

**1.6 Intent routing** — phân loại câu hỏi thành `simple_qa | compare | summarize | report |
compliance_check | unknown` rồi rẽ nhánh. Hai tầng: **keyword fast-path** chạy trước, LLM chỉ
được gọi khi keyword không quyết được và câu dài hơn 6 từ.
*Vì sao hai tầng:* mỗi lần gọi LLM để phân loại là thêm độ trễ và thêm một lần rủi ro rate-limit
cho việc mà regex làm xong.
*Bị hỏi:* `unknown` đi đâu? → vẫn vào `do_retrieve`; thà thử rồi để generator từ chối còn hơn chặn oan.
[graph.py:238](../src/agent/graph.py#L238), [graph.py:294](../src/agent/graph.py#L294)

**1.7 Query contextualization (coreference resolution)** — viết lại câu phụ thuộc ngữ cảnh
("còn điều 2 thì sao?") thành câu độc lập trước khi truy hồi, dựa trên lịch sử hội thoại.
*Vì sao bắt buộc:* retriever không có bộ nhớ — "điều 2" đứng một mình không truy hồi được gì.
[graph.py:154](../src/agent/graph.py#L154), [graph.py:212](../src/agent/graph.py#L212)

**1.8 Short-term vs long-term memory** — `ShortTermMemory` giữ tối đa 10 lượt trong tiến trình;
`LongTermMemory` là lớp riêng lưu lâu dài.
*Bị hỏi:* vì sao cắt ở 10 lượt? → vừa vì context window, vừa vì lượt cũ kéo contextualization đi sai.
[memory.py:25](../src/agent/memory.py#L25), [memory.py:45](../src/agent/memory.py#L45)

**1.9 Các node hiện có** — `do_contextualize`, `router`, `do_retrieve`, `do_temporal_filter`,
`do_grade`, `do_reformulate`, `do_answer`, `do_compare`, `do_summarize`, `do_report`,
`do_compliance`, `do_persist`.

---

## §2. Ingestion và corpus

**2.1 Corpus** — tập văn bản đã nạp. **Biên giới corpus chính là biên giới câu trả lời**: mọi câu
ngoài biên giới phải bị từ chối, không được suy đoán.

**2.2 Manifest** — khai báo nguồn cho từng văn bản (id, tiêu đề, số hiệu, ngày hiệu lực, trạng
thái, sửa đổi tại chỗ). Là **nguồn sự thật** cho metadata, không để code đoán từ nội dung.
`EFFECTIVE_TO_OPEN = "9999-12-31"` là sentinel "còn hiệu lực vô thời hạn" — chọn sentinel thay vì
`NULL` để mọi so sánh ngày dùng chung một toán tử, không phải rẽ nhánh `None`.
[manifest.py:22](../src/ingestion/manifest.py#L22), [manifest.py:55](../src/ingestion/manifest.py#L55)

**2.3 Structural chunking (cắt theo `Điều`/`Khoản`)** — cắt theo **đơn vị pháp lý**, không theo số
ký tự cố định.
*Vì sao:* một Điều là đơn vị trích dẫn nhỏ nhất mà người đọc chấp nhận; cắt ngang giữa Điều thì
câu trả lời trích ra một mẩu không tự đứng được. Điều dài quá `_MAX_DIEU_CHARS = 4000` mới cắt tiếp.
[chunker.py:194](../src/ingestion/chunker.py#L194), [chunker.py:43](../src/ingestion/chunker.py#L43)

**2.4 Chunk overlap** — hai chunk liền nhau chia sẻ một đoạn để câu bị cắt ngang vẫn xuất hiện
trọn trong ít nhất một chunk. Cấu hình `chunk_size=512`, `chunk_overlap=64`.
*Bị hỏi:* overlap cắt theo ký tự hay token? → theo token thì ghép lại mới không mất chữ.
[config.py:98](../src/config.py#L98)

**2.5 Fallback chunking** — văn bản không có cấu trúc `Điều` (PDF người dùng upload) thì rơi về
cắt theo độ dài. *Vì sao cần:* corpus do người dùng nạp không đảm bảo định dạng.
[chunker.py:306](../src/ingestion/chunker.py#L306)

**2.6 VBHN / văn bản hợp nhất** — bản gộp văn bản gốc với các sửa đổi, mang theo **chú thích chân
trang** và **đoạn trích nguyên văn** của văn bản sửa đổi. Không bóc ra thì một chunk chứa lẫn nội
dung cũ và mới, retriever trả về bản sai mà vẫn trông hợp lệ.
[cleaner.py:80](../src/ingestion/cleaner.py#L80), [cleaner.py:86](../src/ingestion/cleaner.py#L86)

**2.7 Unicode normalization (NFC)** — cùng một chữ tiếng Việt có hai cách mã hoá (dựng sẵn / tổ
hợp). Không chuẩn hoá thì cùng một từ thành hai token, `df` đếm sai, BM25 sai theo.
[cleaner.py:29](../src/ingestion/cleaner.py#L29)

**2.8 Metadata cấp điều khoản** — `clause_uid`, `effective_from`, `effective_to`, `status`,
`dieu_header`, `so_hieu`, `source_url`. Đây là thứ làm cho §5 chạy được.
[chunker.py:78](../src/ingestion/chunker.py#L78), [chunker.py:69](../src/ingestion/chunker.py#L69)

**2.9 In-place amendment** — một văn bản sửa **một khoản** của văn bản khác mà không thay cả văn
bản. Phải xử lý riêng vì lọc ở cấp văn bản sẽ bỏ sót.
[manifest.py:33](../src/ingestion/manifest.py#L33), [chunker.py:130](../src/ingestion/chunker.py#L130)

**2.10 Ingest / index / reset** — đọc → làm sạch → chunk → nhúng → ghi vào vector store. `--reset`
xoá collection trước khi ghi để không còn chunk mồ côi của lần nạp cũ.
`scripts/ingest_documents.py`

**2.11 Crawler** — thu thập văn bản từ nguồn công khai. Đây là chỗ sinh ra rác cần `cleaner` xử lý.
[src/ingestion/congbao_crawler.py](../src/ingestion/congbao_crawler.py)

---

## §3. Embedding và vector store

**3.1 Embedding / vector nhúng** — ánh xạ đoạn văn thành vector số thực sao cho văn bản gần nghĩa
thì gần nhau trong không gian đó. Ở đây `paraphrase-multilingual-MiniLM-L12-v2`, **384 chiều**.
*Vì sao model này:* đa ngữ (có tiếng Việt), chạy CPU, nhẹ đủ để deploy trên máy không GPU.
[config.py:37](../src/config.py#L37), [embedder.py:191](../src/rag/embedder.py#L191)

**3.2 Bi-encoder** — mã hoá query và document **độc lập** rồi so bằng cosine. Nhờ độc lập nên
document nhúng sẵn một lần, query chỉ nhúng lúc hỏi → tìm được trong corpus lớn.
*Đánh đổi:* không nhìn thấy tương tác trực tiếp giữa từng cặp từ query–document, nên kém chính xác
hơn cross-encoder (§4.5). Đây là lý do kiến trúc hai tầng retrieve→rerank tồn tại.

**3.3 Cosine similarity** — độ đo góc giữa hai vector, bỏ qua độ dài. Dùng vì độ dài vector embedding
không mang thông tin ngữ nghĩa.
Có một bản tự cài trong compliance để so vector ngoài vector store: [compliance.py:65](../src/rag/compliance.py#L65)

**3.4 Vector store / collection** — kho lưu vector kèm metadata, hỗ trợ truy vấn k láng giềng gần
nhất. Collection là một không gian tên; `documind_legal`.
[config.py:61](../src/config.py#L61)

**3.5 ChromaDB PersistentClient** — chạy nhúng trong tiến trình, ghi xuống `data/chroma_db/`,
không cần server.
*Vì sao:* bớt một dịch vụ phải vận hành cho môi trường dev.

**3.6 Qdrant** — vector DB dạng dịch vụ, dùng khi deploy. Chọn bằng `VECTOR_STORE_PROVIDER`.

**3.7 Provider abstraction** — `vector_backend.py` gói các thao tác `count_chunks`,
`fetch_all_chunks`, `direct_query`, `make_llamaindex_vector_store` để đổi Chroma↔Qdrant không phải
sửa call site nào.
*Bị hỏi:* vì sao không gọi thẳng client? → đổi provider sẽ phải sửa rải rác ở main.py, BM25 load,
health check, fallback query — mỗi chỗ là một cơ hội quên.
[vector_backend.py:26](../src/rag/vector_backend.py#L26)

**3.8 HF cache / offline model** — model tải sẵn về `data/hf_cache/`. Repo phải force-set
`HF_HOME`/`HF_HUB_CACHE`/`SENTENCE_TRANSFORMERS_HOME` sang thư mục local vì biến môi trường hệ
thống trỏ sang ổ mạng có thể offline.
[retriever.py:102](../src/rag/retriever.py#L102)

**3.9 Remote embedding fallback** — `_HFInferenceAPIEmbedding` nhúng qua API khi không nạp được
model local.
*Cảnh báo phải nhớ:* đổi model nhúng mà không re-index thì vector query và vector trong kho **khác
không gian**, kết quả trả về là rác chứ không phải lỗi.
[embedder.py:37](../src/rag/embedder.py#L37), [embedder.py:34](../src/rag/embedder.py#L34)

---

## §4. Retrieval — phần lõi

**4.1 BM25 Okapi (sparse retrieval)** — xếp hạng theo trùng khớp **từ khoá**, với ba thành phần:
- `idf(t) = ln(1 + (N − df + 0.5)/(df + 0.5))` — từ hiếm đáng giá hơn từ phổ biến.
- **tf saturation** qua `k1`: lặp từ 10 lần không đáng giá gấp 10 lần lặp 1 lần.
- **length normalization** qua `b`: document dài bị phạt, để nó không thắng chỉ nhờ dài.

*Vì sao cần trong dự án luật:* số hiệu văn bản, "Điều 37", "Nghị định 145" là **chuỗi chính xác** —
dense embedding làm nhoè chính xác thứ đó.
Hiện đang dùng `BM25Retriever` của LlamaIndex với tham số mặc định — xem
[DEC-0002](decisions/DEC-0002-retrieval-core-tu-viet.md).
[retriever.py:53](../src/rag/retriever.py#L53)

**4.2 Dense retrieval** — xếp hạng theo khoảng cách vector, bắt được diễn đạt khác chữ nhưng cùng
nghĩa ("nghỉ thai sản" ↔ "chế độ thai sản"). Điểm yếu đối xứng với BM25: nhoè ở số hiệu và tên riêng.

**4.3 Hybrid retrieval** — chạy cả hai rồi hợp nhất.
*Vì sao cả hai:* hai nhánh sai ở hai chỗ khác nhau; câu hỏi thật trộn cả từ khoá chính xác lẫn diễn
đạt tự do. Đây là câu "dense đủ rồi, cần BM25 làm gì" — trả lời bằng ví dụ số hiệu văn bản.

**4.4 RRF (Reciprocal Rank Fusion)** — hợp nhất nhiều bảng xếp hạng bằng
`score(d) = Σ 1/(k + rank_i(d))`, `k = 60`.
*Vì sao bỏ điểm gốc chỉ giữ thứ hạng:* cosine và BM25 không cùng đơn vị, cộng thẳng là cộng táo với
cam; thứ hạng thì so được.
*`k` làm gì:* làm phẳng chênh lệch giữa các hạng đầu. `k` nhỏ → hạng 1 áp đảo; `k` lớn → mọi hạng gần
như ngang nhau, fusion thành đếm phiếu.
*Hệ quả phải thuộc:* điểm RRF với 2 list tối đa chỉ `2/61 ≈ 0.0328` — xem §4.6.

**4.5 Cross-encoder reranker** — đưa **cặp (query, document) vào cùng một lần forward** nên model
thấy tương tác trực tiếp giữa các từ, chính xác hơn bi-encoder nhiều.
*Vì sao chỉ rerank được ~20 ứng viên:* chi phí tuyến tính theo số ứng viên và không nhúng trước được
— chấm cả corpus là bất khả thi. Model: `BAAI/bge-reranker-v2-m3`.
[config.py:88](../src/config.py#L88), [retriever.py:102](../src/rag/retriever.py#L102)

**4.6 Thang điểm và hiệu chuẩn ngưỡng** — mục quan trọng nhất của repo này. Ba thang điểm cùng tồn tại:

| Nguồn điểm | Khoảng giá trị điển hình |
|---|---|
| Cross-encoder rerank | rộng, chunk liên quan vượt hẳn 0.05 |
| RRF thô | `≈ 1/(60+rank) ≈ 0.016` |
| BM25 thô | thang khác hẳn, không chặn trên |

Ngưỡng abstain `0.05` hiệu chuẩn cho thang cross-encoder. Áp nó lên thang RRF thì **mọi chunk đều
bị loại** và hệ thống từ chối mọi câu. Vì thế `_effective_min_score()` đọc trạng thái runtime
`_reranker_active` chứ không đọc cờ config — reranker có thể được *bật trong config* nhưng *nạp hỏng*
lúc chạy.
[generator.py:68](../src/rag/generator.py#L68), [generator.py:71](../src/rag/generator.py#L71), [retriever.py:24](../src/rag/retriever.py#L24)

**4.7 Candidate pool `top_k` vs kết quả `top_n`** — `top_k=20` là số ứng viên kéo ra để reranker có
đủ nguyên liệu; `top_n=8` là số chunk cuối cùng đưa vào prompt.
*Bị hỏi:* tăng `top_k` thì được gì mất gì? → recall tăng, nhưng chi phí rerank tăng tuyến tính và
nhiễu vào prompt cũng tăng.

**4.8 Fallback path** — `retrieve_direct_chroma()` truy vấn thẳng vector store khi pipeline
LlamaIndex chưa dựng được. Tên còn chữ `_chroma` vì lý do tương thích ngược, thực tế chạy được cả
Qdrant qua `vector_backend`.
[retriever.py:204](../src/rag/retriever.py#L204)

**4.9 Deduplication** — loại chunk trùng sau fusion trước khi đưa vào prompt; hai nhánh retrieval
hay trả về cùng một chunk.
[graph.py:350](../src/agent/graph.py#L350)

**4.10 Query reformulation** — viết lại câu hỏi khi grade nói context không liên quan, có giữ
`tried_queries` để không lặp lại đúng câu cũ.
[graph.py:438](../src/agent/graph.py#L438), [graph.py:448](../src/agent/graph.py#L448)

---

## §5. Tầng thời gian

**5.1 Temporal-aware retrieval / hiệu lực cấp điều khoản** — chỉ để retrieval thấy bản điều khoản
**có hiệu lực tại `as_of_date`**. Quy tắc:
```
effective_from <= T  AND  (effective_to rỗng OR effective_to > T)  AND  status != 'repealed'
```
*Vì sao là node riêng chứ không nhét vào prompt:* xem [DEC-0001](decisions/DEC-0001-temporal-filter-node.md)
— có số chứng minh prompt-only không làm sạch context.
*Tên gọi đúng:* "temporal-aware retrieval", **không** gọi "point-in-time reconstruction".
[temporal.py:46](../src/rag/temporal.py#L46), [temporal.py:63](../src/rag/temporal.py#L63)

**5.2 `as_of_date`** — mốc thời điểm tra cứu, mặc định hôm nay, chảy từ UI → `/query` →
`AgentState` → filter → generator.
[graph.py:71](../src/agent/graph.py#L71)

**5.3 `time_out_of_range`** — cờ bật khi `as_of_date` sớm hơn mốc sớm nhất corpus phủ
(`corpus_earliest_point_in_time`). Khi bật thì trả lời thẳng "ngoài phạm vi", **không gọi LLM**.
*Vì sao:* trả lời cho mốc corpus không phủ tức là trình bày luật đời sau như thể áp dụng cho đời trước.
[temporal.py:123](../src/rag/temporal.py#L123), [manifest.py:180](../src/ingestion/manifest.py#L180), [generator.py:313](../src/rag/generator.py#L313)

**5.4 `status` của điều khoản** — `repealed` (đã bãi bỏ) bị loại khỏi tập ứng viên.
[temporal.py:31](../src/rag/temporal.py#L31), [chunker.py:69](../src/ingestion/chunker.py#L69)

---

## §6. Generation

**6.1 Provider cascade** — Groq (`llama-3.3-70b-versatile`) → Gemini → endpoint tương thích OpenAI →
**extractive fallback**. Mỗi nấc phòng một tình huống khác nhau (hết quota ngày, rate-limit, chết cả ba).
*Nấc cuối đáng nói nhất:* khi không LLM nào trả lời, vẫn trích thẳng 5 nguồn đầu cho người dùng — chunk
đã tìm được vẫn có giá trị tra cứu.
[generator.py:313](../src/rag/generator.py#L313), [generator.py:153](../src/rag/generator.py#L153)

**6.2 System prompt** — chỉ thị cố định đặt trước context, quy định "chỉ trả lời dựa trên đoạn được
cung cấp".
[generator.py:26](../src/rag/generator.py#L26)

**6.3 Temperature 0.0** — tắt lấy mẫu ngẫu nhiên. *Vì sao không phải 0.1:* để **citation ổn định
giữa các lần chạy** — cùng câu hỏi phải ra cùng số trích dẫn, nếu không thì eval không so sánh được.

**6.4 Grounding và citation `[N]`** — mỗi chunk vào prompt được đánh số, model bắt buộc trích `[N]`.
`_cited_sources()` chỉ trả về nguồn **thật sự được trích**.
*Vì sao quan trọng:* không có nó, một lời từ chối do model tự diễn đạt vẫn đính kèm đủ 8 nguồn, khiến
lời từ chối trông như câu trả lời có căn cứ.
[generator.py:96](../src/rag/generator.py#L96)

**6.5 Abstain / từ chối trả lời** — ba đường độc lập dẫn tới từ chối: không có chunk nào; mọi chunk
dưới ngưỡng liên quan (**không gọi LLM**); `time_out_of_range`.
*Bị hỏi:* vì sao từ chối trước khi gọi LLM chứ không để LLM tự từ chối? → tiết kiệm một lần gọi, và
quan trọng hơn: kết quả **xác định**, test được.
[generator.py:313](../src/rag/generator.py#L313)

**6.6 Context window budgeting** — `_MAX_CHUNK_CHARS = 3000` mỗi chunk, `_MAX_TOTAL_CHARS = 15000`
tổng; cắt chunk trước, rồi dừng khi chạm trần tổng.
*Bị hỏi:* vì sao cắt theo ký tự chứ không token? → xấp xỉ rẻ, không cần tokenizer của từng provider;
cái giá là ước lượng lệch giữa các model.
[generator.py:58](../src/rag/generator.py#L58), [generator.py:134](../src/rag/generator.py#L134)

**6.7 Streaming** — trả lời theo từng mẩu qua WebSocket thay vì chờ trọn câu.
[generator.py:420](../src/rag/generator.py#L420)

**6.8 Extractive vs abstractive** — extractive là trích nguyên văn, abstractive là diễn đạt lại.
Dự án chạy abstractive có trích dẫn, và rơi về extractive khi LLM chết.

---

## §7. Guardrails, an toàn, vận hành

**7.1 Prompt injection** — nội dung do người dùng nhập chứa chỉ thị nhằm cướp quyền điều khiển
("bỏ qua hướng dẫn trên..."). Phát hiện bằng động từ mệnh lệnh + danh từ tự tham chiếu.
[guardrails.py:54](../src/guardrails.py#L54)

**7.2 Homoglyph attack** — trộn ký tự Cyrillic/Hy Lạp nhìn giống chữ Latin để né bộ lọc từ khoá.
[guardrails.py:51](../src/guardrails.py#L51)

**7.3 Citation validation** — hậu kiểm: câu trả lời trích `[N]` mà `N` vượt số chunk thật thì đó là
**trích dẫn bịa**; gắn cảnh báo.
[guardrails.py:103](../src/guardrails.py#L103)

**7.4 Hallucination / bịa** — model sinh nội dung không có trong nguồn. Trong dự án này chống bằng
ba lớp: system prompt, ngưỡng abstain, và citation validation.

**7.5 Rate limiting (slowapi)** — trần `rate_limit_per_minute = 10` mỗi client.
[main.py:173](../src/api/main.py#L173), [config.py:75](../src/config.py#L75)

**7.6 CORS** — trình duyệt chỉ cho frontend gọi backend khác origin khi backend khai báo cho phép.
Liên quan trực tiếp tới deploy tách domain Vercel/Render.
[main.py:189](../src/api/main.py#L189)

**7.7 Lifespan / warm-up** — nạp model và dựng retriever lúc khởi động thay vì lúc request đầu, vì
lần nạp đầu mất khoảng 25 giây.
[main.py:34](../src/api/main.py#L34)

**7.8 Structured logging (JSONL)** — mỗi query ghi một dòng JSON vào `logs/chat_history.jsonl`; dòng
là bản ghi máy đọc được, không phải câu tiếng Anh cho người đọc.
[graph.py:722](../src/agent/graph.py#L722)

**7.9 LangSmith tracing** — `@traceable` bọc `run_agent` thành một **run**, mỗi lần gọi LLM là một
**span** con, xem lại được prompt thật đã gửi.
*Vì sao cần:* không có trace thì debug RAG là đoán mò — không biết câu trả lời sai do retrieval hay
do generation.
[graph.py:819](../src/agent/graph.py#L819)

**7.10 Pydantic Settings, `extra_forbidden`** — cấu hình đọc từ `.env` có kiểm kiểu; biến lạ trong
`.env` làm app **từ chối khởi động** thay vì chạy với cấu hình sai âm thầm.
[config.py](../src/config.py)

**7.11 Scope gate** — bộ chặn câu hỏi ngoài lĩnh vực corpus. Hiện **mới có spec, chưa có code**:
[SPEC-scope-gate.md](spec/SPEC-scope-gate.md). Nói đúng trạng thái này khi bị hỏi.

---

## §8. Đánh giá

**8.1 Gold set** — bộ câu hỏi kèm đáp án đúng và chunk đúng, soạn **trước** khi đo. Hiện có 30 câu
temporal (`data/eval/temporal_questions.json`).
*Bị hỏi:* n=30 có đủ không? → không đủ cho khoảng tin cậy hẹp; vì thế phải nhìn cả các đại lượng đo
trực tiếp cơ chế như `context_distractor` chứ không chỉ accuracy.

**8.2 hit_rate@k** — tỉ lệ câu có ít nhất một chunk đúng nằm trong top-k. Đo **recall của retrieval**,
độc lập với LLM.
[metrics.py:109](../eval/metrics.py#L109)

**8.3 MRR (Mean Reciprocal Rank)** — trung bình `1/thứ hạng của kết quả đúng đầu tiên`. Khác
hit_rate ở chỗ nó **quan tâm vị trí**: đúng ở hạng 1 hơn hẳn đúng ở hạng 8.
[metrics.py:133](../eval/metrics.py#L133)

**8.4 citation_rate** — tỉ lệ câu trả lời có trích dẫn `[N]`.
[metrics.py:222](../eval/metrics.py#L222)

**8.5 ooc_refusal_rate (out-of-corpus)** — tỉ lệ từ chối đúng với câu ngoài phạm vi. Đây là metric
đo **đúng mục tiêu sản phẩm**: trả lời bừa còn tệ hơn không trả lời.
[metrics.py:182](../eval/metrics.py#L182)

**8.6 LLM-as-judge** — dùng LLM chấm đầu ra của LLM, cả trong grader lúc chạy thật lẫn trong RAGAS
lúc eval.
*Giới hạn phải tự nêu ra:* judge cùng họ model với generator thì thiên vị; judge chấm không ổn định
giữa các lần; và judge sai theo cùng kiểu mà generator sai. Nên mọi kết luận phải có thêm ít nhất một
metric cơ học (hit_rate, distractor rate) không đi qua LLM.
[grader.py:77](../src/rag/grader.py#L77)

**8.7 RAGAS** — bộ metric chuẩn: `faithfulness` (câu trả lời có bám nguồn không),
`answer_relevancy` (có trả lời đúng câu hỏi không), `context_precision` (context có sạch không),
`context_recall` (context có đủ không). Ngưỡng đặt trong repo: faithfulness 0.80, answer_relevancy 0.75.
[eval/ragas_eval.py:101](../eval/ragas_eval.py#L101)

**8.8 Ablation / A-B harness** — chạy cùng một bộ câu hỏi qua nhiều cấu hình để quy kết tác dụng cho
từng thành phần. Bốn chiến lược đang có: `bm25_only`, `dense_only`, `hybrid`, `hybrid+rerank`.
*Vì sao phải ablation:* không có nó thì không phân biệt được "nhờ reranker" với "nhờ đổi prompt".
[eval/rag_comparison.py](../eval/rag_comparison.py)

**8.9 context_gold / context_distractor / context_clean** — ba đại lượng của A/B temporal: context
có chứa chunk đúng không / có lẫn bản hết hiệu lực không / có sạch hoàn toàn không. Đo **cơ chế**,
không qua LLM.
`reports/temporal_eval.json`

**8.10 answer_correctness (F1 token overlap)** — metric cục bộ, so tập token giữa câu trả lời và đáp
án. Rẻ và ổn định, nhưng không hiểu diễn đạt khác chữ — dùng kèm chứ không thay LLM-judge.
[metrics.py:48](../eval/metrics.py#L48)

**8.11 Checkpointing khi eval** — lưu kết quả từng phần vì eval qua API bị rate-limit và dễ đứt giữa chừng.
`reports/ragas_checkpoints/`

---

## §9. Compliance engine

**9.1 Rule-based compliance check** — đối chiếu tình huống người dùng với tiêu chí định lượng
pass/fail trong `data/compliance/criteria.json`, **không để LLM tự phán**.
*Vì sao tách khỏi LLM:* kết luận "đủ điều kiện / không đủ" phải tái lập được và giải thích được bằng
con số, không phải bằng văn phong.
[compliance.py:175](../src/rag/compliance.py#L175)

**9.2 Criteria matching hai tầng** — khớp theo từ khoá trước, không được thì khớp theo embedding với
ngưỡng `_EMBEDDING_MATCH_THRESHOLD = 0.45`.
[compliance.py:94](../src/rag/compliance.py#L94), [compliance.py:47](../src/rag/compliance.py#L47)

**9.3 Value extraction** — rút con số từ câu hỏi tự nhiên ("lương 8 triệu") bằng regex trước, LLM sau.
*Vì sao regex trước:* đa số ca là số đứng cạnh nhãn, regex cho kết quả xác định và miễn phí.
[compliance.py:37](../src/rag/compliance.py#L37), [compliance.py:115](../src/rag/compliance.py#L115)

**9.4 Operator table** — bảng ánh xạ toán tử so sánh trong JSON tiêu chí sang hàm Python, để thêm
tiêu chí là sửa dữ liệu chứ không sửa code.
[compliance.py:24](../src/rag/compliance.py#L24)

---

## §10. Nền tảng và triển khai

**10.1 FastAPI, Pydantic schema** — khai báo kiểu cho request/response, tự sinh `/docs`.
[src/api/schemas.py](../src/api/schemas.py)

**10.2 Uvicorn (ASGI)** — server chạy ứng dụng async. ASGI khác WSGI ở chỗ giữ được kết nối dài
(WebSocket, streaming).

**10.3 async / await trong RAG** — truy hồi và gọi LLM đều là I/O chờ mạng; async cho phép phục vụ
request khác trong lúc chờ.
*Bẫy phải biết:* embedding và reranker là **CPU-bound** — chúng chặn event loop, không async hoá được
bằng cách thêm `await`.

**10.4 GZip middleware** — nén response lớn hơn 1000 byte.
[main.py:196](../src/api/main.py#L196)

**10.5 Vite dev proxy** — frontend `:5174` proxy `/api` sang backend `:8081` để tránh CORS lúc dev;
khi deploy tách domain thì dùng `VITE_API_URL` thay proxy.
`frontend/vite.config.ts`

**10.6 Docker multi-stage, CPU-only torch** — cài `torch==2.4.1+cpu` từ index riêng để khỏi kéo về
~2GB CUDA cho máy không GPU.
`requirements.txt` dòng đầu, `Dockerfile`

**10.7 Render / Vercel** — backend container trên Render, frontend tĩnh trên Vercel.
`render.yaml`, `frontend/vercel.json`

**10.8 Cold start** — instance ngủ dậy phải nạp lại model; đây là lý do reranker có thể **nạp hỏng**
trên gói free và kích hoạt đúng bug §4.6.

---

## Sau khi tự chấm xong

1. Đếm số mục ≤1 **trong §0**. Đó là danh sách DEC phải viết, theo thứ tự ưu tiên có sẵn.
2. Các mục §4 (4.1, 4.4, 4.5, 4.6) không học bằng đọc — làm
   [exercises/retrieval_core/](../exercises/retrieval_core/README.md).
3. Chạy [drill](drills/README.md) trên một file ngẫu nhiên, đối chiếu xem thuật ngữ đã chấm mức 2 có
   thật sự nói ra được khi đóng file hay không.
