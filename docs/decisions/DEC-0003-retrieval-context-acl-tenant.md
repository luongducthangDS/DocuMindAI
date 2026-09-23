# DEC-0003: Đẩy ACL, tenant và hiệu lực thời điểm xuống vector store qua một `RetrievalContext` duy nhất

- **Trạng thái:** Accepted
- **Ngày:** 2026-09-23
- **Spec liên quan:** `docs/spec/SPEC-temporal-retrieval.md` (mở rộng ranh giới đã chốt ở đó)
- **Ảnh hưởng tới:** `src/rag/context.py` (mới), `src/api/principal.py` (mới),
  `src/rag/retriever.py`, `src/rag/vector_backend.py`, `src/agent/graph.py`,
  `src/agent/tools.py`, `src/api/routes/query.py`, `src/api/routes/documents.py`

## 1. Bối cảnh — vấn đề gì buộc phải quyết

`retriever.retrieve(query)` chỉ nhận một chuỗi. Không danh tính, không tenant, không
thời điểm. Mọi thứ khác lọc **sau** khi đã lấy về.

Với hiệu lực thời điểm, cái giá là recall, đo được trên 30 câu gold
(`reports/temporal_eval.json`): lọc sau một lượt retrieve top-8 còn lại
**avg_chunks = 3.23**. Reranker tiêu 4.8/8 suất cho điều khoản chưa/không còn hiệu lực.

Với ACL thì cái giá không phải recall mà là **rò rỉ**. Một chunk bị cấm đi tới
`state["retrieved_chunks"]` là đã qua reranker, qua span Langfuse và qua nhánh fallback
`graph.py::_fallback_to_retrieval_answer` trước khi có gì đó loại nó ra.

Khảo sát tìm được **ba** đường retrieval song song, không đường nào biết danh tính:
1. `retrieve_node` trong graph (đường REST).
2. Handler WebSocket trong `query.py` — tự retrieve riêng, và đây là đường UI thật sự dùng.
3. `search_legal_docs` trong `tools.py` — bind vào LLM, nên **model** tự quyết khi nào gọi.

## 2. Các phương án đã cân nhắc

| # | Phương án | Được | Mất |
|---|---|---|---|
| A | **Một `RetrievalContext` compile thành filter đẩy xuống store** | 3 trục dùng chung 1 cơ chế; ACL chặn trước khi chunk tồn tại trong tiến trình; recall hồi phục | Phải mirror ngày sang int; BM25 vẫn phải lọc ở tầng ứng dụng |
| B | Collection riêng cho mỗi tenant | Cách ly tuyệt đối ở tầng hạ tầng | BM25 nạp toàn bộ node lúc khởi động (`main.py::_load_nodes_from_backend`) — RAM tăng tuyến tính theo số tenant; corpus luật dùng chung bị nhân bản |
| C | Giữ post-filter, chỉ thêm lọc ACL sau retrieve | Diff nhỏ nhất | Không sửa được recall; và một ACL chỉ đúng khi mọi tầng phía trên cư xử đúng thì không phải ACL |
| D | Không làm gì | — | Ba đường retrieval không danh tính vẫn còn nguyên |

## 3. Quyết định

Chọn **A**.

**Vì sao:** ACL, multi-tenant và hiệu lực thời điểm nhìn thì là ba yêu cầu, nhưng ở tầng
lưu trữ chúng là **một**: vị từ trên metadata của chunk, áp trước khi search chạy. Xây ba
hệ thống cho một cơ chế là tự tạo ba chỗ để lệch nhau.

**Ràng buộc loại B:** BM25 giữ một index in-process cho toàn corpus; tách collection theo
tenant biến nó thành N index, trong khi 20 văn bản luật là tài sản **chung** của mọi tenant.

**Ràng buộc kỹ thuật phải vòng qua:** chromadb 0.6.3 từ chối `$lte`/`$gt` trên chuỗi
(`Expected operand value to be an int or a float`) — đã thử thật, không suy đoán. Vì vậy
`effective_from` / `effective_to` được soi chiếu sang số nguyên YYYYMMDD
(`effective_from_i`, `effective_to_i`). Trường chuỗi giữ nguyên; `src/rag/temporal.py` vẫn
đọc chúng và **vẫn chạy như lớp hai**.

**Dấu đóng ở điểm ghi, không ở chunker:** `chunker.py` ghi đè `effective_to` cho các bản
khoản đã bị thay sau khi dựng metadata. Tính số nguyên sớm hơn sẽ đóng băng giá trị
trước-khi-sửa, và một khoản hết hiệu lực mang `effective_to_i = 99991231` sẽ đi thẳng qua
bộ lọc — đúng thứ mà toàn bộ việc này sinh ra để chặn.

**Danh tính do máy chủ quyết:** client trình **API key**, không bao giờ trình quyền của
mình. Nếu body request mang được `acl_labels` thì bất kỳ ai cũng tự cấp cho mình
`confidential` và cả chuỗi lọc phía dưới sẽ trung thành thi hành.

**Mặc định là public, không phải không giới hạn:** `ContextVar` cho các đường không
truyền tham số được (tool do LLM gọi) mặc định `PUBLIC_CONTEXT`. Mặc định "không giới hạn"
sẽ biến mọi entry point mới thành một lỗ rò cho tới khi có người nhớ ra phải bịt.

## 4. Bằng chứng

30 câu gold, `--retrieval-only`, cùng retriever production (`reports/temporal_eval_prefilter.json`):

| Cấu hình | context_gold | context_distractor | context_clean | avg_chunks | retrieve_ms (median) | n |
|---|---|---|---|---|---|---|
| `temporal_filter` (lọc sau) | 0,967 | 0,000 | 0,967 | 3,23 | 500 | 30 |
| `pre_filter` (đẩy xuống) | 0,967 | 0,000 | 0,967 | **7,47** | 536 | 30 |

Cùng độ sạch, **gấp 2,3 lần lượng ngữ cảnh hợp lệ**, trả thêm 36 ms (~7%).

`answer_accuracy` (chạy đủ LLM, cùng file): **1,000 cả hai arm**, không câu nào khác nhau.
Gold set 30 câu đã bão hoà nên **không** chứng minh được pre-filter trả lời đúng hơn — lợi ích
đo được chỉ là dư địa ngữ cảnh. Cần câu hỏi đòi nhiều điều khoản cùng lúc mới phân biệt được.

Cách ly kiểm trên store thật, không chỉ trên dict:
`tests/test_access_control.py::TestWhereClauseAgainstRealStore` — 31 test, gồm các khẳng
định phủ định (tenant khác / nhãn không nắm / văn bản đã bãi bỏ **không** xuất hiện).

## 5. Hệ quả

- **Phải làm khi corpus đổi:** `python scripts/backfill_access_meta.py --yes` cho chunk cũ;
  chunk mới đã được đóng dấu sẵn trong `ingest_documents.py` và đường upload.
- **Đổi sang Qdrant:** `where_to_qdrant_filter` dịch đúng 5 toán tử đang dùng và **ném lỗi**
  với toán tử lạ thay vì bỏ qua — một filter bị bỏ qua trong khi có mệnh đề ACL không phải
  là tìm kiếm kém đi, mà là rò dữ liệu.
- **UI:** ô nhập API key ở sidebar, hiển thị tenant + nhãn từ `GET /whoami`. Đường WS mang
  key trong message vì trình duyệt không đặt được header lên handshake WebSocket — bản đầu
  của DEC này đọc key từ header WS, tức là từ trình duyệt thì key không bao giờ tới server.
- **`GET /documents` lọc theo tenant:** lọc chunk thôi chưa đủ, danh sách tên tài liệu upload
  của mọi tenant cũng là rò rỉ.
- **Khoá API trong `data/tenants/tenants.json` là khoá demo**, commit có chủ đích để clone
  về chạy được ngay. Bản triển khai thật phải lấy principal từ identity provider.
