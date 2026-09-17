# Bài tập: tự viết lõi retrieval

Quyết định đứng sau bài này: [`DEC-0002`](../../docs/decisions/DEC-0002-retrieval-core-tu-viet.md).

## Luật

- **Không dùng agent** sinh code cho `core.py`. Không copy từ `src/`, không copy từ
  LlamaIndex/rank_bm25. Bí thì đọc công thức dưới đây, không đọc code người khác.
- **Chỉ stdlib**: `math`, `re`, `collections`, `unicodedata`, `dataclasses`, `typing`.
  Không `numpy`, không `torch`, không `llama_index`.
- Mục tiêu độ dài: **~200 dòng** cho cả 4 phần. Dài hơn 300 là đang làm quá.
- Xong Phần nào chạy test Phần đó:
  ```bash
  pytest exercises/retrieval_core -q --no-cov -p no:cacheprovider
  ```
  Thư mục này nằm ngoài `testpaths` của `pytest.ini` nên không ảnh hưởng suite chính.

## Thứ tự làm

| Phần | Hàm | Ước lượng | Vì sao phải tự viết |
|---|---|---|---|
| 1 | `tokenize` | **đã cho sẵn** | Làm mẫu phong cách + cố định cách đếm token cho test |
| 2 | `split_dieu`, `chunk_by_dieu` | ~60 dòng | Chunking quyết định trần chất lượng của mọi thứ phía sau |
| 3 | `BM25Index` | ~60 dòng | Câu hỏi phỏng vấn số 1 về sparse retrieval |
| 4 | `rrf_fuse` | ~20 dòng | Nơi hằng số `k=60` sinh ra, và là gốc của bug ngưỡng 0.05 |
| 5 | `rerank` | ~25 dòng | Phân biệt bi-encoder vs cross-encoder |

Làm đúng thứ tự. Phần 3 và 4 là phần bị hỏi nhiều nhất.

---

## Đặc tả

### Phần 2 — Chunking theo `Điều`

`split_dieu(text) -> list[tuple[str, str]]`: tách văn bản thành các cặp `(header, body)`.
Header là dòng bắt đầu bằng `Điều <số>.` hoặc `Điều <số>:`. Phần văn bản trước `Điều` đầu
tiên bỏ qua.

`chunk_by_dieu(text, max_tokens=220, overlap_tokens=40) -> list[Chunk]` phải thoả **5 bất biến**:

1. Không chunk nào chứa nội dung của hai Điều khác nhau.
2. Mỗi chunk **mở đầu bằng header** của Điều nó thuộc — chunk phải tự đứng được khi bị
   nhét vào prompt tách rời khỏi các chunk khác.
3. `len(tokenize(chunk.text)) <= max_tokens`, trừ khi riêng header đã dài hơn `max_tokens`.
4. Một Điều dài bị cắt thành nhiều chunk; hai chunk liên tiếp **cùng một Điều** chồng nhau
   đúng `overlap_tokens` token phần thân.
5. Không mất token: ghép phần thân của các chunk cùng Điều theo thứ tự (đã trừ overlap) phải
   khôi phục đúng dãy token gốc của Điều đó.

> Bất biến 2 và 5 là nơi phần lớn implementation tự viết sai. Bất biến 5 là lý do overlap
> phải cắt theo **token**, không theo ký tự.

### Phần 3 — BM25 (Okapi)

Hằng số: `k1 = 1.5`, `b = 0.75`. Công thức phải khớp **đúng** bản này, vì test neo số thật:

```
avgdl = trung bình độ dài (số token) của các document

idf(t)  = ln( 1 + (N - df(t) + 0.5) / (df(t) + 0.5) )

score(q, d) = Σ  idf(t) · ( f(t,d) · (k1 + 1) )
             t∈q   ───────────────────────────────────────
                    f(t,d) + k1 · ( 1 - b + b · |d| / avgdl )
```

- `f(t,d)`: số lần `t` xuất hiện trong `d`. `df(t)`: số document chứa `t`. `N`: tổng document.
- Term không có trong `d` đóng góp 0 (bỏ qua, không cộng idf).
- Term không có trong toàn corpus: `df=0` → idf vẫn tính được theo công thức trên, nhưng
  `f=0` nên đóng góp 0.
- Hiệu năng: tính `idf` và `|d|` **một lần lúc build**, không tính lại mỗi query. Query
  quét theo **inverted index** (`term -> [(doc_id, tf)]`), không quét toàn bộ document.

Tự trả lời trước khi code — viết câu trả lời vào `notes.md`:
- `b = 0` thì công thức thành gì? Corpus toàn điều luật dài ngắn rất lệch thì nên tăng hay giảm `b`?
- `k1` lớn lên thì tf lặp lại 10 lần có giá trị gấp bao nhiêu lần tf = 1?

### Phần 4 — RRF

```
rrf(d) = Σ  1 / (k + rank_i(d))        k = 60, rank tính từ 1
        i∈lists
```

`rrf_fuse(rankings: list[list[str]], k=60, top_k=None) -> list[tuple[str, float]]`,
sắp giảm dần theo điểm. Document vắng mặt trong một list thì list đó **không đóng góp gì**
(không phải cộng 0 rồi chia, và cũng không phải gán rank = ∞).

Tự trả lời trước khi code:
- Vì sao RRF **bỏ điểm gốc** của từng retriever, chỉ giữ thứ hạng? (gợi ý: cosine và BM25
  không cùng đơn vị)
- `k = 60` đang làm gì? Cho `k = 0` và `k = 1000` thì kết quả biến thành gì?
- Điểm RRF tối đa với 2 list là bao nhiêu? So với ngưỡng `0.05` của `generator.py` thì sao?
  **Đây chính là bug đã có thật trong repo** — tự tính ra nó.

### Phần 5 — Rerank

`rerank(query, candidates, scorer, top_n=8)`. `scorer` là callable
`list[tuple[str, str]] -> list[float]` nhận các cặp `(query, doc_text)` — test truyền hàm
giả, không nạp model.

Yêu cầu:
- Gọi `scorer` **đúng một lần** với cả batch, không gọi từng cặp trong vòng lặp.
- Trả về `top_n` ứng viên điểm cao nhất, **giữ nguyên object ứng viên**, gắn điểm mới.
- `candidates` rỗng → trả list rỗng, không nổ.

Tự trả lời trước khi code:
- Vì sao cross-encoder không dùng được để đánh index toàn corpus, mà chỉ rerank được ~20 ứng viên?
- Sau rerank, điểm nằm trên thang nào? Liên hệ với câu hỏi cuối Phần 4.

---

## Xong thì làm gì

1. `notes.md` — các câu tự trả lời ở trên, viết bằng chữ của mình.
2. Điền mục 4 và mục 6 của [`DEC-0002`](../../docs/decisions/DEC-0002-retrieval-core-tu-viet.md),
   chuyển trạng thái `Proposed` → `Accepted`.
3. Chạy drill lại trên `src/rag/retriever.py` (`docs/drills/`) — lần này phải đạt mức 2 ở cả 8 câu.
