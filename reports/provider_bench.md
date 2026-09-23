# Báo cáo 2026-09-23 — Qdrant Cloud vs Chroma Cloud (tầng vector store)

> Nguồn số: `reports/provider_bench.json` · Harness: `eval/provider_bench.py` ·
> Gold set: `data/eval/legal_qa_200.json` (199 câu) · Corpus 1151 chunk / 20 file, `gemini-embedding-001` (3072-dim) ·
> Chạy 2026-09-23 22:18, git `f1c9d68` + thay đổi chưa commit · MLflow experiment `documind-provider-bench`, run `provider-bench-1151`.

## 1. Kết luận

**Qdrant Cloud thắng về tốc độ, chất lượng truy hồi hai bên như nhau.** Production giữ Qdrant.

- Recall/MRR theo gold clause trùng khít từng chữ số giữa exact, Qdrant và Chroma Cloud.
- Filter đẩy xuống (tenant/ACL/`as_of_date`) đúng 100% ở cả hai, không có kết quả trả rỗng sai.
- Qdrant p95 **324 ms** so với Chroma Cloud **1336 ms**. Phần lớn chênh lệch nằm ở **mạng**
  (vị trí cluster so với máy đo), không phải engine.

## 2. Thiết kế phép đo

Chỉ đo tầng khác nhau giữa hai provider. BM25, reranker và LLM giống hệt nhau ở hai bên, đưa
vào chỉ thêm nhiễu.

| Điều kiện | Cách đảm bảo |
|---|---|
| Cùng dữ liệu | Hai cloud đều copy từ Chroma local (`migrate_chroma_to_qdrant.py`, `copy_chroma_to_cloud.py`), không embed lại. Script kiểm **tập id** khớp 1151/1151 trước khi đo, lệch thì dừng |
| Cùng query | Query embedding đọc từ cache `reports/_eval_cache/query_embeddings.json`, hai bên nhận cùng vector, không gọi Gemini |
| Cùng filter | `RetrievalContext(as_of_date=...).to_where()` như production; Qdrant dịch qua `where_to_qdrant_filter`, Chroma nhận nguyên dict |
| Mốc chuẩn | "Exact" = cosine vét cạn numpy trên embeddings local, áp **cùng predicate** đã gửi provider (`matches()`, có test) |
| Công bằng về mạng | Mỗi query chạy trên hai provider xen kẽ, thứ tự ngẫu nhiên (seed 13), 3 lần lặp; bỏ 3 query warm-up |

Hai mode: `filtered` (như production) và `unfiltered` (đo index thuần). Top-k = 10.

## 3. Kết quả

### Chất lượng

| Mode | Chỉ số | Exact | Qdrant | Chroma Cloud |
|---|---|---:|---:|---:|
| filtered | ANN recall@10 so với exact | 1.000 | 1.000 | 0.998 |
| filtered | Recall@1 / @3 / @5 / @8 | 0.667 / 0.914 / 0.962 / 0.973 | y hệt | y hệt |
| filtered | MRR | 0.794 | 0.794 | 0.794 |
| filtered | Kết quả vi phạm filter · query rỗng | — | 0% · 0% | 0% · 0% |
| unfiltered | ANN recall@10 so với exact | 1.000 | 1.000 | 0.998 |

Chroma Cloud lệch exact ở 3 câu mỗi mode (filtered: `g2_a049`, `g2_c026`, `g2_d030`;
unfiltered: `g2_a049`, `g2_a050`, `g2_d030`), mỗi câu thiếu 1–3 chunk ở cuối top-10. Không câu
nào đổi rank của gold clause: `g2_c026` vẫn rank 1; `g2_a049`, `g2_d030` nằm ngoài top-10 ở
**cả exact lẫn Qdrant**, tức là giới hạn của embedding chứ không phải của Chroma. Lý do lệch:
Chroma Cloud dùng index xấp xỉ (SPANN), còn với 1151 vector HNSW của Qdrant cho kết quả trùng exact.

### Tốc độ (đo từ máy dev ở Việt Nam, gồm mạng)

| Chỉ số | Qdrant Cloud | Chroma Cloud |
|---|---:|---:|
| Query filtered p50 / p95 / p99 | **209 / 324 / 402 ms** | 309 / 1336 / 2380 ms |
| Query unfiltered p50 / p95 / p99 | **209 / 355 / 386 ms** | 308 / 1303 / 2781 ms |
| Số query có lần > 1 s (trên 199, filtered) | 0 | 36 |
| RTT mạng thuần p50 / p95 (healthcheck, 40 lần) | 193 / 197 ms | 260 / 734 ms |
| Ước lượng xử lý phía server (p50 query − p50 RTT) | ≈ 15 ms | ≈ 50 ms |
| Khởi tạo client | 4.1 s | 1.4 s |
| Tải toàn corpus (như lúc API dựng BM25) | 7.1 s | 4.2 s |
| Query đầu tiên sau khởi tạo | 811 ms | 805 ms |

- Qdrant đặt ở GCP `australia-southeast1` (Sydney). Chroma Cloud trả IP `34.200.8.245`, thuộc dải
  AWS, nhiều khả năng us-east (suy từ dải IP, chưa xác minh).
- Đuôi p95/p99 của Chroma là **jitter mạng**: chỉ riêng healthcheck, không tìm kiếm gì, đã có p95
  734 ms; các query chậm rải đều suốt lượt chạy chứ không dồn ở đầu (không phải cold start).
- Filter gần như không tốn thêm thời gian ở cả hai provider (p50 filtered ≈ unfiltered).
- Chroma Cloud chỉ thắng ở khởi tạo và tải corpus. Hai thứ này chỉ chạy một lần lúc app start.

## 4. Vận hành

| | Qdrant Cloud | Chroma Cloud |
|---|---|---|
| Tích hợp với code hiện tại | Chạy production qua llama-index `QdrantVectorStore` | `chromadb` 0.6.3 **không tương thích** server Cloud 1.x (`KeyError '_type'`); phải gọi REST v2 (`src/rag/chroma_cloud.py`) hoặc nâng chromadb (đụng store local + DLL Rust bị Smart App Control chặn) |
| Filter | Cần payload index cho mọi field filter (strict mode → 400 nếu thiếu); `ensure_qdrant_payload_indexes` lo việc này | Không cần cấu hình |
| Xem dữ liệu dạng bảng | Dashboard `<QDRANT_URL>/dashboard` | Dashboard web, lọc theo metadata |

## 5. Giới hạn của phép đo

- Latency đo từ máy dev, **không phải từ Render**. Nếu Render đặt ở Mỹ thì kết quả có thể đảo
  ngược. Muốn có số production: chạy lại `eval/provider_bench.py` từ region của Render.
- Chỉ đo dense retrieval top-10 trên 1151 vector, là quy mô nhỏ nên index xấp xỉ gần như không
  lộ sai số. Khi corpus lớn gấp chục lần, ANN recall cần đo lại.
- Một lượt chạy vào một buổi tối; jitter mạng thay đổi theo giờ.
- 150 câu `g2_*` trong gold set là câu Claude soạn, **chưa người duyệt** (`reviewed: false`).

## 6. Phát hiện khác trong ngày: corpus thiếu văn bản cũ

Chạy thử cùng câu hỏi ở nhiều `as_of_date` cho thấy cơ chế chọn văn bản theo thời điểm hoạt
động đúng (2025-03 → NĐ 74/2024; 2026-03 → NĐ 293/2025; trợ cấp thất nghiệp 2024 → Luật 38/2013,
2026 → Luật 74/2025). Nhưng có hai giai đoạn không có văn bản nào còn hiệu lực trong corpus nên
truy hồi lạc đề:

| Câu hỏi | `as_of_date` | Truy hồi được | Thiếu |
|---|---|---|---|
| Lương tối thiểu vùng I | 2023-01-01 | 45/2019 Điều 91 (nguyên tắc chung) | NĐ 38/2022 (và 90/2019 cho 2020–06/2022) |
| Làm thêm giờ tối đa/tháng | 2019-06-01 | Luật BHXH 58/2014, NĐ 115/2015 (lạc đề) | BLLĐ 10/2012/QH13, NĐ 45/2013, NĐ 05/2015 |

Đề xuất: nạp nhóm lương tối thiểu (90/2019, 38/2022) trước, rồi tới Bộ luật Lao động 2012. Sau
mỗi lần corpus đổi: đồng bộ lại hai cloud và chạy lại benchmark này.

## Tái lập

```powershell
python scripts/copy_chroma_to_cloud.py        # đồng bộ Chroma Cloud (Qdrant: migrate_chroma_to_qdrant.py)
python eval/provider_bench.py --repeat 3 --mlflow provider-bench
```
