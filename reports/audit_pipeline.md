# DocuMind AI — Audit pipeline (2026-06-20)

Soi toàn bộ: ingest → chunk → embed → store → retrieve → rerank → prompt → eval.
Kết luận trước, chi tiết sau.

## TL;DR

**Pipeline về cơ bản khỏe. Điểm "lỗi" bạn thấy (faithfulness/relevancy/citation = 0) KHÔNG phải model kém — là bug phương pháp đo trong eval.** Hai việc đáng sửa thật: (1) ngưỡng abstain áp sai thang điểm cho strategy `hybrid`, (2) reranker cross-encoder là model tiếng Anh dùng cho tiếng Việt.

Bằng chứng pipeline tốt: baseline RAGAS cũ (ghi trong `embedder.py`) đạt faithfulness 0.87 / relevancy 0.82 / context_recall 0.77; smoke hiện tại context_recall 1.0, hit_rate 0.8.

## Những thứ KHÔNG hở (đã kiểm tra, ổn)

- **OCR/scan extract:** chunk từ QĐ-853/828/1228 (bản scan) ra tiếng Việt sạch, đủ dấu, giữ cấu trúc markdown. Không bị garble.
- **Chunking (`chunk_by_dieu`):** 91 chunk, char len median 728 / mean 1156 — kích thước hợp lý. 84/91 cắt theo Điều, 7 fallback_window.
- **Embedding:** multilingual-MiniLM-L12 (384d), normalize + cosine — phù hợp tiếng Việt, đúng dim với index.
- **Corpus production `documind_legal`:** đúng 91 chunk UNETI, phân bổ khớp spec (740:15, 747:19, 748:11, 670:8, 828:18, 1228:7, 853:13). **Không bị nhiễm.**
- **Retrieval:** RRF + BM25 + dense kéo đúng chunk (recall 1.0). Điểm RRF nhỏ (~0.016) là bản chất công thức `1/(k+rank)`, KHÔNG phải dấu hiệu chunk kém.

## Điểm hở thật — ưu tiên theo impact

| # | Vấn đề | Mức | Ảnh hưởng |
|---|--------|-----|-----------|
| P0 | Eval `hybrid` chạy `rerank=False` → chunk mang điểm RRF ~0.016 < ngưỡng `_MIN_RELEVANCE_SCORE=0.05` → generator abstain → mọi câu trả lời thành "không tìm thấy" → faithfulness/relevancy/citation/correctness sập về 0 | Cao | Làm hỏng kết quả eval, không phản ánh chất lượng thật. So sánh ablation bất công với hybrid |
| P1 | Reranker = `cross-encoder/ms-marco-MiniLM-L-6-v2` (train trên MS-MARCO **tiếng Anh**) dùng cho cặp query–doc **tiếng Việt** → logit không đáng tin; cộng với gate 0.05 có thể loại nhầm chunk đúng (từ chối oan) hoặc giữ chunk sai | Cao | Rủi ro chất lượng **production thật** (đường rerank là đường mặc định của app) |
| P2 | Cache cũ `reports/answers_cache/*.json` chứa câu từ chối; re-run không xóa sẽ tái dùng rác | TB | Lặp lại kết quả 0 dù đã sửa |
| P2 | `correctness_token_f1` so verbose answer với ground_truth ngắn → luôn thấp (0.05) | Thấp | Đừng dùng metric này làm thước đo chính; tin semantic + RAGAS |
| P3 | Collection orphan `documind_test` (2042 chunk luật quốc gia) còn nằm trong cùng file sqlite | Thấp | Không ảnh hưởng production, chỉ phình DB + dễ gây nhầm khi debug |
| P3 | Doc drift: `retriever.py` ghi "bge-m3"/"356-chunk", thực tế MiniLM/91 chunk; comment `generator.py:55` ghi sai "RRF scores are 0-1" | Thấp | Gây hiểu sai khi maintain — chính comment sai này dẫn tới bug P0 |
| P3 | 9/91 chunk > 3000 ký tự bị `_build_context` cắt (cap 3000/chunk) | Thấp | Mất nội dung cuối chunk dài |

## Đề xuất fix (theo thứ tự)

1. **P0 — tách ngưỡng abstain khỏi thang điểm.** Cho `generate_answer(query, chunks, min_score=...)`; eval truyền ngưỡng theo strategy (hybrid RRF ~0.01, rerank giữ 0.05). Hoặc normalize điểm chunk về 0–1 trước khi lọc. Production không đổi → an toàn.
2. **P1 — đổi reranker sang multilingual:** `BAAI/bge-reranker-v2-m3` hoặc `jina-reranker-v2-base-multilingual`, rồi tune lại `_MIN_RELEVANCE_SCORE` theo thang điểm mới. Đo bằng strategy `rerank` trên full set để xác nhận lợi ích.
3. **P2 — xóa cache trước mỗi lần benchmark lại:** `Remove-Item reports\answers_cache\*.json`.
4. **P3 — dọn DB:** drop collection `documind_test` để thu nhỏ sqlite.
5. **P3 — sync lại docstring/comment** cho khớp model + corpus thật.
6. **(Tùy chọn) Nâng cấp embedding BGE-M3** — `embedder.py` ước tính +5–8pp context_recall cho tiếng Việt; cần re-index 91 chunk.

## Cách chạy eval cho ra số đáng tin

```powershell
Remove-Item "reports\answers_cache\*.json"   # bỏ cache câu từ chối
# sau khi sửa P0, chạy full 4 strategy:
python eval/run_evals.py --strategies bm25 dense hybrid rerank --output reports/full_ragas_110q.json
```
Checkpoint tự lưu `reports/ragas_checkpoints/` — đứt giữa chừng chạy lại lệnh y hệt sẽ skip câu đã chấm.
