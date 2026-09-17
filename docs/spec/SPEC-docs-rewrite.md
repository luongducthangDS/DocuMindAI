# Spec: `docs-rewrite`

> Module 9/9. Phụ thuộc: `eval-goldset` (cần số thật trước). Xem `SPEC.md`.
> Ràng buộc: memory `documind-benchmark-integrity` (rủi ro CV lớn nhất).

## Objective

Viết lại README.md + EVALUATION.md cho vertical lao động + định vị (a). **0 tham chiếu ngân hàng, 0 con số eval vô căn cứ.** Đây là mặt tiền interviewer/hội đồng đọc đầu tiên.

## Boundaries (delta)

- **Never** đưa số benchmark chưa có trong `reports/benchmark_results.json`.
- **Never** mô tả tính năng thời gian là "point-in-time reconstruction" / "full reconstruction" (mâu thuẫn (a)) — dùng "temporal-aware retrieval + hiệu lực cấp điều/khoản".
- **Never** để sót chuỗi "ngân hàng / banking / SBV / NHNN / DTI / Thông tư 39/2016 / lãi suất" trong README/EVALUATION/architecture diagram.
- Chạy `grep -ri "bank\|ngân hàng\|NHNN\|SBV\|DTI" README.md EVALUATION.md` = rỗng trước khi xong.
- Giữ cấu trúc README hiện có (badges, executive overview, architecture, mermaid) — thay nội dung, không viết lại bố cục.

## Công việc

1. **README.md:**
   - Executive overview: trợ lý tra cứu QPPL lao động–tiền lương–BHXH khu vực tư VN; vấn đề gốc = corpus có biên giới.
   - Key capabilities: (1) scope classifier cưỡng chế + manifest hiển thị + thư viện câu hỏi tình huống [headline]; (2) temporal-aware retrieval + hiệu lực cấp điều/khoản; (3) hybrid retrieval + rerank; (4) compliance engine (tiêu chí lao động); (5) grounding + citation; (6) LLM chain resilient.
   - Corpus section: N văn bản (link manifest), mốc point-in-time 2015→nay, mục "ngoài phạm vi".
   - Benchmark section: **chỉ** số từ `reports/benchmark_results.json` + link `EVALUATION.md` + `reports/failure_analysis.md`. Nêu thẳng kích thước corpus + hạn chế.
   - Cập nhật badge test count theo `pytest` thật; cập nhật mermaid diagram (thêm `scope_gate`, bỏ nhánh banking).
2. **EVALUATION.md:** viết lại — methodology (gold viết trước), 3 nhóm câu (A/B/C), metrics (retrieval + scope + temporal + RAGAS), bảng số thật, failure analysis, hạn chế + hướng mở rộng.
3. **docs/corpus/CORPUS_SPEC.md:** bỏ nhãn DRAFT, cập nhật trạng thái "implemented"; đồng bộ với schema thực tế đã ingest.
4. Xoá/cập nhật file doc cũ nhắc banking (kiểm `docs/`, root `*.md`).
5. Cập nhật memory `documind-benchmark-integrity` + `documind-labor-pivot` (trạng thái "đã implement").

## Success Criteria

- [ ] `grep -ri "ngân hàng\|banking\|NHNN\|SBV\| DTI\|39/2016" README.md EVALUATION.md docs/` → rỗng.
- [ ] Mọi con số trong README/EVALUATION truy được về `reports/benchmark_results.json` hoặc `reports/failure_analysis.md`.
- [ ] README mô tả tính năng thời gian đúng tên (không "reconstruction").
- [ ] Badge test count = số `pytest` thật hiện tại.
- [ ] Mermaid architecture diagram có `scope_gate` node + `as_of_date` flow.
- [ ] EVALUATION.md có section "Hạn chế" (kích thước corpus, coverage, điều chưa test).
- [ ] Interviewer mở `reports/benchmark_results.json` → khớp README (test đối chiếu, hoặc checklist thủ công).

## Verify

`grep` checks trên + đọc toàn bộ README + EVALUATION 1 lượt + đối chiếu từng số với file report.

## Open Questions

1. Có giữ tên "DocuMind AI" không, hay đổi tên phản ánh vertical lao động? (đề xuất: giữ — brand đã có, thêm tagline).
2. README song ngữ hay chỉ tiếng Việt? (memory: CV nhắm thị trường VN → tiếng Việt chính, English summary ngắn ở đầu là đủ).
3. Demo sống (link Render/Vercel) đưa vào README luôn hay chờ module deploy riêng? (Open question #6 `SPEC.md`).
