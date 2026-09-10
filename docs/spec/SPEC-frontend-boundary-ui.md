# Spec: `frontend-boundary-ui`

> Module 8/9. Phụ thuộc: `temporal-retrieval`, `scope-gate`, `question-library`. Xem `SPEC.md`.
> Frontend hiện có: `frontend/src/main.tsx` (827 dòng, React 19 + Vite), `styles.css` (694 dòng). Dark-mode, WebSocket streaming, citation drawer.

## Objective

UI cho định vị (a): người dùng **thấy cái hộp trước khi hỏi**. 4 phần:
1. **Panel corpus manifest** ở trang chủ: N văn bản VERIFIED + ngày cập nhật + mục "ngoài phạm vi" viết thẳng.
2. **Thư viện câu hỏi theo tình huống**: chip/accordion nhóm tình huống, click → điền câu hỏi.
3. **Bộ chọn `as_of_date`** + hiển thị hiệu lực trong câu trả lời ("theo quy định tại &lt;ngày&gt;", badge khi điều đã bị sửa).
4. **Render template từ chối** của `scope_gate`: phạm vi corpus + loại OOC + gợi ý câu (click được).

## Contract API (backend cung cấp — xác nhận khi làm module tương ứng)

- `GET /corpus/manifest` (mới, hoặc field trong `/health`): `{ documents: [{so_hieu, ten, ngay_hieu_luc}], updated_at, out_of_scope: [...], earliest_point_in_time }`.
- `GET /questions/library` (mới): trả `data/questions/library.json`.
- `POST /query` body thêm `as_of_date?: string`.
- `POST /query` response khi OOC: field phân biệt (`decision: "OUT_OF_SCOPE"`, `suggested_questions: [...]`, `scope_reason`, `corpus_scope`).

## Boundaries (delta)

- **Ask first:** thêm endpoint API mới (`/corpus/manifest`, `/questions/library`) — thuộc contract, xác nhận với backend module.
- Mở rộng `main.tsx` hiện có — giữ dark-mode + streaming + citation drawer. Không viết lại UI.
- `npm run build` phải sạch; giữ bundle trong `frontend/dist/` nếu repo đang commit dist (kiểm tra).
- User-facing text tiếng Việt.
- Không thêm dependency npm nặng (charting, UI kit) — CSS thuần như hiện tại.

## Công việc

1. Component `CorpusManifestPanel` — fetch `/corpus/manifest`, render danh sách + "ngoài phạm vi" (collapsible), hiển thị `updated_at`.
2. Component `QuestionLibrary` — fetch `/questions/library`, nhóm tình huống → chips; click điền vào ô hỏi.
3. `AsOfDatePicker` — mặc định hôm nay; gắn vào query request; hiển thị nổi bật khi ≠ hôm nay.
4. Trong khối câu trả lời: badge "Điều này đã được sửa đổi — đang hiển thị bản có hiệu lực tại &lt;ngày&gt;" khi metadata chunk có `superseded_by`/`amended_by_doc`.
5. `RefusalCard` — khi response `decision != IN_SCOPE`: hiển thị `scope_reason` + `corpus_scope` + `suggested_questions` (button → gửi luôn).
6. `styles.css` cho các component mới, theo palette hiện có.

## Success Criteria

- [ ] Trang chủ (chưa hỏi gì) hiển thị: số văn bản + ngày cập nhật + danh sách ngoài phạm vi + ≥1 nhóm câu hỏi tình huống.
- [ ] Chọn `as_of_date = 2025-01-01`, hỏi "lương tối thiểu vùng I" → câu trả lời ghi 4.96tr + nêu mốc; đổi sang 2026-02-01 → 5.31tr.
- [ ] Hỏi câu OOC ("thuế TNCN của tôi") → `RefusalCard` hiện phạm vi corpus + 1–3 câu gợi ý click được, KHÔNG hiện câu trả lời bịa.
- [ ] Câu hỏi về Điều 139 với `as_of_date` sau 2026-07-01 → badge "đã được sửa đổi".
- [ ] `cd frontend && npm run build` sạch; `npm run dev` chạy, không lỗi console.
- [ ] Responsive ~400px không vỡ layout (panel manifest collapse).

## Verify

`npm run build` + kiểm thủ công 5 luồng: trang chủ, in-scope + as_of, OOC refusal, temporal badge, thư viện câu hỏi click. (Theo `~/.claude/CLAUDE.md`: verify trình duyệt gộp 1 lần cuối, screenshot viewport, đóng browser ngay.)

## Open Questions

1. `/corpus/manifest` + `/questions/library` là endpoint riêng hay nhồi vào `/health`/`/query` response? (đề xuất: endpoint riêng, cache-friendly).
2. `as_of_date` picker: mọi câu hỏi đều có, hay ẩn sau "tùy chọn nâng cao" để không rối người dùng phổ thông? (đề xuất: mặc định ẩn, hiện khi câu trả lời có yếu tố thời gian).
3. Repo có đang commit `frontend/dist/` không — build lại có làm diff to không?
