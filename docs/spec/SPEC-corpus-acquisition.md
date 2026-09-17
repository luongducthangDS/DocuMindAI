# Spec: `corpus-acquisition`

> Module 1/9. Phụ thuộc: — (gốc). Xem `SPEC.md` (capability map).
> Kế thừa: `docs/corpus/CORPUS_SPEC.md` + `docs/corpus/corpus_manifest.yaml` (DRAFT session 4).

## Objective

Biến DRAFT corpus manifest thành corpus **đã xác minh, có toàn văn sạch, sẵn sàng ingest**. Không viết code pipeline ở module này — output là dữ liệu + manifest đã đóng verification nợ.

Kết thúc module này: `data/raw/lao_dong/` chứa toàn văn 18 văn bản `locked_list_v1` + VBHN BLLĐ; `corpus_manifest.yaml` không còn dòng `UNVERIFIED` nào trong danh sách chốt; mọi quan hệ `sua_doi_boi`/`thay_the` đã đối chiếu lược đồ chính thức.

## Boundaries (delta so với SPEC.md)

- **Ask first** trước khi bulk-download (Ted có thể tự tải).
- **Never** giữ dòng `UNVERIFIED` trong `locked_list_v1`: quy tắc Ted — tra 1 vòng, không ra ngày + lược đồ → **LOẠI THẲNG**.
- **Never** mở rộng quá 18 VB + VBHN khi chưa đóng verification nợ (nhóm F, Công đoàn, optional đã hoãn — giữ định nghĩa trong manifest, không ingest).
- Không đổi format `corpus_manifest.yaml` mà không cập nhật `SPEC-clause-schema-ingestion.md` (module sau đọc nó).

## Công việc (thứ tự đã chốt session 4)

1. **Xoá tài liệu ngân hàng bịa** khỏi corpus cũ: `05_quy_che_cho_vay_tieu_dung_tin_chap.md`, `06_bieu_phi_dich_vu_tai_khoan_va_the.md`, file 01–04 (tóm lược viết tay + URL `sbv.gov.vn` bịa). Kiểm tra `data/raw/` thực tế trước khi xoá (memory 2 ngày tuổi).
2. **Tra 1 vòng 6 dòng chưa xác minh:** `11/2025/TT-BNV`, `12/2025/TT-BNV`, `25/2025/TT-BYT`, `134/2015/NĐ-CP`, `28/2015/NĐ-CP`, `18+19/2021/TT-BLĐTBXH`, `24/2022/TT-BLĐTBXH`. Không ra số hiệu + ngày ban hành + lược đồ → loại khỏi danh sách.
3. **Đối chiếu Công báo** (`congbao.chinhphu.vn`): mọi `ngay_ban_hanh`/`ngay_hieu_luc` đang `PARTIAL` — đặc biệt NĐ 293/2025, NĐ 158/2025, NĐ 159/2025, NĐ 374/2025.
4. **Đối chiếu lược đồ vbpl.vn:** mọi quan hệ `sua_doi_boi`/`thay_the`/`bi_thay_the_boi`. Trọng tâm: 4 luật nghi sửa BLLĐ in-place (41/2024, 113/2025, 71/2025, 124/2025) — xác nhận điều/khoản nào thực sự bị sửa. Đây là dữ liệu cho `clause-schema-ingestion` dựng version.
5. **Acquire toàn văn sạch** 18 VB + VBHN BLLĐ (`18/VBHN-VPQH` bản 2026) vào `data/raw/lao_dong/<doc_id>.md`. Ưu tiên text/HTML sạch (chinhphu.vn / congbao / vbpl mới) hơn PDF. Mỗi file có header YAML frontmatter khớp `doc_id` trong manifest.
6. **Xác nhận Điều 139 BLLĐ:** lấy cả bản khoản 1 hiện hành (2021) và bản mới (Luật Dân số 2025, HL 2026-07-01) — case demo in-place duy nhất.

## Success Criteria

- [ ] `data/raw/lao_dong/` có ≥ (18 − số dòng bị loại ở bước 2) file `.md` toàn văn, mỗi file có frontmatter `doc_id`, `so_hieu`, `ngay_ban_hanh`, `ngay_hieu_luc`, `nguon`.
- [ ] `corpus_manifest.yaml`: mọi dòng trong `locked_list_v1` có `verify_status: VERIFIED`; dòng nào không lên VERIFIED được đã bị xoá khỏi `locked_list_v1` và ghi lý do ở comment.
- [ ] Bảng `sua_doi_boi` cho BLLĐ 2019: mỗi quan hệ có `pham_vi` cụ thể tới điều/khoản + `verify: VERIFIED` (đối chiếu vbpl.vn hoặc toàn văn luật sửa).
- [ ] Điều 139 BLLĐ: 2 bản text (2021, 2026-07-01) lưu sẵn để module sau dựng version.
- [ ] Tài liệu ngân hàng bịa đã xoá; `git status` không còn file corpus cũ.
- [ ] Không file nào trong `data/raw/lao_dong/` là tóm lược viết tay — phải là toàn văn chính thức.

## Verify

- Đọc `corpus_manifest.yaml`, `grep -c UNVERIFIED` trên phần `locked_list_v1` = 0.
- Mở 3 file `data/raw/lao_dong/` ngẫu nhiên, đối chiếu điều đầu + điều cuối với nguồn `chinhphu.vn`.
- Script kiểm tra frontmatter (viết ở `clause-schema-ingestion`, hoặc test tạm ở đây).

## Open Questions — D2 CHỐT (2026-09-11, Ted)

1. **Nguồn toàn văn** khi vbpl.vn là SPA: **de-facto đã chốt** qua thực tế dùng trong `corpus_manifest.yaml` (`meta.nguon_uu_tien`) — ưu tiên `vanban.chinhphu.vn` / `congbao.chinhphu.vn` / `vbpl.vn` cho số hiệu + mốc ngày chính thức; `thuvienphapluat.vn` / `luatvietnam.vn` dùng làm nguồn cross-check thứ cấp (đã áp dụng khi bound N1 ở session 4). VB lịch sử (BHXH 2014, Việc làm 2013) + VBHN cũng theo thứ tự ưu tiên này, thử `vbpl.vn` hoặc trang bộ chủ quản trước khi rơi xuống nguồn thứ cấp.
2. **VBHN các NĐ** (145/2020, TT 59/2015, NĐ 28/2015): **CHẤP NHẬN dùng bản gốc nếu không tra được số hiệu VBHN** — nhóm này là phụ trợ (không phải BLLĐ chính), không chặn tiến độ. Phải ghi rõ hạn chế "không phản ánh sửa đổi giữa chừng" trong `corpus_manifest.yaml` (đã ghi, xem `van_ban_hop_nhat`) và trong README (mục Hạn chế, làm ở T30/T31).
3. **NĐ 38/2022** (lương tối thiểu 2022): **CẮT ở 2024** — bỏ khỏi corpus vòng này (đã xoá khỏi `corpus_manifest.yaml`). point-in-time lương tối thiểu bắt đầu từ `74/2024/NĐ-CP` (2024-07-01), không lùi tới 2022.
