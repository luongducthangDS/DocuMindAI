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

## Open Questions

1. **Nguồn toàn văn** khi vbpl.vn là SPA: Ted tự tải, hay dùng luatvietnam.vn / thuvienphapluat (chặn bot) / chinhphu.vn "toàn văn" (chỉ có VB mới)? Với VB lịch sử (BHXH 2014, Việc làm 2013) + VBHN — nguồn nào?
2. **VBHN các NĐ** (145/2020, TT 59/2015, NĐ 28/2015): số hiệu VBHN chưa tra — cần cho point-in-time nhóm lịch sử. Nếu không lấy được → nhóm `het_hieu_luc_giu_cho_point_in_time` chỉ dùng bản gốc (không phản ánh sửa đổi giữa chừng), chấp nhận được không?
3. **NĐ 38/2022** (lương tối thiểu 2022): giữ để point-in-time lùi tới 2022, hay cắt ở 2024?
