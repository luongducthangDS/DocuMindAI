# Spec: `compliance-criteria`

> Module 6/9. Phụ thuộc: `corpus-acquisition`, `temporal-retrieval`. Xem `SPEC.md`.
> Engine hiện có: `src/rag/compliance.py` + `data/compliance/criteria.json` (đang là banking).

## Objective

Thay bộ tiêu chí ngân hàng bằng tiêu chí **lao động / tiền lương / BHXH**, giữ nguyên cơ chế engine (`match → extract → evaluate → verdict + citation`). Tiêu chí gắn **mốc áp dụng** để phối với `temporal-retrieval`.

## Contract: `data/compliance/criteria.json` (mở rộng schema hiện có)

Schema hiện tại mỗi criterion: `id, topic, keywords[], condition{field, operator, value, unit}, so_hieu, dieu_khoan, source_url, verdict_template{pass,fail}`.

**Thêm:**
- `effective_from`, `effective_to` (date\|null) — criterion chỉ áp cho `as_of_date` trong khoảng.
- `criterion_kind`: `threshold` (như hiện tại) | `formula_only` (chỉ trả công thức + dữ liệu thiếu, khớp `NEEDS_PERSONAL_DATA`).
- Với chuỗi version (lương tối thiểu vùng 2022/2024/2026): nhiều criterion cùng `topic`, khác `effective_from` + `value`.

## Tiêu chí đề xuất (từ memory `documind-labor-pivot`)

| topic | condition | căn cứ | ghi chú |
|---|---|---|---|
| `luong_toi_thieu_vung_I` | `luong_thang >= 5_310_000` (từ 2026-01-01) | NĐ 293/2025 | + bản 4.96tr (NĐ 74/2024, 2024-07-01..2025-12-31) |
| `lam_them_gio_nam` | `gio <= 200` (hoặc `<= 300` nếu ngành đặc biệt) | Điều 107 BLLĐ 2019 | 2 nhánh — cần `NEEDS_PERSONAL_DATA` hỏi ngành |
| `ty_le_dong_bhxh_bat_buoc_nld` | `ty_le == 8` (%) hưu trí-tử tuất, NLĐ | Luật BHXH 2024 + NĐ 158/2025 | doc-level theo mốc 2025-07-01 |
| `tro_cap_thoi_viec` | `formula_only`: 1/2 tháng lương × số năm làm việc (trừ thời gian đóng BHTN) | Điều 46 BLLĐ + NĐ 145/2020 | cần dữ liệu cá nhân → formula_only |
| `tuoi_huu_2026` | `nam >= 61.5` / `nu >= 57.0` (năm 2026) | NĐ 135/2020 | bảng tra theo năm nghỉ — 1 version/năm |
| `nghi_thai_san` | `thang == 6` (nữ, chuẩn); từ 2026-07-01: con thứ 2 → 7 tháng | Điều 139 BLLĐ (bản 2021 vs Luật Dân số 2025) | **case demo in-place** — phối `temporal-retrieval` |

## Boundaries (delta)

- **Never** bịa ngưỡng — mỗi `value` + `so_hieu` + `dieu_khoan` đối chiếu toàn văn `data/raw/lao_dong/` tại thời điểm viết criterion.
- Giữ engine `src/rag/compliance.py` — mở rộng, không viết lại. Cập nhật `_NUMBER_NEAR_LABEL_RE` labels (hiện là "lãi suất|hạn mức|..." của banking) sang nhãn lao động ("lương|giờ làm thêm|tỷ lệ đóng|thâm niên|tuổi|...").
- `criterion_kind == formula_only` → engine trả `verdict = "formula"` (mới) thay vì pass/fail; node render công thức + "dữ liệu còn thiếu".
- **Ask first:** đổi cấu trúc return của `check_compliance()` (graph.py `_render_compliance_answer` đọc nó).

## Công việc

1. Viết `data/compliance/criteria.json` mới (≥6 tiêu chí trên) — **đối chiếu toàn văn trước**, ghi `source_url` tới điều cụ thể.
2. Mở rộng `compliance.py`: field `effective_from/to` → `match_criteria` lọc theo `as_of_date`; `criterion_kind` → nhánh `formula_only` trả `verdict="formula"` + `missing_fields[]`.
3. Cập nhật regex nhãn tiếng Việt lao động.
4. `graph.py` `compliance_check_node` + `_render_compliance_answer`: xử lý `verdict="formula"`.
5. Đồng bộ với `scope-gate`: `NEEDS_PERSONAL_DATA` từ scope_gate → route vào `do_compliance` (thay vì generator thường) khi có criterion `formula_only` khớp.
6. Cập nhật `tests/test_compliance.py` toàn bộ (đang test DTI/lãi suất banking).

## Success Criteria

- [ ] `data/compliance/criteria.json`: ≥6 tiêu chí lao động, 0 tiêu chí banking; mỗi tiêu chí `source_url` trỏ điều cụ thể trong corpus VERIFIED.
- [ ] "Làm thêm 250 giờ/năm có vượt quy định không?" → `fail` (vượt 200h), citation Điều 107 BLLĐ; nếu nêu "ngành dệt may" → hỏi lại hoặc `pass` (300h).
- [ ] "Lương tối thiểu vùng I hiện nay" với `as_of_date=2025-03-01` → 4.96tr (NĐ 74/2024); `as_of_date=2026-02-01` → 5.31tr (NĐ 293/2025).
- [ ] "Trợ cấp thôi việc của tôi là bao nhiêu?" → `verdict=formula`: công thức Điều 46 + liệt kê dữ liệu thiếu (thời gian làm việc, lương bình quân 6 tháng, thời gian đóng BHTN), KHÔNG ra con số.
- [ ] `pytest tests/test_compliance.py` xanh (bộ test mới).
- [ ] Regression: `pytest tests/test_agent.py` xanh.

## Verify

`pytest tests/test_compliance.py tests/test_agent.py -v` + 4 tình huống thủ công (2 threshold, 1 temporal, 1 formula).

## Open Questions

1. Giữ hardcode `criteria.json` (đơn giản, đúng scope) hay để engine tự sinh criterion từ corpus? (Open question #5 `SPEC.md` — đề xuất: giữ hardcode, ~10 tiêu chí là đủ cho demo).
2. `formula_only` verdict có cần LLM render công thức cho tự nhiên, hay template tĩnh đủ? (đề xuất: template tĩnh + LLM chỉ để diễn giải nếu có).
3. Số tiêu chí mục tiêu: 6 (bảng trên) hay mở tới ~12 (thêm: nghỉ phép năm, lương thử việc 85%, BHXH một lần, mức hưởng thất nghiệp 60%)?
