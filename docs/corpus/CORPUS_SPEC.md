# DocuMind AI — Corpus Spec (DRAFT, chờ duyệt)

> **Trạng thái:** đề xuất — chưa implement. Lượt này chỉ định nghĩa phạm vi + schema + hành vi.
> **Vertical:** Lao động – Tiền lương – Bảo hiểm xã hội (Việt Nam).
> **Vấn đề đang giải:** corpus không có biên giới — người dùng không biết hỏi gì được,
> không biết gì nằm ngoài phạm vi. Đây KHÔNG phải vấn đề chất lượng retrieval.

Liên quan: [`corpus_manifest.yaml`](corpus_manifest.yaml) (deliverable #1).

---

## 1. Nguyên tắc phạm vi

> **CHỐT 2026-09-09 — PRIVATE-SECTOR ONLY.** Chỉ quan hệ lao động theo hợp đồng lao động
> khu vực tư. Khu vực công (CBCCVC), DNNN, lực lượng vũ trang có logic lương/ngạch riêng —
> **là corpus khác**, không phải phần mở rộng. Mục 4 (#6/#7/#9) là ranh giới cứng, không nới.

**Trong phạm vi:** văn bản QPPL cấp trung ương (Quốc hội, Chính phủ, Bộ) điều chỉnh
quan hệ lao động cá nhân theo HĐLĐ khu vực tư, tiền lương tối thiểu vùng, và các chế độ
bảo hiểm gắn với việc làm: **BHXH bắt buộc/tự nguyện** và **bảo hiểm thất nghiệp**.

**Hoãn sang vòng sau:** bảo hiểm TNLĐ–BNN (nhóm F), Luật Công đoàn, lao động nước ngoài,
xử phạt VPHC. Xem `corpus_manifest.yaml`.

**Ngoài phạm vi (cứng):** xem mục 4.

**Mốc thời gian:** corpus phục vụ truy vấn point-in-time từ **2015-01-01** (Luật Việc làm
2013 có hiệu lực) đến hiện tại. Mọi câu trả lời phải gắn với một `as_of_date` (mặc định =
hôm nay) và chỉ dùng bản điều/khoản có hiệu lực tại mốc đó.

---

## 2. Deliverable #2 — Schema metadata ở cấp điều/khoản

Điểm cốt lõi: **hiệu lực gắn ở cấp điều/khoản/điểm, không phải cấp văn bản.** Một khoản có
thể có **nhiều bản** (version) nối tiếp nhau theo thời gian; retrieval tại `as_of_date` chỉ
được thấy đúng bản có hiệu lực tại mốc đó.

### 2.1. Schema tối thiểu (Ted yêu cầu)

```
{ doc_id, dieu, khoan, effective_from, effective_to, superseded_by }
```

### 2.2. Schema đầy đủ đề xuất (mỗi bản của một khoản = 1 record)

| Trường | Kiểu | Bắt buộc | Ý nghĩa |
|---|---|:---:|---|
| `clause_uid` | str | ✓ | Khóa ổn định: `<doc_id>__d<dieu>[_k<khoan>][_p<diem>]`. VD `45-2019-QH14__d139_k1`. |
| `version_id` | str | ✓ | `<clause_uid>__v<effective_from>`. Phân biệt các bản của cùng khoản. |
| `doc_id` | str | ✓ | Trỏ về `corpus_manifest.yaml`. |
| `doc_so_hieu` | str | ✓ | Hiển thị trích dẫn ("Điều 139 Bộ luật Lao động 2019"). |
| `chuong` / `muc` | str \| null | – | Ngữ cảnh điều hướng. |
| `dieu` | int | ✓ | Số điều. |
| `dieu_tieu_de` | str | ✓ | Tên điều (dùng cho câu hỏi auto-gen + hiển thị). |
| `khoan` | int \| null | ✓ | `null` = phần mở đầu điều. |
| `diem` | str \| null | – | "a", "b"… nếu tách tới điểm. |
| `text` | str | ✓ | Nội dung bản này của khoản. |
| `effective_from` | date | ✓ | Ngày bản này bắt đầu áp dụng. |
| `effective_to` | date \| null | ✓ | `null` = còn hiệu lực; có giá trị = ngày bản này bị thay/bãi. |
| `status` | enum | ✓ | `in_force` \| `superseded` \| `not_yet_in_force` \| `repealed`. |
| `superseded_by` | str \| null | ✓ | `version_id` của bản thay thế (nếu `effective_to != null`). |
| `amended_by_doc` | str \| null | ✓ | `so_hieu` văn bản làm phát sinh bản này (VD `Luật Dân số 2025`). |
| `supersedes_version` | str \| null | – | `version_id` bản trước mà bản này thay. |
| `source_url` | str | ✓ | URL toàn văn (ưu tiên chinhphu.vn / congbao). |
| `consolidated_from` | str \| null | – | Số hiệu VBHN nếu `text` lấy từ văn bản hợp nhất. |
| `verify_status` | enum | ✓ | `VERIFIED` \| `PARTIAL` \| `UNVERIFIED`. `UNVERIFIED` không được index. |

### 2.3. Quy tắc truy vấn point-in-time

Một `version` được coi là "áp dụng tại `T`" khi:

```
effective_from <= T
AND (effective_to IS NULL OR effective_to > T)
AND status != 'repealed' tại thời điểm T
```

- Retriever chỉ nhận tập version thỏa điều kiện trên (bộ lọc dựng ở tầng trên retrieval
  node — **không sửa retrieval node lượt này**).
- Nếu `T` < `meta.earliest_point_in_time` → trả lời "ngoài khoảng thời gian corpus phủ".
- Nếu câu hỏi không nêu mốc → dùng hôm nay, và nêu rõ trong câu trả lời "theo quy định
  hiện hành tại <ngày>".

### 2.4. Ví dụ minh họa (điểm bán của tính năng)

`Điều 139 Bộ luật Lao động 2019 — nghỉ thai sản`:

| version_id | effective_from | effective_to | status | amended_by_doc |
|---|---|---|---|---|
| `…__d139_k1__v2021-01-01` | 2021-01-01 | 2026-07-01 | superseded | – |
| `…__d139_k1__v2026-07-01` | 2026-07-01 | null | in_force | Luật Dân số 2025 |

→ Cùng câu hỏi "chế độ nghỉ thai sản", `as_of_date = 2026-05-01` và `2026-09-09` cho ra
hai câu trả lời khác nhau, mỗi câu trích đúng bản luật của thời điểm.

> **CHỐT 2026-09-09 — N1 đã bound: KHÔNG đủ để bán "point-in-time reconstruction".**
> Bound N1 (số khoản sửa in-place trong văn bản còn hiệu lực) ≈ **3–12**, gần chắc < 30.
> Chỉ **Điều 139 (thai sản)** là case in-place demo được — và metric đúng ("trong ~50 câu
> tần suất cao, bao nhiêu câu đổi đáp án qua cơ chế in-place") hiện = **~1 câu**.
> Các thay đổi lớn khác (lương tối thiểu 3 nghị định, BHXH 2014→2024, BHTN 2013→2026) là
> **doc-level metadata filter** — P-153 đã có.
>
> → Tính năng thời gian **vẫn làm** nhưng gọi đúng tên: *"temporal-aware retrieval + hiệu
> lực cấp điều/khoản"*. **Điểm bán chính = corpus có biên giới** (mục 5). Điều 139 = demo phụ.
> Chi tiết + quyết định: `corpus_manifest.yaml` mục "BOUND N1".

*(Quan hệ sửa đổi Điều 139 lấy từ nguồn thứ cấp — phải đối chiếu lược đồ vbpl.vn +
lấy toàn văn bản mới trước khi dựng record.)*

---

## 3. Deliverable #1 — Corpus manifest

File: [`corpus_manifest.yaml`](corpus_manifest.yaml).

**Tóm tắt nhóm A–G (22 văn bản, chưa tính optional):**

| Nhóm | Văn bản | Ghi chú |
|---|---|---|
| A. Lõi lao động | BLLĐ 2019 (45/2019/QH14); NĐ 145/2020; NĐ 135/2020 (tuổi hưu); TT 10/2020 | BLLĐ 2019 đang bị sửa bởi ≥3 luật 2025–2026 |
| B. Lương tối thiểu | NĐ 293/2025 (hiện hành); NĐ 74/2024 (hết HL 2026-01-01); NĐ 38/2022 (optional) | chuỗi version theo thời gian |
| C. BHXH hiện hành | Luật BHXH 2024 (41/2024/QH15); NĐ 158/2025 (bắt buộc); NĐ 159/2025 (tự nguyện) | hiệu lực từ 2025-07-01 |
| D. BHXH lịch sử | Luật BHXH 2014 (58/2014/QH13); NĐ 115/2015; TT 59/2015; NĐ 134/2015 | cho point-in-time trước 2025-07-01 |
| E. Thất nghiệp | Luật Việc làm 2025 (74/2025/QH15); NĐ 374/2025; Luật Việc làm 2013 (38/2013/QH13); NĐ 28/2015 | khung mới hiệu lực 2026-01-01 |
| F. TNLĐ–BNN | Luật ATVSLĐ 2015 (84/2015/QH13); NĐ 88/2020; NĐ 58/2020 (mức đóng) | chỉ nạp phần chế độ bảo hiểm |
| G. Công đoàn | Luật Công đoàn 2024 (50/2024/QH15) | có thể để nhóm phụ |

**Mức độ xác minh:** VERIFIED 12 · PARTIAL 7 · UNVERIFIED 3. Chi tiết + nguồn từng dòng
trong file. **Không dòng nào được ingest khi còn UNVERIFIED** hoặc khi quan hệ sửa đổi
chưa đối chiếu lược đồ vbpl.vn.

**Hạn chế nghiên cứu lượt này (khai báo thẳng):**
- vbpl.vn bản mới là SPA → không fetch được "Lược đồ / Văn bản liên quan". Mọi quan hệ
  `sua_doi_boi` / `thay_the` dưới đây từ nguồn thứ cấp (thuvienphapluat, chinhphu.vn tin
  tức, luatvietnam).
- Vài `ngay_ban_hanh` (NĐ 293/2025, NĐ 158/2025, NĐ 159/2025) từ nguồn thứ cấp — cần
  đối chiếu Công báo.
- Chưa tra: Thông tư của Bộ Nội vụ/BLĐTBXH hướng dẫn NĐ 158/159/374; NĐ 134/2015 và
  NĐ 28/2015 (số/ngày).

---

## 4. Deliverable #3 — 10 loại câu hỏi corpus KHÔNG trả lời được

Mỗi loại kèm lý do và (nếu có) hướng chuyển.

| # | Loại câu hỏi | Vì sao ngoài phạm vi |
|---|---|---|
| 1 | Quy định riêng của tỉnh/thành (HĐND/UBND cấp tỉnh): danh mục địa bàn áp dụng vùng lương sau sáp nhập tỉnh, chính sách hỗ trợ lao động đặc thù địa phương | Corpus chỉ có văn bản trung ương. |
| 2 | Nội quy lao động / thang bảng lương / thỏa ước lao động tập thể của **một doanh nghiệp cụ thể** ("công ty X quy định…") | Không phải QPPL; corpus không chứa. |
| 3 | Tư vấn cá biệt theo hồ sơ: "trường hợp của tôi có được hưởng…", "công ty tôi làm vậy đúng không" | Cần đánh giá hợp đồng, chứng cứ, tình tiết cụ thể — vượt chức năng tra cứu QPPL. |
| 4 | Tính ra **số tiền cụ thể** (lương hưu, trợ cấp thôi việc/mất việc, trợ cấp thất nghiệp, BHXH một lần) | Cần dữ liệu cá nhân: quá trình đóng, mức lương đóng từng giai đoạn, hệ số trượt giá. Corpus có công thức, không có dữ liệu người dùng. |
| 5 | Thuế thu nhập cá nhân trên tiền lương, giảm trừ gia cảnh, quyết toán thuế | Thuộc Luật Thuế TNCN — không trong corpus. |
| 6 | Tiền lương, phụ cấp, nâng bậc của **cán bộ, công chức, viên chức** khu vực công (lương cơ sở, bảng lương Nhà nước) | Hệ thống văn bản riêng cho khu vực công. Corpus chỉ phủ quan hệ hợp đồng lao động. |
| 7 | Cơ chế tiền lương, tiền thưởng trong **doanh nghiệp nhà nước** 100% vốn Nhà nước | Nghị định chuyên ngành riêng về quản lý lao động – tiền lương DNNN. |
| 8 | Bảo hiểm y tế: mức đóng, mức hưởng, thông tuyến khám chữa bệnh | Thuộc Luật BHYT — ngoài scope trừ khi mở rộng. |
| 9 | Chế độ hưu trí, tử tuất, TNLĐ với **quân nhân, công an, cơ yếu** | Văn bản riêng cho lực lượng vũ trang. |
| 10 | Thủ tục **khởi kiện tranh chấp lao động tại tòa án**, trình tự tố tụng, án phí | Thuộc Bộ luật Tố tụng dân sự. Corpus chỉ có quy định nội dung (quyền/nghĩa vụ), không có quy trình tố tụng. |

**Hai loại bổ sung (biên):**
- 11 | Dự thảo / chính sách đề xuất **chưa ban hành** ("sắp tới có thay đổi gì") — corpus chỉ chứa văn bản đã ban hành.
- 12 | Quy định đã hết hiệu lực **trước mốc sớm nhất** của corpus (BLLĐ 2012, BHXH trước 2016) nếu không nạp bản lịch sử tương ứng.

---

## 5. Deliverable #4 — Interface cho Scope Classifier

### 5.1. Vị trí trong graph

Node mới `scope_gate`, chạy **trước** router intent hiện có và **trước** retrieval.
Không sửa retrieval node. Không thêm dependency (dùng lại LLM client + embedder + config
đã có).

```
User input ──► condense (đã có) ──► [scope_gate] ──► router intent (đã có) ──► retrieve/compliance/…
                                        │
                                        └─(OUT_OF_SCOPE)─► refusal template ──► response  (bỏ qua retrieval)
```

### 5.2. Input

```python
ScopeGateInput:
    query: str                     # câu hỏi (đã condense nếu multi-turn)
    as_of_date: date = today        # mốc point-in-time người dùng chọn
    conversation_summary: str | None = None
```

### 5.3. Output

```python
ScopeGateResult:
    decision: Literal[
        "IN_SCOPE",
        "OUT_OF_SCOPE",
        "NEEDS_PERSONAL_DATA",     # câu hỏi tính tiền cụ thể — trả công thức, không trả số
        "INDIVIDUAL_ADVICE",       # xin tư vấn cá biệt theo hồ sơ
    ]
    scope_topic: str | None         # nhãn chủ đề khớp: "luong_toi_thieu" | "thai_san" |
                                    #   "tro_cap_thoi_viec" | "bhtn" | "tuoi_huu" | "lam_them_gio" | …
    out_of_scope_kind: str | None   # 1 trong 12 loại ở mục 4: "tinh_dia_phuong" |
                                    #   "doanh_nghiep_cu_the" | "thue_tncn" | "khu_vuc_cong" | …
    confidence: float               # 0..1
    reason: str                     # 1 câu, dùng cho log + hiển thị
    suggested_questions: list[str]   # 1–3 câu GẦN NHẤT corpus trả lời được (từ thư viện câu hỏi)
    time_out_of_range: bool = False  # as_of_date < earliest_point_in_time
```

### 5.4. Hành vi theo `decision`

| decision | Hành vi |
|---|---|
| `IN_SCOPE` | Đi tiếp vào router/retrieval bình thường. Truyền `as_of_date` xuống bộ lọc version. |
| `OUT_OF_SCOPE` | **Không** gọi retrieval/generator. Trả template từ chối gồm: (a) nêu phạm vi corpus — nhóm chủ đề + "cập nhật đến &lt;ngày&gt;"; (b) nêu loại việc corpus không xử lý (`out_of_scope_kind`); (c) 1–3 `suggested_questions`. Không trả lời một phần, không đoán. |
| `NEEDS_PERSONAL_DATA` | Cho đi tiếp, nhưng gắn cờ để generator: trả **quy tắc/công thức + căn cứ điều luật**, liệt kê **dữ liệu cá nhân còn thiếu**, tuyệt đối không bịa con số kết quả. |
| `INDIVIDUAL_ADVICE` | Từ chối tư vấn cá biệt + disclaimer ("không thay thế tư vấn pháp lý") + trỏ về quy định chung liên quan + gợi ý cách hỏi lại theo hướng tra cứu. |
| bất kỳ + `time_out_of_range=True` | Trả lời rõ: corpus chỉ phủ từ &lt;earliest&gt;; không suy đoán quy định trước mốc đó. |

### 5.5. Cơ chế 2 tầng (không thêm dependency)

1. **Prefilter rẻ (rule):** allowlist chủ đề + từ khóa (danh sách `topic → keywords`
   lưu ở `data/scope/topics.yaml`, tĩnh, sinh từ corpus). Bắt nhanh case rõ ràng
   (in-scope chắc chắn / out-of-scope chắc chắn) → bỏ qua tầng 2.
2. **LLM phân loại:** dùng **chính LLM client của router intent** (đã cấu hình,
   temp=0). Prompt phân loại trả JSON đúng `ScopeGateResult`; few-shot lấy từ 12 loại
   out-of-scope (mục 4) + mẫu in-scope từ thư viện câu hỏi.
3. **`suggested_questions`:** chọn từ **thư viện câu hỏi tĩnh** (sinh sẵn từ corpus,
   gom theo tình huống) bằng so khớp embedding có sẵn (embedder đã có) — không hạ tầng mới.

### 5.6. Quyết định mở (cần Ted chốt)

- **Fail-open hay fail-closed** khi tầng 2 (LLM) lỗi/timeout?
  Đề xuất: **fail-open** (coi IN_SCOPE) + cờ `low_confidence` để generator thận trọng và
  thêm disclaimer. Lý do: từ chối oan tệ hơn cho demo. Ted có thể muốn fail-closed.
- **Công đoàn (nhóm G)** có nằm trong scope mặc định không, hay là nhóm phụ tắt sẵn?
- **Ngưỡng `confidence`** để phân tầng prefilter ↔ LLM (đề xuất 0.85).

---

## 6. Ràng buộc đã tuân thủ lượt này

- [x] Không viết code — chỉ spec + manifest.
- [x] Không thêm dependency (classifier tái dùng LLM client + embedder + config sẵn có).
- [x] Không sửa retrieval node (bộ lọc version + scope_gate đều nằm ở tầng trên).
- [x] Mọi con số/ngày kèm nguồn; chỗ không xác minh được đánh dấu PARTIAL/UNVERIFIED, không đoán.

## 7. Việc lượt sau — thứ tự đã chốt (2026-09-09)

**Danh sách chốt: 18 văn bản + VBHN BLLĐ.** Chi tiết `corpus_manifest.yaml` → `locked_list_v1`.

**N1 đã bound** (đường tắt: đọc điều "sửa đổi, bổ sung" của từng luật sửa — text sạch
chinhphu.vn, không cần VBHN ở bước quyết định): **≈ 3–12, < 30.** Metric đúng ("~50 câu
tần suất cao — bao nhiêu đổi đáp án qua cơ chế in-place") ≈ **1** (nghỉ thai sản Điều 139).

→ **Đề xuất phương án (a)** — chờ Ted xác nhận Y/N:
- Không bán "point-in-time reconstruction" làm điểm khác biệt.
- Tính năng thời gian **vẫn làm**, gọi đúng tên "temporal-aware retrieval + hiệu lực cấp điều/khoản"
  (xử lý các mốc doc-level: lương tối thiểu theo ngày, BHXH 2014↔2024, BHTN 2013↔2026).
- **Điểm bán chính = corpus có biên giới** (mục 5): manifest hiển thị + scope classifier + thư viện câu hỏi.
- (b) "bỏ trục thời gian" — tệ hơn, không chọn.

Thứ tự:
1. Ted xác nhận (a).
2. Tra 1 vòng 6 dòng chưa xác minh (11+12/2025-TT-BNV, 134/2015, 28/2015 + nhóm F).
   Không ra ngày + lược đồ → **LOẠI THẲNG** (không giữ UNVERIFIED).
3. Đối chiếu Công báo: mọi ngày ban hành/hiệu lực PARTIAL (18 văn bản).
4. Đối chiếu lược đồ vbpl.vn: mọi quan hệ sua_doi_boi / thay_the.
5. Acquire text sạch 18 văn bản + VBHN BLLĐ (18/VBHN-VPQH — bản 2026, mới hơn 125/VBHN-VPQH bản 2025).
6. Thư viện câu hỏi (40–60 câu, theo tình huống) + gold answers ([[eval-methodology]]).
7. Chốt 3 quyết định mở ở mục 5.6 (fail-open/closed; Công đoàn đã hoãn; ngưỡng confidence).
8. Implement: schema ingestion → scope_gate → temporal filter.

**Đã hoãn download** cho tới khi bước 2–4 xong.
