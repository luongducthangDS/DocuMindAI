# Spec: `scope-gate`

> Module 4/9. Phụ thuộc: `clause-schema-ingestion`. Xem `SPEC.md` + `docs/corpus/CORPUS_SPEC.md` §4–§5.

## Objective

Node `scope_gate` chạy **trước** router intent + **trước** retrieval. Cưỡng chế biên giới corpus: câu ngoài phạm vi → từ chối kèm nêu phạm vi + gợi ý câu gần nhất. Đây là **điểm bán chính** của định vị (a).

## Contract: `ScopeGateResult` (từ `CORPUS_SPEC.md` §5.3)

```python
ScopeGateInput:
    query: str                      # đã condense nếu multi-turn
    as_of_date: date = today
    conversation_summary: str | None = None

ScopeGateResult:
    decision: Literal["IN_SCOPE", "OUT_OF_SCOPE", "NEEDS_PERSONAL_DATA", "INDIVIDUAL_ADVICE"]
    scope_topic: str | None          # "luong_toi_thieu" | "thai_san" | "tro_cap_thoi_viec" | "bhtn" | "tuoi_huu" | "lam_them_gio" | ...
    out_of_scope_kind: str | None    # 1 trong 12 loại §4: "tinh_dia_phuong" | "doanh_nghiep_cu_the" | "thue_tncn" | "khu_vuc_cong" | ...
    confidence: float
    reason: str                      # 1 câu, log + hiển thị
    suggested_questions: list[str]    # 1–3 câu gần nhất corpus trả lời được (từ question-library)
    time_out_of_range: bool = False
```

## Vị trí trong graph

```
START → do_contextualize → do_scope_gate → router → ...
                               │
                               ├─(OUT_OF_SCOPE / classifier lỗi)─► do_refusal → do_persist → END
                               └─(IN_SCOPE / NEEDS_PERSONAL_DATA / INDIVIDUAL_ADVICE)─► router (truyền cờ xuống)
```

## Hành vi theo `decision` (từ §5.4)

| decision | Hành vi |
|---|---|
| `IN_SCOPE` | Đi tiếp router/retrieval. Truyền `as_of_date` xuống. |
| `OUT_OF_SCOPE` | **Không** gọi retrieval/generator. Template từ chối: (a) nêu phạm vi corpus (nhóm chủ đề + "cập nhật đến &lt;ngày&gt;"); (b) nêu `out_of_scope_kind`; (c) 1–3 `suggested_questions`. Không trả lời một phần. |
| `NEEDS_PERSONAL_DATA` | Đi tiếp, gắn cờ generator: trả **công thức + căn cứ điều luật** + liệt kê dữ liệu cá nhân còn thiếu, KHÔNG bịa con số kết quả. |
| `INDIVIDUAL_ADVICE` | Từ chối tư vấn cá biệt + disclaimer + trỏ quy định chung + gợi ý hỏi lại theo hướng tra cứu. |
| `time_out_of_range=True` | Nêu rõ corpus phủ từ &lt;earliest&gt;; không suy đoán. |

## Cơ chế 2 tầng (từ §5.5) — không thêm dependency

1. **Prefilter rẻ (rule):** `data/scope/topics.yaml` — allowlist `topic → keywords` (tĩnh, sinh từ corpus). Bắt case rõ ràng in/out → bỏ qua tầng 2.
2. **LLM classifier:** dùng **chính LLM client router intent** (Groq→Gemini chain, temp=0). Prompt trả JSON `ScopeGateResult`; few-shot lấy từ 12 loại OOC §4 + mẫu in-scope từ question-library.
3. **`suggested_questions`:** so khớp embedding (embedder có sẵn) query ↔ thư viện câu hỏi tĩnh.

## Fail-closed (Ted chốt 2026-09-10)

Khi tầng 2 (LLM) lỗi/timeout và prefilter **không** kết luận được:
- `decision = OUT_OF_SCOPE`, `reason = "classifier_unavailable"`, `confidence = 0.0`.
- Template từ chối **phân biệt rõ**: *"Hệ thống tạm thời chưa phân loại được câu hỏi của bạn, vui lòng thử lại."* — KHÔNG nói "câu hỏi ngoài phạm vi" (không đổ lỗi cho người dùng), vẫn kèm (a) phạm vi corpus + (c) gợi ý.
- Log ở mức `error` để phân biệt với OOC thật khi đọc metric.
- Prefilter bắt được in-scope chắc chắn → vẫn cho đi tiếp (fail-closed chỉ áp khi cả 2 tầng không kết luận).

## Boundaries (delta)

- **Never** thêm dependency; tái dùng LLM client + embedder + config.
- **Never** sửa retrieval node.
- **Ask first:** thêm field `AgentState` / đổi thứ tự node hiện có ngoài việc chèn `do_scope_gate` + `do_refusal`.
- `topics.yaml` sinh từ corpus (bán tự động, người review), không hardcode tay toàn bộ.

## Công việc

1. `src/agent/scope_gate.py`: `classify_scope(inp: ScopeGateInput) -> ScopeGateResult` (prefilter → LLM → fail-closed).
2. `data/scope/topics.yaml`: script sinh nháp từ `dieu_tieu_de` của corpus + 12 loại OOC §4; Ted/review chỉnh.
3. Prompt classifier + few-shot (12 OOC + ~6 in-scope).
4. Graph: node `do_scope_gate` + `do_refusal`; conditional edge sau `do_scope_gate`; truyền `scope_topic`/`needs_personal_data`/`individual_advice` cờ vào `AgentState` cho generator.
5. Template từ chối (`src/agent/refusal.py` hoặc trong node): 4 biến thể (OOC, NEEDS_PERSONAL_DATA, INDIVIDUAL_ADVICE, classifier_unavailable) + biến `time_out_of_range`.
6. Xoá/hạ vai trò intent `unknown` cũ trong router (scope_gate thay thế phần "ngoài phạm vi") — kiểm mọi nơi dùng `intent == "unknown"`.

## Success Criteria

- [ ] Bộ fixture `tests/fixtures/scope_cases.json` ~35 câu gán nhãn (viết TRƯỚC): 12 loại OOC (mỗi loại ≥2 câu) + ≥8 IN_SCOPE + ≥3 NEEDS_PERSONAL_DATA + ≥3 INDIVIDUAL_ADVICE + 2 time_out_of_range.
- [ ] `pytest tests/test_scope_gate.py`: accuracy trên fixture ≥ ngưỡng đặt trong test (đề xuất ≥ 0.85 với LLM thật mock bằng câu trả lời vàng; prefilter-only ≥ 0.6). Báo confusion theo `out_of_scope_kind`.
- [ ] Fail-closed test: mock LLM raise → `decision=OUT_OF_SCOPE`, reason=`classifier_unavailable`, message chứa "tạm thời chưa phân loại".
- [ ] Câu OOC thật qua API → response không gọi retrieval (assert `chunk_count == 0`), body có phạm vi corpus + ≤3 `suggested_questions`.
- [ ] Câu IN_SCOPE rõ ràng qua prefilter → không tốn LLM call tầng 2 (đo qua log/mock).
- [ ] Regression: `pytest tests/test_agent.py` xanh; câu in-scope cũ vẫn ra câu trả lời.

## Verify

`pytest tests/test_scope_gate.py tests/test_agent.py -v` + 5 câu OOC + 3 câu in-scope thủ công qua API.

## Open Questions

1. Ngưỡng `confidence` prefilter↔LLM (đề xuất 0.85) — chốt giờ hay sau eval? (Open question #3 của `SPEC.md`).
2. `conversation_summary` cho multi-turn: lấy từ đâu (đã có `_build_history_from_messages` trong graph.py) — đủ chưa?
3. `NEEDS_PERSONAL_DATA` vs intent `compliance_check` hiện có: 2 cái chồng nhau. Gộp logic (scope_gate quyết, compliance_check node thực thi) hay để song song? (liên quan `SPEC-compliance-criteria.md`).
