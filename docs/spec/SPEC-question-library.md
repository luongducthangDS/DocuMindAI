# Spec: `question-library`

> Module 5/9. Phụ thuộc: `clause-schema-ingestion`. Xem `SPEC.md` + memory `documind-labor-pivot` (thay đổi sản phẩm #2).

## Objective

Thư viện câu hỏi **tĩnh** 40–60 câu sinh từ corpus, **gom theo tình huống** (không theo tên văn bản). Hai người dùng:
1. `scope-gate`: nguồn cho `suggested_questions` (so khớp embedding).
2. `frontend-boundary-ui`: hiển thị "bạn có thể hỏi gì" ở trang chủ + panel gợi ý.

## Contract: `data/questions/library.json`

```json
{
  "situations": [
    {
      "id": "bi_cho_thoi_viec",
      "label": "Bị cho thôi việc / chấm dứt HĐLĐ",
      "questions": [
        {
          "id": "thoi_viec_01",
          "text": "Công ty đơn phương chấm dứt hợp đồng lao động cần báo trước bao nhiêu ngày?",
          "primary_dieu": ["45-2019-QH14__d36", "45-2019-QH14__d37"],
          "scope_topic": "cham_dut_hdld"
        }
      ]
    }
  ],
  "generated_at": "2026-09-..",
  "corpus_version": "<hash/ngày manifest>"
}
```

- `primary_dieu`: `clause_uid` điều liên quan (để verify câu trả lời được, và cho eval-goldset dùng lại).
- `scope_topic`: khớp `scope_topic` của `ScopeGateResult`.
- Nhóm tình huống đề xuất (≥8): bị cho thôi việc · tính trợ cấp thôi việc/mất việc · nghỉ thai sản – thai sản · làm thêm giờ – nghỉ phép · lương tối thiểu vùng · đóng BHXH bắt buộc/tự nguyện · hưởng lương hưu – tuổi hưu · trợ cấp thất nghiệp · hợp đồng lao động – thử việc · kỷ luật lao động.

## Boundaries (delta)

- **Never** sinh câu hỏi mà corpus không trả lời được — mỗi câu phải map tới `clause_uid` tồn tại + đã VERIFIED.
- **Never** dùng LLM sinh câu rồi commit thẳng — nếu sinh bằng LLM, người review từng câu + đối chiếu `primary_dieu` trước khi commit (giống `eval-methodology`: không fit theo output).
- Không thêm dependency.

## Công việc

1. `scripts/gen_question_library.py`: duyệt `dieu_tieu_de` toàn corpus → nhóm theo tình huống (mapping tình huống→điều viết tay hoặc LLM-assisted + review) → sinh nháp câu hỏi.
2. Review + chỉnh tay `library.json` (40–60 câu, ≥8 tình huống, mỗi tình huống 3–8 câu).
3. `src/agent/question_library.py`: `load_library()`, `nearest_questions(query, k=3) -> list[str]` (embedding cosine vs `text`, tái dùng embedder).
4. Test cấu trúc + test `nearest_questions` trả câu cùng `scope_topic` cho query mẫu.

## Success Criteria

- [ ] `data/questions/library.json`: 40–60 câu, ≥8 tình huống; mỗi câu có `primary_dieu` trỏ `clause_uid` tồn tại trong index.
- [ ] `scripts/validate_corpus.py` (hoặc test riêng) kiểm mọi `primary_dieu` khớp một chunk đã index.
- [ ] `nearest_questions("thu nhập 8 triệu có được vay không")` → trả câu thuộc tình huống hợp lý hoặc rỗng nếu quá xa (đây là câu OOC — không nên khớp mạnh).
- [ ] `tests/test_question_library.py` xanh.
- [ ] `scope-gate` dùng được `nearest_questions` cho `suggested_questions` (integration test ở `SPEC-scope-gate.md`).

## Verify

`pytest tests/test_question_library.py` + đọc tay 10 câu ngẫu nhiên đối chiếu `primary_dieu` với text điều.

## Open Questions

1. Sinh bằng LLM (1 lần, offline, review) hay viết tay 100%? (Open question #4 của `SPEC.md`).
2. Mapping tình huống→điều: viết tay bảng (~10 tình huống × vài điều) đủ chưa, hay cần bán tự động?
3. Có versioning library theo `corpus_version` để phát hiện lệch khi corpus đổi không? (đề xuất: có field, chưa cần cơ chế check tự động).
