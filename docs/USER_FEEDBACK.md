# Feedback trải nghiệm người dùng — 2026-09-26

Claude đóng vai 3 người dùng (nhân viên văn phòng, HR, chủ doanh nghiệp nhỏ), hỏi bằng ngôn ngữ
đời thường qua `POST /api/v1/query` trên backend local (Chroma 1151 chunks, reranker tắt). Khoảng
31 câu hỏi qua API + 12 tình huống gọi thẳng `check_compliance()`. Lúc chạy, Gemini trả 503
(high demand) liên tục nên ~6/15 câu rơi xuống `extractive_fallback` — đây là lỗi upstream, không
tính là bug, nhưng nó làm lộ các mục P1 bên dưới.

Đánh dấu `[x]` khi đã sửa và ghi commit bên cạnh.

## P0 — Phán quyết ✅/❌ sai có thẩm quyền (compliance engine)

Đây là loại lỗi nguy hiểm nhất: sai, kèm trích dẫn điều luật, người dùng không có lý do để nghi.
Tái hiện: `python -c "from src.rag.compliance import check_compliance as c; print(c('<câu>'))"`.

- [ ] **F1. Không quy đổi đơn vị người dùng viết.** `extract_situation_value` lấy số rồi so thẳng
  với `condition.value` theo đơn vị của tiêu chí (`src/rag/compliance.py`).
  | Câu hỏi | Kết quả | Đúng phải là |
  |---|---|---|
  | "công ty bắt em thử việc **3 tháng**… nhân viên văn phòng" | ✅ Đạt (v=3, so với 60 **ngày**) | ❌ 90 ngày > 60 |
  | "làm thêm **3 tiếng mỗi ngày** có sao không" | ✅ Đạt, khớp tiêu chí *trần giờ/năm* (v=3) | Không phải tiêu chí năm; đây là trần theo ngày (Điều 107 k2 điểm b) |

  Hướng sửa: bắt cả đơn vị cạnh con số (`ngày|tháng|năm|giờ|tiếng|%|triệu`), quy đổi về đơn vị
  tiêu chí (tháng→30 ngày…); đơn vị không quy đổi được hoặc mâu thuẫn (giờ/ngày vs giờ/năm) →
  `insufficient_info`, không phán.

- [ ] **F2. Lương thử việc cho bằng tiền bị hiểu là phần trăm.** "lương thử việc **9 triệu**, lương
  chính thức **10 triệu**" → ❌ Không đạt (v=9, so với 85%). Thực tế 90% ≥ 85% → Đạt. Hướng sửa: với
  tiêu chí đơn vị `%`, nếu câu có 2 số tiền thì tính tỉ lệ; chỉ có 1 số tiền mà không có `%` →
  `insufficient_info`.

- [ ] **F3. Phán quyết bỏ qua điều kiện áp dụng của tiêu chí.**
  - "**công nhân phổ thông** thử việc 45 ngày" → ✅ Đạt theo ngưỡng 60 ngày (dành cho trình độ cao
    đẳng trở lên). Điều 25 còn khoản 3 (30 ngày) và khoản 4 (6 ngày làm việc) — corpus có đủ.
  - "làm thêm 250 giờ/năm **ngành dệt may**" → ❌ Không đạt. Điều 107 k3 cho phép tới 300 giờ với
    dệt, may, da, giày — tức là đúng luật. Template `fail` có nhắc ngoại lệ nhưng icon vẫn là ❌.

  Hướng sửa (tối thiểu): khi tiêu chí có ngưỡng phụ thuộc đối tượng mà câu hỏi không nêu hoặc nêu
  đối tượng khác → `insufficient_info` kèm liệt kê các mức; không trả ✅/❌. Thêm các câu trên vào
  `tests/test_compliance.py` làm regression.

## P1 — Sai/khó hiểu khi LLM lỗi hoặc câu hỏi lệch

- [ ] **F4. Chế độ trích nguyên văn phá vỡ việc từ chối ngoài phạm vi.** "Thuế thu nhập cá nhân tính
  thế nào?" khi Gemini lỗi → hiện 3 điều khoản BHXH/lao động không liên quan, gắn nhãn "các điều
  khoản liên quan nhất". Cùng câu đó khi Gemini chạy được thì trả "nằm ngoài phạm vi". Hệ quả: câu
  ngoài phạm vi chỉ bị chặn khi LLM còn sống. Hướng sửa: cần một cổng phạm vi không phụ thuộc
  generator (vd ngưỡng điểm rerank/dense), hoặc để fallback nói rõ "có thể không liên quan".

- [ ] **F5. Không hiểu năm ghi trong câu hỏi.** "Mức lương tối thiểu vùng I **năm 2025**…" với
  `as_of_date` trống → trả mức 2026 (5.310.000đ, NĐ 293/2025). Người dùng thường không mở ô chọn
  ngày mà ghi năm luôn vào câu. Hướng sửa: nếu câu có "năm YYYY" và người dùng không chọn ngày →
  lấy làm `as_of_date` (vd 31/12 năm đó hoặc 01/07), và nói rõ đã hiểu như vậy.

- [ ] **F6. Rác chữ ký số lọt vào thân điều khoản.** Điều 3 NĐ 293/2025 hiển thị cho người dùng:
  "…theo vùng như sau: Người ký: CỔNG THÔNG TIN ĐIỆN TỬ CHÍNH PHỦ Email: thongtinchinhphu@chinhphu.vn
  Cơ quan: VĂN PHÒNG CHÍNH PHỦ Thời gian ký: 19.11.2025…". 13/20 file trong `data/raw/lao_dong/`
  có dòng này (`grep -c "Người ký:"`). Cần lọc lúc ingest rồi ingest lại + migrate Qdrant.

- [ ] **F7. Tên nguồn không phân biệt được bản cũ/mới.** Nguồn hiện "Luật Bảo hiểm xã hội — Điều 98"
  cho cả 58/2014/QH13 và 41/2024/QH15; "Quy định mức lương tối thiểu…" cho cả 74/2024 và 293/2025.
  Với sản phẩm bán tính năng tra cứu theo thời điểm, nguồn phải kèm số hiệu (và năm).

- [ ] **F8. Hỏi theo địa danh không kéo được Phụ lục vùng.** "em làm ở Hà Nội thì lương tối thiểu là
  bao nhiêu" → top 3 là Điều 3/5/1 của NĐ 293, không có Phụ lục (dòng 104 của file có danh sách
  phường Hà Nội thuộc vùng I). Người dùng không biết mình ở "vùng" nào — đây là câu hỏi rất phổ biến.

## P2 — Mài giũa trải nghiệm

- [ ] **F9. Lỗi 429 hiện "❌ Query failed" tiếng Anh trên UI.** slowapi trả `{"error": "Rate limit
  exceeded: 10 per 1 minute"}` (không có `detail`) nên `extractErrorMessage` rơi về chuỗi mặc định
  (`frontend/src/main.tsx` `api.query`). Nên nói "Bạn hỏi hơi nhanh, thử lại sau ít giây".
- [ ] **F10. Câu 1 ký tự trả lỗi 422 thô của pydantic** ("String should have at least 2 characters").
  UI nên chặn trước khi gửi, hoặc API trả message tiếng Việt.
- [ ] **F11. Ngày ISO lộ ra trong câu trả lời:** "tại thời điểm **2026-09-26**" (câu nghỉ phép, thai
  sản), trong khi câu khác viết "26/09/2026". Prompt nên truyền ngày dạng `dd/mm/yyyy`.
- [ ] **F12. Từ chối ngoài phạm vi vẫn gắn trích dẫn và nguồn.** "Thuế TNCN" → "nằm ngoài phạm vi"
  rồi gợi ý 3 câu hỏi kèm `[3]`, `[5]` và hiện 3 nguồn. Câu từ chối không nên có nguồn.
- [ ] **F13. `used_llm` rỗng** ở câu "sa thải nhân viên tự ý bỏ việc bao nhiêu ngày" (câu trả lời
  vẫn đúng). Log/analytics sẽ đếm thiếu.
- [ ] **F14. Độ trễ:** câu thường 4–15s, có câu 27–38s (tuổi nghỉ hưu 32s, lương tối thiểu Hà Nội
  38,6s rồi vẫn rơi fallback). Nên đặt trần thời gian cho vòng xoay key×model — khi 3 key đều 503
  thì chuyển fallback sớm thay vì chờ hết mọi cặp.

## Những gì đã tốt (giữ nguyên)

- Nhớ ngữ cảnh: "thế lương thử việc được bao nhiêu phần trăm" sau câu thử việc → đúng 85%, Điều 26.
- `as_of_date=2025-06-01` → đúng NĐ 74/2024 (4.960.000đ); tuổi nghỉ hưu nam 2026 = 61 tuổi 6 tháng
  (đúng lộ trình); Luật BHXH 2024 hiệu lực 01/07/2025 kèm bối cảnh chuyển tiếp.
- Câu trả lời khi Gemini chạy được: gọn, có bảng, trích dẫn đúng điều (nghỉ phép Điều 113/114,
  thanh toán khi nghỉ việc Điều 48, sa thải Điều 125, thai sản Điều 139).
- Chặn prompt injection bằng tiếng Việt; lời chào ("chào bạn", "ok") trả giới thiệu phạm vi ngay.
- Compliance đúng ở các ca thẳng: làm thêm 350 giờ/năm ❌, lương thử việc 80% ❌, 5 triệu vùng I ❌,
  nghỉ phép 10 ngày ❌.
