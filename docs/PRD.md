# PRD: DocuMind AI

Nguồn sự thật về phạm vi: `src/config.py` (`DOMAIN_NAME` / `DOMAIN_SCOPE` / `DOMAIN_TOPICS`) và `frontend/src/main.tsx` (`PRODUCT`). Mô tả đầy đủ: [README](../README.md), `CLAUDE.md` mục "Project overview", thư mục [spec/](spec/). File này chỉ tóm tắt.

## Bài toán
AI agent RAG tra cứu pháp luật lao động và bảo hiểm xã hội Việt Nam (Bộ luật Lao động, Luật BHXH, Luật Việc làm và nghị định/thông tư hướng dẫn): trả lời kèm trích dẫn điều khoản, từ chối khi ngoài phạm vi tài liệu đã nạp, tra cứu theo thời điểm (`as_of_date`) vì nhiều văn bản đã bị thay thế; kiểm tra tuân thủ theo tiêu chí định lượng.

## Người dùng
Người làm nhân sự / pháp chế doanh nghiệp, người lao động tra cứu quyền lợi; nhà tuyển dụng (README: "portfolio AI mức production"). (unverified: nhóm người dùng nghiệp vụ suy ra từ phạm vi sản phẩm)

## Mục tiêu
1. Câu trả lời có trích dẫn đúng điều khoản, đúng hiệu lực theo thời điểm.
2. Từ chối ngoài phạm vi; không bịa.
3. Kiểm tra tuân thủ bằng tiêu chí trong `data/compliance/criteria.json`, mỗi tiêu chí trích văn bản có trong corpus.
4. Eval tái lập (legal_qa_200, multi-hop, RAGAS) và ACL/nDCG trên nhánh platform.

## Không làm
Văn bản ngoài phạm vi lao động/BHXH (corpus ngân hàng cũ đã bỏ); tư vấn pháp lý thay luật sư. (unverified: câu chữ disclaimer chính xác, xem `PRODUCT` trong frontend)
