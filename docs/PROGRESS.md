# Progress

Chi tiết đầy đủ: `CLAUDE.md` / `AGENTS.md` mục "Trạng thái hiện tại", và `AGENTS.md` của thư mục chung `01-ai-engineering`.

## Trạng thái (2026-11-11)
- Nhánh đang mở: `fix/oos-fallback-model-verdict-gold` (đồng bộ origin) với 52 file thay đổi chưa commit (session khác).
- **Local dùng Gemini** từ 09/11 (FPT key 401): flash-lite, `gemini-embedding-001`, collection `documind_legal_gemini001` (1249 đoạn, fingerprint `5ce81d96ccb9`), reranker tắt.
- Đo 10/11 trên Gemini: legal_qa_200 pre_filter 0,933 (FPT 0,990), agentic 0,947; multi-hop pre_filter 0,826, agentic 0,913; p50 3,0 s / 4,3 s. 8/208 câu rơi về trích nguyên văn do 429/503.
- **Production** (Render + Qdrant `documind_legal_fpt`) vẫn trỏ FPT nên đang hỏng tới khi có key mới hoặc đổi env.
- PR #13: job test pass, job eval fail vì secret FPT 401.

## Bước tiếp theo
Mục đầu tiên chưa tick trong `TASKS.md` (cần key / quyết định của user).
