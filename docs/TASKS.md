# Tasks

## Đã xong (gần đây)
- [x] GraphRAG mở rộng theo dẫn chiếu (`GRAPH_EXPAND=1`) + vòng research agent (09/11, theo AGENTS.md thư mục chung: chưa commit)
- [x] Local chuyển hẳn sang Gemini, đo lại legal_qa_200 và multi-hop (10/11)
- [x] Nhánh `feat/platform-upgrade` (worktree `DocuMindAI-platform/`): DOCX/XLSX, human approval, ACL eval, nDCG, `/api/v1/metrics` — đã push, chưa PR

## Tiếp theo (lần lượt từ trên xuống)
- [ ] Có FPT key mới (401 từ 09/11) hoặc chuyển production sang Gemini: production (Render + Qdrant `documind_legal_fpt`) đang hỏng
- [ ] Thay secret `FPT_API_KEY` trong GitHub Actions để job eval của PR #13 chạy lại
- [ ] Hoàn tất và commit WIP hiện tại trên `fix/oos-fallback-model-verdict-gold` (52 file, của session khác)
- [ ] Gom quy tắc từ `CLAUDE.md`/`AGENTS.md` về `docs/RULES.md`, rút `AGENTS.md` thành điểm vào ngắn và `CLAUDE.md` thành `@AGENTS.md` (làm sau khi WIP xong để không đè session khác)
- [ ] Tách phần kém ~5 điểm của Gemini so với FPT: flash-lite hay thiếu reranker
