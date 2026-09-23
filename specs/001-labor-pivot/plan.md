# Implementation Plan: Corpus có biên giới cho vertical Lao động – Tiền lương – BHXH

**Branch**: `001-labor-pivot` | **Date**: 2026-09-18 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-labor-pivot/spec.md`

## Summary

Initiative đưa DocuMind AI từ "trợ lý RAG không có biên giới" sang "hệ thống tra cứu lao động – tiền lương – BHXH biết rõ mình phủ tới đâu". Ba trụ: (1) manifest corpus công khai trên giao diện, (2) cổng phạm vi fail-closed chạy trước mọi truy hồi, (3) truy hồi theo thời điểm hiệu lực ở cấp điều/khoản. Cộng thêm thư viện câu hỏi tình huống và gold set ≥ 50 câu viết trước để mọi con số công bố đều có nguồn.

Cách tiếp cận kỹ thuật: tái dùng toàn bộ hạ tầng hiện có (LangGraph, hybrid retriever, embedder, config) và **không thêm phụ thuộc runtime mới**. Lọc thời điểm đã tồn tại như một node riêng ở tầng trên truy hồi (`temporal_filter_node`), không sửa `retrieve_node`. Cổng phạm vi cắm vào graph **trước** `router`, dùng lại LLM client sẵn có, với đường thoát fail-closed khi phân loại lỗi.

## Technical Context

**Language/Version**: Python 3.10.11 (chạy thực tế), target 3.11; TypeScript/React 19 cho frontend

**Primary Dependencies**: FastAPI 0.115 + uvicorn, LangGraph 0.2, langchain-core, sentence-transformers, ChromaDB 0.6, rank_bm25, Vite 6

**Storage**: ChromaDB `PersistentClient` tại `data/chroma_db/` (collection `documind_legal`, gắn nhãn `embedding_model`/`embedding_dim`); Qdrant Cloud là provider thay thế qua `VECTOR_STORE_PROVIDER`; corpus nguồn là Markdown tại `data/raw/lao_dong/` (21 văn bản); log là JSONL. Không có cơ sở dữ liệu quan hệ.

**Testing**: pytest. Đo thực tế ngày 2026-09-18: **216 passed** trong 67s (`pytest -q`, interpreter `Python310`, bỏ addopts coverage). Eval riêng: `eval/run_evals.py`, `eval/temporal_eval.py`, `eval/embedding_ab.py`.

**Target Platform**: Backend Linux container (Render), frontend static (Vercel); dev trên Windows 11 qua `start.ps1`

**Project Type**: Web service + SPA (backend `src/`, frontend `frontend/`)

**Performance Goals**: Truy hồi + sinh trong vài giây cho một truy vấn người dùng; cổng phạm vi không được làm tăng đáng kể độ trễ vì nó chạy trước mọi thứ khác. Không có mục tiêu thông lượng cao — đây là công cụ tra cứu, không phải API công cộng lưu lượng lớn.

**Constraints**: Không thêm dependency runtime mới cho toàn initiative. Lọc thời điểm phải nằm ở tầng trên truy hồi, không sửa `retrieve_node`/`build_hybrid_retriever`. Cổng phạm vi phải fail-closed. Mô hình embedding chỉ đọc từ `EMBEDDING_MODEL`.

**Scale/Scope**: 21 văn bản lao động trong corpus nguồn; vector store hiện có 1146 chunk; gold set mục tiêu ≥ 50 câu; thư viện 40–60 câu; 12 loại câu ngoài phạm vi.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Đối chiếu với `.specify/memory/constitution.md` v1.0.0:

| Nguyên tắc | Trạng thái | Căn cứ |
|---|---|---|
| I. Grounded Answers or Refusal | ⚠️ PASS có điều kiện | Ngưỡng `score < 0.05` và temperature `0.0` đã có trong `src/rag/generator.py`. Nhưng FR-010 (câu trả lời nêu thời điểm hiệu lực đang áp dụng) chưa có trong đường sinh câu trả lời. |
| II. Corpus Boundary Is the Product | ❌ FAIL | `src/agent/graph.py` không có node cổng phạm vi: các node là contextualize → router → retrieve → temporal_filter → grade → answer. Không có API hay thành phần giao diện nào đọc `docs/corpus/corpus_manifest.yaml`. Đây là nguyên tắc mà chính initiative này sinh ra để thực thi, và nó đang chưa được thực thi. |
| III. Evidence-Based Evaluation | ⚠️ PARTIAL | Chấm theo `clause_uid`/`version_id` đã có (DEC-0004, `eval/metrics.py`). Gold set hiện 25 câu ở `data/eval/test_questions.json` + 30 câu temporal — chưa đạt ngưỡng ≥ 50 câu phủ đủ ba nhóm, và nhóm "ngoài phạm vi" chưa tồn tại vì chưa có cổng phạm vi để chấm. |
| IV. Configuration Over Hard-Coding | ✅ PASS | `EMBEDDING_MODEL` là nguồn duy nhất, `EmbeddingModelMismatch` bảo vệ collection, `src/rag/vector_backend.py` che provider, `src/hf_env.py` gom cache HF về một chỗ. |
| V. Green Test Suite Before Claim | ❌ FAIL | `pytest -q` xanh (216/216) nhưng **không tài liệu nào nói đúng con số**: README ghi "84/84 (100%)", CLAUDE.md ghi "166/166 (đo 2026-09-16)". Cả hai đều sai so với lần chạy 2026-09-18. |

**Kết luận gate**: KHÔNG pass sạch. Hai vi phạm (II, V) không phải là lý do dừng thiết kế — chúng chính là công việc mà feature này tồn tại để giải. Chúng được chuyển thành hạng mục bắt buộc trong `tasks.md` và được theo dõi ở Complexity Tracking bên dưới.

## Project Structure

### Documentation (this feature)

```text
specs/001-labor-pivot/
├── plan.md              # File này
├── spec.md              # Đặc tả (đã có)
├── research.md          # Phase 0
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1
├── contracts/           # Phase 1
├── checklists/
│   └── requirements.md  # Checklist chất lượng spec (đã có)
└── tasks.md             # Phase 2 (do /speckit-tasks tạo)
```

### Source Code (repository root)

```text
src/
├── api/
│   ├── main.py              # khởi tạo retriever, health
│   ├── schemas.py           # đã mang as_of_date
│   └── routes/
│       ├── query.py         # đường vào truy vấn (nơi as_of_date đi qua)
│       ├── documents.py
│       └── reports.py
│       └── corpus.py        # [MỚI] endpoint manifest corpus
├── agent/
│   ├── graph.py             # [SỬA] chèn scope_gate trước router
│   ├── scope.py             # [MỚI] bộ phân loại phạm vi + fail-closed
│   └── memory.py
├── rag/
│   ├── retriever.py         # không đổi (ràng buộc spec)
│   ├── temporal.py          # lọc theo as_of_date (đã có)
│   ├── generator.py         # [SỬA] nêu thời điểm hiệu lực trong câu trả lời
│   └── vector_backend.py
└── ingestion/
    ├── chunker.py, manifest.py, versions.py   # clause_uid/version_id (đã có)

frontend/src/
├── main.tsx                 # [SỬA] khối manifest ở trang chủ + thư viện câu hỏi
└── styles.css

data/
├── raw/lao_dong/            # 21 văn bản nguồn
├── eval/
│   ├── test_questions.json      # [THAY] gold set ≥ 50 câu, 3 nhóm
│   └── temporal_questions.json  # 30 câu temporal (đã có)
└── questions/               # [MỚI] thư viện câu hỏi tình huống

docs/
├── corpus/corpus_manifest.yaml  # nguồn sự thật cho manifest hiển thị
├── spec/                        # spec module gốc của initiative
└── decisions/                   # ADR
```

**Structure Decision**: Giữ nguyên bố cục hiện tại (backend `src/` + frontend `frontend/`). Ba điểm chèn mới: `src/agent/scope.py` (cổng phạm vi), `src/api/routes/corpus.py` (đọc manifest), và `data/questions/` (thư viện câu hỏi tĩnh). Không tạo package hay service mới — ràng buộc "không thêm dependency" và nguyên tắc IV đều nghiêng về mở rộng tại chỗ.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Nguyên tắc II đang FAIL khi vào Phase 0 | Cổng phạm vi và manifest chính là nội dung của feature này; không thể pass trước khi làm | Không có phương án đơn giản hơn — bỏ cổng phạm vi là bỏ chính định vị sản phẩm |
| Nguyên tắc V đang FAIL khi vào Phase 0 | Tài liệu công bố sai số test là nợ tồn từ trước initiative | Không thể hoãn: sửa số test trong tài liệu là việc một dòng, để lại là vi phạm nguyên tắc ngay ở gate kế tiếp |
| Thêm một node chặn trước router | Biên giới corpus phải cưỡng chế trước khi tốn chi phí truy hồi, và phải chặn được cả khi router đoán sai intent | Kiểm tra sau truy hồi bị loại: câu ngoài phạm vi vẫn tốn truy hồi và vẫn có thể lọt vào ngữ cảnh sinh |
