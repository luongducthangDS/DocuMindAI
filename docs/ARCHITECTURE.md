# Architecture

Bản đầy đủ nằm trong `CLAUDE.md`: mục "Cấu trúc thư mục quan trọng", "Config quan trọng (.env)", "Corpus & Compliance", "Các lưu ý kỹ thuật". File này chỉ để định hướng; không chép lại.

## Stack
Python 3.10+ · FastAPI · LangGraph · Chroma (local) / Qdrant Cloud (production) qua `vector_backend` · hybrid retrieval BM25 + dense + RRF + reranker `bge-reranker-v2-m3` (tắt ở local từ 09/11) · LLM/embedding FPT AI Marketplace (`Vietnamese_Embedding` 1024-dim) hoặc Gemini (`gemini-embedding-001`) · React 19 + Vite · Render · GitHub Actions (job test + job eval).

## Định hướng nhanh
```
src/api/            FastAPI: query (REST + WebSocket), documents, health, reports, metrics
src/rag/            embedder, vector_backend (đổi VECTOR_STORE_PROVIDER không sửa call site), retriever, generator, grader
src/agent/research.py   vòng research agent (AGENT_RESEARCH_ROUNDS) — xem AGENTS.md của thư mục chung
data/raw/lao_dong/  20 văn bản (+ _versions/), index 19 doc_id
data/compliance/criteria.json   6 tiêu chí định lượng
frontend/src/main.tsx   UI + PRODUCT
scripts/            ingest_documents.py, restore_chroma_from_qdrant.py, eval
```

## Danh tính index
Index được nhận diện bằng hash nội dung (`vector_backend.corpus_fingerprint`), không bằng số chunk. So hash `ingest_documents.py --dry-run` với hash trong log app.
