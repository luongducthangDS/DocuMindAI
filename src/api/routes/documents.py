"""
Document management routes:
  POST /api/v1/upload  — ingest uploaded PDF
  GET  /api/v1/documents — list indexed documents
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File, status
from loguru import logger

from src.api.schemas import DocumentListResponse, DocumentMeta, IngestResponse
from src.config import get_settings
from src.ingestion.chunker import chunk_by_dieu
from src.ingestion.loader import load_pdf

router = APIRouter(prefix="/api/v1", tags=["documents"])

# In-memory document registry (replaced by DB in production)
_doc_registry: dict[str, DocumentMeta] = {}


@router.post("/upload", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)) -> IngestResponse:
    """
    Upload a PDF and ingest it into the vector store.
    Security checks: file size, MIME type, PDF magic bytes.
    """
    settings = get_settings()
    from src.api.main import ensure_rag_initialized

    await ensure_rag_initialized()

    # Validate MIME type before reading body
    allowed_types = {"application/pdf", "application/x-pdf"}
    content_type = (file.content_type or "").lower()
    if content_type not in allowed_types and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are accepted",
        )

    # Read with size guard
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_size_mb} MB limit",
        )

    try:
        doc = load_pdf(data, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error("PDF processing failed: {}", exc)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Failed to extract text from PDF")

    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="PDF produced no extractable text",
        )

    # Chunk and index
    chunks = chunk_by_dieu(doc["content"], doc)
    if not chunks:
        return IngestResponse(
            status="error",
            indexed_chunks=0,
            document_title=doc["title"],
            message="No valid chunks extracted from document",
        )

    indexed = await _index_chunks(chunks, doc)
    doc_id = str(uuid.uuid4())
    _doc_registry[doc_id] = DocumentMeta(
        id=doc_id,
        title=doc["title"],
        doc_type=doc.get("doc_type", "uploaded_pdf"),
        source="user_upload",
        url=doc.get("url", ""),
        chunk_count=indexed,
    )

    return IngestResponse(
        status="success",
        indexed_chunks=indexed,
        document_title=doc["title"],
        message=f"Đã index {indexed} chunks vào ChromaDB",
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """List all indexed documents."""
    docs = list(_doc_registry.values())
    return DocumentListResponse(total=len(docs), documents=docs)


@router.post("/reload", status_code=status.HTTP_200_OK)
async def reload_retriever() -> dict:
    """
    Reload the hybrid retriever from ChromaDB.
    Call this after CLI ingestion to pick up new documents without restarting.
    """
    import asyncio
    import src.rag.retriever as r_module
    from src.api.main import ensure_rag_initialized

    await ensure_rag_initialized()

    if r_module._active_index is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG index not initialized",
        )

    await asyncio.to_thread(_rebuild_retriever, r_module)
    return {"status": "ok", "message": "Retriever reloaded"}


async def _index_chunks(chunks: list, doc: dict) -> int:
    """Add chunks to the active vector index and refresh retriever. Returns count indexed."""
    import asyncio

    try:
        import src.rag.retriever as r_module
        from llama_index.core.schema import TextNode

        index = getattr(r_module, "_active_index", None)
        if index is None:
            logger.warning("No active index found — chunks not persisted")
            return 0

        nodes = [
            TextNode(text=c.text, metadata=c.metadata)
            for c in chunks
            if c.is_valid
        ]
        index.insert_nodes(nodes)
        logger.info("Indexed {} nodes for '{}'", len(nodes), doc["title"][:40])

        # Rebuild retriever in background thread so BM25 corpus includes new nodes
        await asyncio.to_thread(_rebuild_retriever, r_module)

        return len(nodes)

    except Exception as exc:
        logger.error("Indexing failed: {}", exc)
        return 0


def _rebuild_retriever(r_module) -> None:
    """Reload all nodes from ChromaDB and rebuild the hybrid retriever."""
    try:
        from src.rag.embedder import get_chroma_collection
        from src.rag.retriever import build_hybrid_retriever
        from src.api.main import _load_nodes_from_collection
        from src.config import get_settings

        _, collection = get_chroma_collection()
        nodes = _load_nodes_from_collection(collection)
        r_module._active_retriever = build_hybrid_retriever(
            r_module._active_index, nodes=nodes, rerank=get_settings().enable_reranker
        )
        logger.info("Retriever rebuilt with {} nodes after upload", len(nodes))
    except Exception as exc:
        logger.warning("Retriever rebuild failed (non-fatal): {}", exc)
