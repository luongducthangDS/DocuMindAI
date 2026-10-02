"""
Document management routes:
  POST /api/v1/upload  — ingest uploaded PDF (admin: X-Admin-Key)
  GET  /api/v1/documents — list indexed documents
  POST /api/v1/reload  — rebuild retriever (admin: X-Admin-Key)
"""

from __future__ import annotations

import asyncio
from collections import Counter

from fastapi import APIRouter, Depends, HTTPException, Request, status
from loguru import logger
from starlette.datastructures import UploadFile

from src.agent.memory import get_long_term_memory
from src.api.principal import require_admin, resolve_context
from src.api.routes.query import _get_client_ip
from src.api.schemas import DocumentListResponse, DocumentMeta, IngestResponse
from src.config import get_settings
from src.ingestion.chunker import chunk_by_dieu
from src.ingestion.loader import load_pdf

router = APIRouter(prefix="/api/v1", tags=["documents"])

# Budget cho extract + chunk một PDF. Quá hạn thì trả lỗi, nhưng thread vẫn chạy
# tới xong (không huỷ được thread) — MAX_PDF_PAGES mới là trần thật của nó.
PDF_PROCESSING_TIMEOUT_S = 30.0
# Content-Length là cả body multipart (boundary + header phần), không chỉ file.
_MULTIPART_OVERHEAD_BYTES = 64 * 1024


def _check_content_length(request: Request) -> None:
    """Từ chối theo header, trước khi đọc byte body nào."""
    raw = request.headers.get("content-length")
    if raw is None:
        raise HTTPException(status_code=status.HTTP_411_LENGTH_REQUIRED, detail="Content-Length required")
    try:
        length = int(raw)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Content-Length") from None
    settings = get_settings()
    if length < 0 or length > settings.max_upload_bytes + _MULTIPART_OVERHEAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_size_mb} MB limit",
        )


def _extract_and_chunk(source, filename: str):
    """Phần CPU-bound của upload (pdfplumber + chunker) — chạy trong thread."""
    doc = load_pdf(source, filename)
    return doc, (chunk_by_dieu(doc["content"], doc) if doc else [])

# Văn bản đang có trong index, gom từ chính các node nạp cho BM25 (init, reload,
# sau upload). Bản trước chỉ đọc registry upload trong RAM: corpus ingest bằng
# script không bao giờ hiện, tab "Văn bản đã lập chỉ mục" luôn báo 0 dù index đủ.
# Mỗi phần tử: (tenant_id, DocumentMeta) — tenant giữ ngoài schema response.
_indexed_docs: list[tuple[str, DocumentMeta]] = []


def set_indexed_documents(nodes: list) -> None:
    """Gom node theo (tenant, doc_id) — doc_id thiếu (PDF upload) thì theo title."""
    groups: dict[tuple[str, str], dict] = {}
    counts: Counter = Counter()
    for node in nodes:
        m = node.metadata
        key = (m.get("tenant_id") or "public", m.get("doc_id") or m.get("title", ""))
        groups.setdefault(key, m)
        counts[key] += 1

    global _indexed_docs
    _indexed_docs = sorted(
        (
            (tenant, DocumentMeta(
                id=doc_id,
                title=m.get("title") or doc_id,
                doc_type=m.get("doc_type", ""),
                source=m.get("source", ""),
                url=m.get("source_url") or m.get("url", ""),
                so_hieu=m.get("so_hieu", ""),
                ngay_ban_hanh=m.get("ngay_ban_hanh", ""),
                chunk_count=counts[(tenant, doc_id)],
            ))
            for (tenant, doc_id), m in groups.items()
        ),
        key=lambda td: td[1].ngay_ban_hanh,
        reverse=True,
    )


def _upload_label(ctx) -> str:
    """Most restrictive label the uploading tenant holds."""
    ranked = ["confidential", "internal", "public"]
    for label in ranked:
        if label in ctx.acl_labels:
            return label
    return "public"


def _audit_upload_error(filename: str, size: int, error: str, ip: str, ua: str) -> None:
    try:
        get_long_term_memory().log_upload(
            filename=filename,
            file_size_bytes=size,
            indexed_chunks=0,
            status="error",
            error_detail=error[:300],
            ip_address=ip,
            user_agent=ua,
        )
    except Exception:
        pass  # audit failure must never block the user-facing error response


@router.post(
    "/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_admin)],
)
async def upload_document(request: Request) -> IngestResponse:
    """
    Upload a PDF (multipart field `file`) and ingest it into the vector store.
    Security checks, in order: admin key, Content-Length, MIME, size, magic bytes,
    page count, processing timeout.

    Không khai báo `file: UploadFile = File(...)`: FastAPI đọc xong cả body
    multipart TRƯỚC khi chạy dependency, nên require_admin và kiểm Content-Length
    sẽ chạy sau khi người gọi (kể cả ẩn danh) đã đẩy hết file lên. Tự parse form
    ở đây mới giữ được thứ tự: xác thực → kiểm kích thước → đọc body.
    """
    _check_content_length(request)

    settings = get_settings()
    ctx = resolve_context(request)
    ip = _get_client_ip(request)
    ua = request.headers.get("user-agent", "")[:200]

    form = await request.form(max_files=1, max_fields=1)
    file = form.get("file")
    if not isinstance(file, UploadFile):
        await form.close()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Missing multipart field 'file'",
        )
    filename = file.filename or "unknown.pdf"

    from src.api.main import ensure_rag_initialized

    await ensure_rag_initialized()

    # Validate MIME type before reading body
    allowed_types = {"application/pdf", "application/x-pdf"}
    content_type = (file.content_type or "").lower()
    if content_type not in allowed_types and not filename.lower().endswith(".pdf"):
        await form.close()
        _audit_upload_error(filename, 0, "Invalid MIME type", ip, ua)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are accepted",
        )

    # Không `await file.read()`: Starlette đã spool upload > 1MB ra file tạm trên
    # disk; thread đọc thẳng từ đó. Content-Length đã chặn trần body, đây là trần
    # của riêng file (UploadFile.size do parser multipart đếm khi ghi).
    size = file.size or 0
    if size > settings.max_upload_bytes:
        await form.close()
        _audit_upload_error(filename, size, "File too large", ip, ua)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_size_mb} MB limit",
        )

    try:
        doc, chunks = await asyncio.wait_for(
            asyncio.to_thread(_extract_and_chunk, file.file, filename),
            timeout=PDF_PROCESSING_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning("PDF processing timed out after {}s: {}", PDF_PROCESSING_TIMEOUT_S, filename)
        _audit_upload_error(filename, size, "Processing timeout", ip, ua)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="PDF processing timed out")
    except ValueError as exc:
        _audit_upload_error(filename, size, str(exc), ip, ua)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error("PDF processing failed: {}", exc)
        _audit_upload_error(filename, size, f"PDF processing error: {exc}", ip, ua)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Failed to extract text from PDF")
    finally:
        # Quá timeout thì thread vẫn chạy; đóng file tạm làm lần đọc kế tiếp của nó
        # lỗi và kết thúc sớm thay vì parse tiếp.
        await form.close()

    if doc is None:
        _audit_upload_error(filename, size, "No extractable text", ip, ua)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="PDF produced no extractable text",
        )

    if not chunks:
        get_long_term_memory().log_upload(
            filename=filename, file_size_bytes=size, indexed_chunks=0,
            status="error", error_detail="No valid chunks extracted",
            ip_address=ip, user_agent=ua,
        )
        return IngestResponse(
            status="error",
            indexed_chunks=0,
            document_title=doc["title"],
            message="No valid chunks extracted from document",
        )

    # _index_chunks rebuild retriever → set_indexed_documents, danh sách tự có file này.
    indexed = await _index_chunks(chunks, doc, ctx=ctx)

    get_long_term_memory().log_upload(
        filename=filename,
        file_size_bytes=size,
        indexed_chunks=indexed,
        status="success" if indexed > 0 else "partial",
        ip_address=ip,
        user_agent=ua,
    )

    return IngestResponse(
        status="success",
        indexed_chunks=indexed,
        document_title=doc["title"],
        message=f"Đã index {indexed} chunks vào ChromaDB",
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(request: Request) -> DocumentListResponse:
    """List the indexed documents (corpus + uploads) the caller's tenant may see.

    Filtering the chunks is not enough on its own: a listing that returned every
    tenant's uploads would still disclose what other tenants hold — titles are
    often the sensitive part ("Phương án cắt giảm lao động 2026").
    """
    from src.api.main import ensure_rag_initialized

    # Index rỗng/hỏng thì 503, không trả "0 văn bản" như thể hệ thống trống thật.
    await ensure_rag_initialized()
    visible = set(resolve_context(request).visible_tenants)
    docs = [d for tenant, d in _indexed_docs if tenant in visible]
    return DocumentListResponse(total=len(docs), documents=docs)


@router.post("/reload", status_code=status.HTTP_200_OK, dependencies=[Depends(require_admin)])
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


async def _index_chunks(chunks: list, doc: dict, ctx=None) -> int:
    """Add chunks to the active vector index and refresh retriever. Returns count indexed.

    Every node is stamped with the uploader's tenant and ACL label before it is
    written. An unstamped chunk would default to tenant `public` at query time
    (RetrievalContext.allows), i.e. one tenant's upload would be readable by all
    of them — so stamping happens here, at the only write path for uploads.
    """
    import asyncio

    try:
        import src.rag.retriever as r_module
        from llama_index.core.schema import TextNode

        index = getattr(r_module, "_active_index", None)
        if index is None:
            logger.warning("No active index found — chunks not persisted")
            return 0

        from src.rag.context import PUBLIC_CONTEXT, stamp_access_meta

        ctx = ctx or PUBLIC_CONTEXT
        # An uploader writes at the *least* privileged label it holds: a doc
        # nobody but its tenant should read must not be stamped `public` just
        # because the uploader also happens to hold that label.
        acl_label = min(ctx.acl_labels) if ctx.tenant_id == "public" else _upload_label(ctx)
        nodes = [
            TextNode(
                text=c.text,
                metadata=stamp_access_meta(c.metadata, tenant_id=ctx.tenant_id, acl_label=acl_label),
            )
            for c in chunks
            if c.is_valid
        ]
        # insert_nodes gọi Gemini embed đồng bộ — chạy thẳng ở đây là đứng cả event loop.
        await asyncio.to_thread(index.insert_nodes, nodes)
        logger.info("Indexed {} nodes for '{}'", len(nodes), doc["title"][:40])

        # Rebuild retriever in background thread so BM25 corpus includes new nodes
        await asyncio.to_thread(_rebuild_retriever, r_module)

        return len(nodes)

    except Exception as exc:
        logger.error("Indexing failed: {}", exc)
        return 0


def _rebuild_retriever(r_module) -> None:
    """Reload all nodes from the vector store and rebuild the hybrid retriever."""
    try:
        from src.rag.vector_backend import get_backend
        from src.rag.retriever import build_hybrid_retriever
        from src.api.main import _load_nodes_from_backend
        from src.config import get_settings

        backend = get_backend()
        nodes = _load_nodes_from_backend(backend)
        set_indexed_documents(nodes)
        r_module._active_retriever = build_hybrid_retriever(
            r_module._active_index, nodes=nodes, rerank=get_settings().enable_reranker
        )
        logger.info("Retriever rebuilt with {} nodes after upload", len(nodes))
    except Exception as exc:
        logger.warning("Retriever rebuild failed (non-fatal): {}", exc)
