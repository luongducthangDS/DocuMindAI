"""
FastAPI application entry point.
Configures: CORS, rate limiting, startup/shutdown lifespan, routers.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import src.logger  # noqa: F401 — initializes loguru
from src.api.routes import documents, health, query, reports
from src.config import get_settings


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize heavy resources once at startup."""
    settings = get_settings()
    settings.ensure_dirs()

    logger.info("Starting DocuMind AI — environment: {}", settings.environment)

    # Set LangSmith env vars
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        logger.info("LangSmith tracing enabled for project: {}", settings.langchain_project)

    # Initialize embedding + vector index
    await _init_rag()

    logger.info("DocuMind AI ready on {}:{}", settings.api_host, settings.api_port)
    yield

    logger.info("DocuMind AI shutting down")


async def _init_rag() -> None:
    """Load embedder and connect to ChromaDB. Non-fatal on failure."""
    import asyncio

    try:
        # Run all blocking I/O (model load, chromadb init) in a thread pool
        # to avoid blocking the asyncio event loop.
        await asyncio.to_thread(_init_rag_sync)
    except Exception as exc:
        logger.error("RAG init failed (system will run in degraded mode): {}", exc)


def _load_nodes_from_collection(collection) -> list:
    """Convert all ChromaDB documents into LlamaIndex TextNodes for BM25."""
    try:
        from llama_index.core.schema import TextNode

        result = collection.get(include=["documents", "metadatas"])
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        nodes = [
            TextNode(text=d, metadata=m or {})
            for d, m in zip(docs, metas)
            if d  # skip empty docs
        ]
        return nodes
    except Exception as exc:
        logger.warning("Could not load nodes from ChromaDB: {}", exc)
        return []


def _init_rag_sync() -> None:
    """Synchronous RAG initialization — called from thread pool."""
    from llama_index.core import Settings as LlamaSettings, VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import StorageContext
    import src.rag.retriever as r_module

    from src.rag.embedder import get_embedder, get_chroma_collection
    from src.rag.retriever import build_hybrid_retriever

    embedder = get_embedder()
    LlamaSettings.embed_model = embedder
    LlamaSettings.llm = None  # We call LLM directly via Groq SDK

    chroma_client, collection = get_chroma_collection()

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_ctx,
        embed_model=embedder,
    )

    # Load existing nodes from ChromaDB for BM25 corpus
    existing_nodes = _load_nodes_from_collection(collection)
    logger.info("Loaded {} nodes from ChromaDB for BM25 index", len(existing_nodes))

    # Expose to other modules
    r_module._active_index = index
    r_module._active_retriever = build_hybrid_retriever(index, nodes=existing_nodes, rerank=True)

    # Configure agent tools
    from src.agent.tools import configure_tools
    configure_tools(r_module._active_retriever, index)

    logger.info("RAG stack initialized: embedder + ChromaDB + hybrid retriever")


# ── App Instance ──────────────────────────────────────────────────────────────

settings = get_settings()

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.rate_limit_per_minute}/minute"],
)

app = FastAPI(
    title="DocuMind AI",
    description="RAG + Agentic AI for Vietnamese Legal Documents",
    version="1.0.0",
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,   # no cookies
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Global Exception Handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception: {} {}", request.url, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Please retry."},
    )


# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(query.router)
app.include_router(documents.router)
app.include_router(reports.router)
app.include_router(health.router)


@app.get("/", include_in_schema=False)
async def root():
    return {"service": "DocuMind AI", "version": "1.0.0", "docs": "/docs"}
