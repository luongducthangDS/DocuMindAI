"""
BGE-M3 embedder — best multilingual/Vietnamese embedding model.
Wraps FlagEmbedding for colbert+dense hybrid (optional) or HuggingFace standard.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from loguru import logger

from src.config import get_settings

if TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding

_INDEXED_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def get_embedder() -> "BaseEmbedding":
    """
    Returns a cached LlamaIndex-compatible embedder.
    Uses HuggingFace local model — no API call, no cost, runs offline.
    """
    settings = get_settings()
    # The bundled Chroma corpus is indexed with this 384-dim model. A Railway
    # variable from an older deploy can otherwise silently break retrieval.
    model_name = _INDEXED_EMBEDDING_MODEL
    if settings.embedding_model and settings.embedding_model != model_name:
        logger.warning(
            "Ignoring EMBEDDING_MODEL={} because bundled ChromaDB was indexed with {}",
            settings.embedding_model,
            model_name,
        )

    logger.info("Loading embedding model: {}", model_name)

    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        embedder = HuggingFaceEmbedding(
            model_name=model_name,
            max_length=512,
            trust_remote_code=False,  # security: never trust remote code by default
        )
        logger.info("Embedder ready: {}", model_name)
        return embedder

    except Exception as exc:
        logger.error("Failed to load {} — falling back to smaller model: {}", model_name, exc)

        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            max_length=256,
            trust_remote_code=False,
        )


def get_chroma_collection():
    """
    Return ChromaDB collection.
    Tries HTTP server first (production), falls back to local PersistentClient (dev).
    Uses heartbeat probe before attempting collection ops to fail fast.
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    settings = get_settings()
    chroma_settings = ChromaSettings(anonymized_telemetry=False)

    # Try HTTP server only when explicitly configured.
    if settings.chroma_host:
        try:
            client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
                settings=chroma_settings,
            )
            client.heartbeat()  # fast connectivity check before heavy ops
            collection = client.get_or_create_collection(
                name=settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaDB HTTP server ready: {}", settings.chroma_collection)
            return client, collection
        except Exception as exc:
            logger.warning(
                "HTTP ChromaDB unavailable ({}), using local PersistentClient",
                str(exc)[:60],
            )

    # Fallback: local persistent (no server needed).
    # Do NOT pass Settings here — let chromadb use its own defaults for local mode.
    chroma_path = str((settings.data_dir / "chroma_db").resolve())
    local_client = chromadb.PersistentClient(path=chroma_path)
    collection = local_client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("ChromaDB local persistent ready: {} @ {}", settings.chroma_collection, chroma_path)
    return local_client, collection
