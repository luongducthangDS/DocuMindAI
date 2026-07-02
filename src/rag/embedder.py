"""
Embedding model for DocuMind AI.

CURRENT MODEL: paraphrase-multilingual-MiniLM-L12-v2 (384-dim)
  - Fast, lightweight, works offline
  - The bundled ChromaDB corpus is already indexed with this model — do NOT change
    the model without re-indexing (dimensions must match)

UPGRADE PATH to BGE-M3 (better Vietnamese recall, context_recall +5-8 pp est.):
  1. pip install FlagEmbedding
  2. Re-index corpus: python scripts/rebuild_index.py --model BAAI/bge-m3
  3. Set EMBEDDING_MODEL=BAAI/bge-m3 in .env
  Note: bge-m3 is 570M params — needs 2GB+ RAM and ~60s cold start on CPU.

RAGAS benchmark (MiniLM baseline, 2025-05):
  faithfulness=0.8714 | answer_relevancy=0.8231 | context_recall=0.7683
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, List

from loguru import logger
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings

if TYPE_CHECKING:
    from huggingface_hub import InferenceClient

_INDEXED_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class _HFInferenceAPIEmbedding(BaseEmbedding):
    """LlamaIndex embedder that calls HuggingFace's Inference API instead of loading
    the model in-process. Same model, same 384-dim pooled vectors (verified to match
    the local SentenceTransformer output byte-for-byte via cosine similarity) — used
    so torch/transformers/model weights (~700MB combined) never load into RAM on
    memory-constrained hosts like Render's free 512MB tier. Importing this class
    does NOT import llama_index.embeddings.huggingface (torch-based), only
    llama_index.core (no torch dependency).
    """

    _client: "InferenceClient" = PrivateAttr()

    def __init__(self, model_name: str, hf_token: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        from huggingface_hub import InferenceClient

        self._client = InferenceClient(token=hf_token or None)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        result = self._client.feature_extraction(texts, model=self.model_name)
        return result.tolist() if hasattr(result, "tolist") else result

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed_batch([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed_batch(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)


@lru_cache(maxsize=1)
def get_embedder() -> "BaseEmbedding":
    """
    Returns a cached LlamaIndex-compatible embedder.

    embedding_provider="local" (default): loads the model in-process via
    sentence-transformers/torch — offline, fast, ~700MB RAM.
    embedding_provider="hf_api": calls HuggingFace's Inference API instead —
    no local model load, for RAM-constrained hosts (see _HFInferenceAPIEmbedding).
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

    if settings.embedding_provider == "hf_api":
        logger.info("Using HF Inference API for embeddings (no local model load): {}", model_name)
        return _HFInferenceAPIEmbedding(model_name=model_name, hf_token=settings.hf_token)

    logger.info("Loading embedding model: {}", model_name)

    # low_cpu_mem_usage: load weights tensor-by-tensor instead of all at once,
    # halving peak RAM.  Critical on machines with limited pagefile (Windows).
    _model_kwargs = {"low_cpu_mem_usage": True}

    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        embedder = HuggingFaceEmbedding(
            model_name=model_name,
            max_length=512,
            trust_remote_code=False,  # security: never trust remote code by default
            model_kwargs=_model_kwargs,
        )
        logger.info("Embedder ready: {}", model_name)
        return embedder

    except Exception as exc:
        logger.error("Failed to load {} — retrying with local_files_only=True: {}", model_name, exc)
        # Retry the SAME model from local disk only (no HF download).
        # This avoids the hf_xet / G:\ issue on unmounted drives while keeping
        # the correct embedding dimensions for the indexed ChromaDB corpus.
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(
            model_name=model_name,
            max_length=512,
            trust_remote_code=False,
            model_kwargs={**_model_kwargs, "local_files_only": True},
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


_QDRANT_VECTOR_SIZE = 384  # must match the embedding model's output dim (MiniLM-L12-v2)


def get_qdrant_client_and_collection() -> tuple:
    """
    Return (QdrantClient, collection_name). Creates the collection if it doesn't
    exist yet (cosine distance, 384-dim to match the embedding model).

    Requires QDRANT_URL in .env (Qdrant Cloud cluster URL) + QDRANT_API_KEY.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    settings = get_settings()
    if not settings.qdrant_url:
        raise RuntimeError(
            "VECTOR_STORE_PROVIDER=qdrant nhưng QDRANT_URL chưa được set trong .env"
        )

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    collection_name = settings.qdrant_collection

    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=_QDRANT_VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("Created new Qdrant collection: {}", collection_name)

    logger.info("Qdrant ready: {} @ {}", collection_name, settings.qdrant_url)
    return client, collection_name
