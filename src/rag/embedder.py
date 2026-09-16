"""
Embedding model for DocuMind AI.

Model dùng cho cả index lẫn query đọc từ MỘT nguồn sự thật: settings.embedding_model
(.env → EMBEDDING_MODEL). Không hard-code ở đây, không hard-code trong scripts.

Ràng buộc thật cần giữ là "vector trong store phải sinh ra từ đúng model này" —
ràng buộc đó được thực thi bằng metadata ghi kèm collection (xem
assert_store_matches_model), chứ không phải bằng cách ghi đè config của người dùng.
Lệch model ⇒ báo lỗi to và dừng, vì kết quả retrieval khi đó là rác im lặng.

Đổi model:
  1. EMBEDDING_MODEL=<model mới> trong .env
  2. python scripts/reembed_corpus.py --yes     (re-embed tại chỗ, giữ nguyên chunk)
  3. pytest -q && python eval/temporal_eval.py  (xác nhận không vỡ)

A/B trên gold set lao động 28 câu, 1146 chunks (reports/embedding_ab.json, 2026-09-16):
  paraphrase-multilingual-MiniLM-L12-v2 (384d) → final@8 = 0.821
  AITeamVN/Vietnamese_Embedding        (1024d) → final@8 = 1.000
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, List

from loguru import logger
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.hf_env import HF_CACHE_DIR, use_local_hf_cache

if TYPE_CHECKING:
    from huggingface_hub import InferenceClient

# Khoá metadata ghi kèm collection để biết vector trong đó sinh từ model nào.
STORE_META_MODEL = "embedding_model"
STORE_META_DIM = "embedding_dim"


class EmbeddingModelMismatch(RuntimeError):
    """Vector store được index bằng model khác với model đang cấu hình.

    Đây là lỗi chặn đường, không phải cảnh báo: cosine giữa hai không gian vector
    khác nhau vẫn trả về số, vẫn xếp hạng được, nên hệ thống sẽ chạy "bình thường"
    và trả lời sai — dạng hỏng tệ nhất vì không ai thấy.
    """


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
    model_name = settings.embedding_model

    if settings.embedding_provider == "hf_api":
        logger.info("Using HF Inference API for embeddings (no local model load): {}", model_name)
        return _HFInferenceAPIEmbedding(model_name=model_name, hf_token=settings.hf_token)

    logger.info("Loading embedding model: {}", model_name)

    # low_cpu_mem_usage: load weights tensor-by-tensor instead of all at once,
    # halving peak RAM.  Critical on machines with limited pagefile (Windows).
    _model_kwargs = {"low_cpu_mem_usage": True}

    # Truyền cache_folder TƯỜNG MINH thay vì trông vào biến môi trường:
    # huggingface_hub đọc HF_HOME/HF_HUB_CACHE đúng một lần lúc import và đóng băng
    # giá trị đó. Module này import llama_index (kéo theo huggingface_hub) ở đầu file,
    # nên mọi thao tác os.environ bên trong hàm đều đã muộn — cache vẫn trỏ về
    # HF_HOME của máy (G:\, thường offline) và model nằm sẵn trong repo bị coi như
    # không tồn tại. Tham số thì không có vấn đề thời điểm đó.
    cache_folder = str(HF_CACHE_DIR) if HF_CACHE_DIR.exists() else None
    if cache_folder:
        use_local_hf_cache(offline=True)  # cho các thư viện đọc env muộn (sentence_transformers)

    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    embedder = HuggingFaceEmbedding(
        model_name=model_name,
        max_length=512,
        trust_remote_code=False,  # security: never trust remote code by default
        cache_folder=cache_folder,
        model_kwargs=_model_kwargs,
    )
    logger.info("Embedder ready: {} (cache: {})", model_name, cache_folder or "mặc định HF")
    return embedder


@lru_cache(maxsize=1)
def get_embedding_dim() -> int:
    """Chiều vector của model đang cấu hình, hỏi thẳng model thay vì hard-code.

    Một lần embed chuỗi rỗng là đủ, và rẻ so với việc để hằng số lệch âm thầm.
    """
    dim = len(get_embedder().get_query_embedding(""))
    logger.info("Embedding dim = {} ({})", dim, get_settings().embedding_model)
    return dim


def store_identity() -> dict:
    """Metadata nhận dạng để ghi kèm collection lúc index."""
    return {
        STORE_META_MODEL: get_settings().embedding_model,
        STORE_META_DIM: get_embedding_dim(),
    }


def assert_store_matches_model(collection_metadata: dict | None, *, where: str = "vector store") -> None:
    """Dừng ngay nếu store được index bằng model khác model đang cấu hình.

    Collection cũ (index trước khi có metadata này) chỉ cảnh báo — không thể
    khẳng định nó sai, nhưng cũng không thể khẳng định nó đúng.
    """
    settings = get_settings()
    meta = collection_metadata or {}
    indexed_model = meta.get(STORE_META_MODEL)

    if not indexed_model:
        logger.warning(
            "{} không ghi {} — không kiểm chứng được nó đã index bằng model nào. "
            "Chạy scripts/reembed_corpus.py để gắn nhãn.",
            where, STORE_META_MODEL,
        )
        return

    if indexed_model != settings.embedding_model:
        raise EmbeddingModelMismatch(
            f"{where} được index bằng '{indexed_model}' nhưng EMBEDDING_MODEL đang là "
            f"'{settings.embedding_model}'. Truy vấn sẽ trả kết quả sai một cách im lặng. "
            f"Cách xử lý: đặt lại EMBEDDING_MODEL='{indexed_model}', hoặc re-embed corpus "
            f"bằng `python scripts/reembed_corpus.py --yes`."
        )


def get_chroma_collection(*, verify: bool = True):
    """
    Return ChromaDB collection.
    Tries HTTP server first (production), falls back to local PersistentClient (dev).
    Uses heartbeat probe before attempting collection ops to fail fast.

    verify=True: đối chiếu nhãn model của collection với EMBEDDING_MODEL và ném
    EmbeddingModelMismatch nếu lệch. Chỉ so tên model — không nạp model, nên rẻ.
    verify=False dành cho script đang chủ động ghi lại corpus bằng model khác
    (scripts/reembed_corpus.py).
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
            if verify:
                assert_store_matches_model(
                    collection.metadata, where=f"Chroma collection '{settings.chroma_collection}'"
                )
            logger.info("ChromaDB HTTP server ready: {}", settings.chroma_collection)
            return client, collection
        except EmbeddingModelMismatch:
            raise  # lỗi cấu hình, không phải lỗi kết nối — không được nuốt rồi fallback
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
    if verify:
        assert_store_matches_model(
            collection.metadata, where=f"Chroma collection '{settings.chroma_collection}'"
        )
    logger.info("ChromaDB local persistent ready: {} @ {}", settings.chroma_collection, chroma_path)
    return local_client, collection


def get_qdrant_client_and_collection() -> tuple:
    """
    Return (QdrantClient, collection_name). Creates the collection if it doesn't
    exist yet (cosine distance, chiều vector hỏi thẳng model đang cấu hình).

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
            vectors_config=VectorParams(size=get_embedding_dim(), distance=Distance.COSINE),
        )
        logger.info("Created new Qdrant collection: {}", collection_name)
    else:
        # Collection sẵn có: chiều vector của nó là bằng chứng khách quan về model
        # đã index — lệch thì mọi truy vấn về sau là rác, nên chặn ngay tại đây.
        info = client.get_collection(collection_name)
        vectors = info.config.params.vectors
        indexed_dim = getattr(vectors, "size", None)
        wanted = get_embedding_dim()
        if indexed_dim is not None and indexed_dim != wanted:
            raise EmbeddingModelMismatch(
                f"Qdrant collection '{collection_name}' có vector {indexed_dim} chiều nhưng "
                f"EMBEDDING_MODEL='{settings.embedding_model}' sinh vector {wanted} chiều. "
                f"Chạy lại migrate/re-embed trước khi truy vấn."
            )

    logger.info("Qdrant ready: {} @ {}", collection_name, settings.qdrant_url)
    return client, collection_name
