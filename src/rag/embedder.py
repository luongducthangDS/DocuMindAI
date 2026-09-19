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

Nhà cung cấp DUY NHẤT là Gemini Embedding API (2026-09-19) — không còn nhánh
model local (sentence-transformers) lẫn HF Inference API. Reranker thì vẫn chạy
local, xem src/rag/retriever.py.

A/B lịch sử trên gold set lao động 28 câu, 1146 chunks (reports/embedding_ab.json,
2026-09-16), đo khi còn chạy model local:
  paraphrase-multilingual-MiniLM-L12-v2 (384d) → final@8 = 0.821
  AITeamVN/Vietnamese_Embedding        (1024d) → final@8 = 1.000
Chưa đo lại cho gemini-embedding-001.
"""

from __future__ import annotations

import time
from collections import deque
from functools import lru_cache
from typing import List

from loguru import logger
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings

# Khoá metadata ghi kèm collection để biết vector trong đó sinh từ model nào.
STORE_META_MODEL = "embedding_model"
STORE_META_DIM = "embedding_dim"


class EmbeddingModelMismatch(RuntimeError):
    """Vector store được index bằng model khác với model đang cấu hình.

    Đây là lỗi chặn đường, không phải cảnh báo: cosine giữa hai không gian vector
    khác nhau vẫn trả về số, vẫn xếp hạng được, nên hệ thống sẽ chạy "bình thường"
    và trả lời sai — dạng hỏng tệ nhất vì không ai thấy.
    """


# Free tier Gemini Embedding chặn ở HAI trần cùng lúc, đo thực tế 2026-09-18:
#   - số CONTENT, không phải số HTTP request (5 request × 20 content = 100 rồi 429),
#     nên gộp batch to không giảm mức tiêu thụ, chỉ giảm số lần đi mạng;
#   - số TOKEN mỗi phút — corpus này chunk ~900 ký tự nên 50 chunk/request đã chạm
#     trần token dù mới 50/100 content.
# Hai ngưỡng dưới đặt dưới trần thật để chừa biên, vì server đếm theo cửa sổ riêng
# của nó chứ không phải cửa sổ trượt của tiến trình này.
_GEMINI_CONTENTS_PER_MIN = 90
_GEMINI_TOKENS_PER_MIN = 20_000
# Tiếng Việt có dấu tốn token hơn tiếng Anh; 3 ký tự/token là ước lượng thận trọng.
_CHARS_PER_TOKEN = 3
# Chờ lâu hơn mức này nghĩa là quota NGÀY đã hết — báo lỗi thay vì ngủ tiếp.
_MAX_QUOTA_WAIT = 300


class _GeminiAPIEmbedding(BaseEmbedding):
    """Embedder gọi Gemini Embedding API thay vì nạp model vào RAM.

    Một điểm không được quên: Gemini sinh vector
    KHÁC NHAU cho cùng một chuỗi tuỳ task_type — retrieval_document lúc index,
    retrieval_query lúc hỏi. Dùng lẫn hai loại vẫn chạy, vẫn xếp hạng được, chỉ là
    kém đi âm thầm; nên hai đường text/query ở dưới tách bạch có chủ đích.
    """

    _keys: List[str] = PrivateAttr()
    _sent: List[deque] = PrivateAttr()  # mỗi key một bucket (timestamp, tokens) đã gửi
    _cooldown: dict = PrivateAttr()  # key -> thời điểm được dùng lại sau khi dính 429
    _strikes: dict = PrivateAttr()   # key -> số lần 429 liên tiếp

    def __init__(self, model_name: str, api_keys: List[str], **kwargs):
        # 50 là mức đã đo được với chunk thật của corpus (~900 ký tự/chunk): 100
        # chunk/request bị 429 ngay request đầu dù quota phút còn nguyên, 50 thì qua.
        kwargs.setdefault("embed_batch_size", 50)
        super().__init__(model_name=model_name, **kwargs)
        keys = [k for k in api_keys if k]
        if not keys:
            raise ValueError("Embedding qua Gemini cần GOOGLE_API_KEY trong .env")
        self._keys = keys
        self._sent = [deque() for _ in keys]
        self._cooldown = {}
        self._strikes = {}

    def _reserve_key(self, texts: List[str]) -> str:
        """Chọn key còn chỗ trong phút cho lô sắp gửi, chờ nếu mọi key đều đầy.

        Quota tính theo project chứ không theo key, nên nhiều key ở nhiều project
        cộng dồn được trần — mỗi key vì thế cần bucket riêng. Chủ động chờ thay vì
        để 429 bắn ra: re-embed cả corpus chạm trần liên tục, mà retry mù thì mỗi
        lần hỏng lại đốt thêm quota của phút sau.
        """
        # ponytail: bucket in-process. Nhiều worker song song thì cần bucket chia sẻ
        # — chưa có nhu cầu đó.
        tokens = sum(max(len(t) // _CHARS_PER_TOKEN, 1) for t in texts)
        per_text = tokens // max(len(texts), 1)
        while True:
            now = time.monotonic()
            minute_waits = []   # chạm trần phút — chờ vài chục giây là qua
            cooldowns = []      # đang bị phạt sau 429
            for key, bucket in zip(self._keys, self._sent):
                if self._cooldown.get(key, 0.0) > now:
                    cooldowns.append(self._cooldown[key] - now)
                    continue
                while bucket and now - bucket[0][0] >= 60:
                    bucket.popleft()
                if not bucket or (
                    len(bucket) + len(texts) <= _GEMINI_CONTENTS_PER_MIN
                    and sum(t for _, t in bucket) + tokens <= _GEMINI_TOKENS_PER_MIN
                ):
                    bucket.extend((now, per_text) for _ in texts)
                    return key
                minute_waits.append(60 - (now - bucket[0][0]) + 1)
            waits = minute_waits + cooldowns
            # Không key nào chỉ chờ trần phút, và cooldown ngắn nhất cũng dài hơn
            # _MAX_QUOTA_WAIT → đã cạn quota NGÀY. Ngủ tiếp là treo tiến trình cả
            # tiếng mà log không nói vì sao; báo lỗi để người chạy quyết định.
            if not waits or (not minute_waits and min(cooldowns) > _MAX_QUOTA_WAIT):
                raise RuntimeError(
                    f"Cả {len(self._keys)} key Gemini đều cạn quota (mỗi project 1000 embed/ngày). "
                    "Thêm key ở project khác vào GOOGLE_API_KEY_2/_3, bật billing, hoặc chờ quota reset."
                )
            logger.info("Gemini embed: mọi key đều đang chờ quota, chờ {:.0f}s ({} content / ~{} token)",
                        min(waits), len(texts), tokens)
            time.sleep(min(waits))

    # Lưới an toàn cho 429 còn lọt qua cả vòng xoay key (trần phút của mọi key).
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=15, max=70))
    def _embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Thử lần lượt từng key khả dụng trước khi chịu thua.

        Đổi key là việc tức thì, còn backoff của tenacity chờ hàng chục giây; để
        tenacity lo việc xoay key thì một tiến trình mới (bucket rỗng, chưa biết key
        nào đã cạn) đốt sạch lượt retry chỉ để đi qua các key chết.
        """
        last_exc: Exception | None = None
        for _ in range(len(self._keys)):
            key = self._reserve_key(texts)
            try:
                return self._call_api(key, texts, task_type)
            except Exception as exc:  # noqa: BLE001 — phân loại ngay bên dưới
                if not ("429" in str(exc) or "quota" in str(exc).lower()):
                    raise
                last_exc = exc
                strikes = self._strikes.get(key, 0) + 1
                self._strikes[key] = strikes
                # Cạn quota NGÀY thì mọi lần thử lại đều hỏng; sau 3 lần liên tiếp
                # coi như hết ngày và cho nghỉ dài, khỏi phí lượt của các key khác.
                self._cooldown[key] = time.monotonic() + (3600 if strikes >= 3 else 65)
                logger.warning("Key Gemini ...{} dính 429 (lần {}), nghỉ {}s",
                               key[-6:], strikes, 3600 if strikes >= 3 else 65)
        raise last_exc  # type: ignore[misc]

    def _call_api(self, key: str, texts: List[str], task_type: str) -> List[List[float]]:
        import google.generativeai as genai

        genai.configure(api_key=key)
        name = self.model_name if self.model_name.startswith("models/") else f"models/{self.model_name}"
        # API từ chối chuỗi rỗng; get_embedding_dim() lại dò chiều bằng đúng chuỗi đó.
        payload = [t if t.strip() else " " for t in texts]
        result = genai.embed_content(model=name, content=payload, task_type=task_type)
        self._strikes[key] = 0
        emb = result["embedding"]
        return emb if isinstance(emb[0], list) else [emb]

    # task_type tách bạch có chủ đích: Gemini sinh vector KHÁC NHAU cho cùng một
    # chuỗi tuỳ retrieval_query (lúc hỏi) hay retrieval_document (lúc index).
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([query], "retrieval_query")[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([text], "retrieval_document")[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts, "retrieval_document")

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)


@lru_cache(maxsize=1)
def get_embedder() -> "BaseEmbedding":
    """Embedder duy nhat cua du an: Gemini Embedding API.

    Khong nap model nao vao RAM — khong con nhanh local (sentence-transformers)
    lan HF Inference API (2026-09-19, quyet dinh cua Ted). Reranker van chay
    local, xem src/rag/retriever.py.
    """
    settings = get_settings()
    model_name = settings.embedding_model
    logger.info("Using Gemini Embedding API (no local model load): {}", model_name)
    keys = [settings.google_api_key, settings.google_api_key_2, settings.google_api_key_3]
    return _GeminiAPIEmbedding(model_name=model_name, api_keys=keys)


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


def get_async_qdrant_client():
    """AsyncQdrantClient cho LlamaIndex.

    QdrantVectorStore giữ client sync và async tách biệt: thiếu `aclient` thì mọi
    lời gọi aretrieve() ném ValueError("Async client is not initialized!") — retriever
    im lặng rơi xuống fallback, câu trả lời tụt chất lượng mà không báo lỗi.
    """
    from qdrant_client import AsyncQdrantClient

    settings = get_settings()
    if not settings.qdrant_url:
        raise RuntimeError(
            "VECTOR_STORE_PROVIDER=qdrant nhưng QDRANT_URL chưa được set trong .env"
        )
    return AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)


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
