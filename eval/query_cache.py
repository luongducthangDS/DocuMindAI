"""
eval/query_cache.py — vector câu hỏi đã embed sẵn, để eval chạy được ở nơi không
đủ RAM cho model.

Vì sao tồn tại: model `AITeamVN/Vietnamese_Embedding` (bge-m3, 0.6B tham số,
~2.2GB trọng số) không load nổi trên máy dev còn ~2GB RAM trống, và đường
`hf_api` thì hỏng với model này (DEC-0006). Nhưng eval retrieval **không cần**
embed lại corpus: 1146 chunk trong ChromaDB đã có vector. Thứ duy nhất còn thiếu
là vector của mỗi **câu hỏi**.

Nên: embed câu hỏi MỘT LẦN ở nơi có RAM (Colab / Kaggle / VM) bằng
`scripts/build_query_embedding_cache.py`, commit file cache, và eval đọc từ đó.

An toàn: cache ghi kèm tên model, số chiều và SHA-256 của từng câu hỏi. Lệch model
hoặc lệch số chiều ⇒ ném lỗi ngay, không lặng lẽ trả vector của model khác — cùng
tinh thần với `EmbeddingModelMismatch` ở `src/rag/embedder.py`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CACHE_FORMAT = 1


class QueryEmbeddingCacheError(RuntimeError):
    """Cache không dùng được cho cấu hình hiện tại."""


def question_hash(question: str) -> str:
    """SHA-256 của câu hỏi đã chuẩn hoá khoảng trắng.

    Chuẩn hoá để một lần sửa khoảng trắng trong gold không làm hỏng cache, nhưng
    sửa CHỮ thì có — đúng ý: câu hỏi khác là câu hỏi khác.
    """
    normalised = " ".join((question or "").split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def build_payload(
    questions: list[str],
    embeddings: list[list[float]],
    model_name: str,
    extra: dict[str, Any] | None = None,
) -> dict:
    """Nội dung file cache. Dùng chung bởi script sinh cache và test."""
    if len(questions) != len(embeddings):
        raise ValueError(f"{len(questions)} câu hỏi nhưng {len(embeddings)} vector")
    dims = {len(e) for e in embeddings}
    if len(dims) != 1:
        raise ValueError(f"vector không cùng số chiều: {sorted(dims)}")

    return {
        "format": CACHE_FORMAT,
        "model": model_name,
        "dim": dims.pop(),
        **(extra or {}),
        "queries": [
            {"hash": question_hash(q), "question": q, "embedding": e}
            for q, e in zip(questions, embeddings)
        ],
    }


class QueryEmbeddingCache:
    """Tra vector của một câu hỏi từ file cache."""

    def __init__(self, payload: dict, expected_model: str | None = None):
        if payload.get("format") != CACHE_FORMAT:
            raise QueryEmbeddingCacheError(
                f"cache format {payload.get('format')} (cần {CACHE_FORMAT})"
            )
        self.model = str(payload.get("model", ""))
        self.dim = int(payload.get("dim", 0))
        self.meta = {k: v for k, v in payload.items() if k != "queries"}

        if expected_model and self.model != expected_model:
            raise QueryEmbeddingCacheError(
                f"cache sinh bằng '{self.model}' nhưng EMBEDDING_MODEL đang là "
                f"'{expected_model}'. Vector của hai model khác nhau vẫn cho ra "
                f"điểm cosine trông hợp lệ — dùng lẫn là sai im lặng."
            )

        self._by_hash: dict[str, list[float]] = {}
        for entry in payload.get("queries", []):
            self._by_hash[str(entry["hash"])] = list(entry["embedding"])
        if not self._by_hash:
            raise QueryEmbeddingCacheError("cache rỗng")

    @classmethod
    def load(cls, path: Path, expected_model: str | None = None) -> "QueryEmbeddingCache":
        if not path.exists():
            raise QueryEmbeddingCacheError(f"không thấy file cache: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload, expected_model=expected_model)

    def __len__(self) -> int:
        return len(self._by_hash)

    def get(self, question: str) -> list[float]:
        key = question_hash(question)
        try:
            return self._by_hash[key]
        except KeyError:
            raise QueryEmbeddingCacheError(
                f"câu hỏi không có trong cache (hash {key[:12]}…): "
                f"{question[:60]!r}. Sinh lại cache bằng "
                f"scripts/build_query_embedding_cache.py"
            ) from None

    def covers(self, questions: list[str]) -> list[str]:
        """Những câu hỏi cache còn thiếu — kiểm tra trước khi chạy cho đỡ hỏng giữa chừng."""
        return [q for q in questions if question_hash(q) not in self._by_hash]


def make_cached_embedder(cache: QueryEmbeddingCache):
    """Embedder kiểu LlamaIndex chỉ biết tra cache, không load model nào.

    Đường embed **document** cố tình ném lỗi: eval này chỉ truy vấn corpus đã
    index sẵn, nên nếu có gì đó đòi embed văn bản mới thì giả định "không cần
    model" đã sai và ta muốn biết ngay, không muốn một vector zero lặng lẽ.
    """
    from llama_index.core.embeddings import BaseEmbedding
    from pydantic import PrivateAttr

    class _CachedQueryEmbedding(BaseEmbedding):
        _cache: QueryEmbeddingCache = PrivateAttr()

        def __init__(self, cache: QueryEmbeddingCache, **kwargs):
            super().__init__(model_name=cache.model, **kwargs)
            self._cache = cache

        def _get_query_embedding(self, query: str) -> list[float]:
            return self._cache.get(query)

        def _get_text_embedding(self, text: str) -> list[float]:
            raise QueryEmbeddingCacheError(
                "embedder chạy bằng cache chỉ phục vụ câu hỏi, không embed được "
                "văn bản mới — corpus phải đã được index sẵn."
            )

        def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            return [self._get_text_embedding(t) for t in texts]

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._get_query_embedding(query)

        async def _aget_text_embedding(self, text: str) -> list[float]:
            return self._get_text_embedding(text)

        async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            return self._get_text_embeddings(texts)

    return _CachedQueryEmbedding(cache)
