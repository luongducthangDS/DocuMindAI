"""
Hybrid retriever: dense vector (paraphrase-multilingual-MiniLM-L12-v2) + sparse
BM25, fused via RRF. Optional cross-encoder reranker for precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

# Module-level singletons — set by api/main.py at startup
_active_retriever = None
_active_index = None

# True only once a cross-encoder reranker actually loaded and is wrapping the
# active retriever. settings.enable_reranker alone isn't enough to gate the
# generator's score threshold — the reranker can be *requested* but still fail
# to load at runtime (missing model cache, OOM, Render free tier, etc.), in
# which case chunk scores stay on the raw RRF fusion scale (~0.016) and a
# 0.05 threshold calibrated for cross-encoder scores silently discards every
# chunk. generator._effective_min_score() reads this flag, not the config.
_reranker_active = False

if TYPE_CHECKING:
    from llama_index.core import VectorStoreIndex
    from llama_index.core.schema import NodeWithScore


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict

    @property
    def citation_label(self) -> str:
        title = self.metadata.get("title", "Không rõ")
        dieu = self.metadata.get("dieu_header", "")
        so_hieu = self.metadata.get("so_hieu", "")
        label = title
        if so_hieu:
            label = f"{so_hieu} — {title}"
        if dieu:
            label += f"\n    {dieu[:100]}"
        url = self.metadata.get("source_url", "")
        if url:
            label += f"\n    {url}"
        return label


def build_hybrid_retriever(
    index: "VectorStoreIndex",
    nodes: list,
    top_k: int = 20,
    rerank: bool = True,
):
    """
    QueryFusionRetriever: combines dense + BM25 with Reciprocal Rank Fusion.
    Falls back to vector-only if BM25 init fails (e.g. empty corpus).

    top_k=20: candidate pool before reranking. For a small corpus (~100 chunks),
    pulling 20 candidates gives the cross-encoder enough material to find the best 8.
    RRF fusion similarity_top_k must equal top_k (not top_k//2) so that both
    the dense and sparse lists contribute their full candidate sets to fusion.
    """
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.retrievers.bm25 import BM25Retriever

    vector_retriever = index.as_retriever(similarity_top_k=top_k)

    try:
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=top_k,
        )
        retrievers = [vector_retriever, bm25_retriever]
        logger.info("Hybrid retriever ready (dense + BM25, top_k={})", top_k)
    except Exception as exc:
        logger.warning("BM25 init failed, using vector-only: {}", exc)
        retrievers = [vector_retriever]

    hybrid = QueryFusionRetriever(
        retrievers=retrievers,
        similarity_top_k=top_k,   # keep full pool — reranker will filter to top_n
        num_queries=1,             # no query expansion at retriever level
        mode="reciprocal_rerank",
        # Sync: retrieve_node gọi đường đồng bộ, nhưng use_async=True vẫn đẩy các
        # sub-retriever qua event loop dùng-một-lần. Với Qdrant điều đó nghĩa là
        # "Async client is not initialized" (chưa có aclient) rồi "Event loop is
        # closed" ở lượt sau — retriever im lặng rơi xuống fallback, câu trả lời
        # tụt chất lượng mà không báo lỗi. num_queries=1 nên chạy song song cũng
        # chẳng lợi gì: chỉ có dense + BM25, tuần tự là đủ.
        use_async=False,
    )

    if rerank:
        # top_n=8 balances precision vs recall; 5 was too aggressive for 20 candidates
        return _wrap_with_reranker(hybrid, top_n=8)
    # No cross-encoder to trim the pool (e.g. Render free tier): RRF fusion scores
    # are all clustered ~0.01-0.03 with no meaningful gap between relevant and
    # irrelevant hits, so a score threshold can't discriminate — truncate to RRF's
    # own rank order instead, matching the reranked top_n=8.
    return _TruncatedRetriever(hybrid, top_n=8)


def _wrap_with_reranker(base_retriever, top_n: int = 8):
    """
    Adds cross-encoder reranker on top of fusion retriever.
    Uses small but effective MiniLM model — runs locally, no API.
    """
    global _reranker_active
    # SentenceTransformerRerank không nhận cache_folder (chỉ có model/top_n/device/
    # trust_remote_code), nên ở đây buộc phải đi qua biến môi trường — khác với
    # embedder, nơi cache được truyền thẳng làm tham số.
    # Env được giữ nguyên sau khi load (khối finally cũ khôi phục lại HF_HOME trỏ về
    # G:\ đã bị gỡ): trả về một đường dẫn offline chỉ tạo ra lỗi khó hiểu cho lần
    # load model kế tiếp, không bảo vệ được gì.
    from src.hf_env import resolve_cached_model, use_local_hf_cache

    use_local_hf_cache(offline=True)

    try:
        from llama_index.core.postprocessor import SentenceTransformerRerank
        from src.config import get_settings

        model_name = get_settings().reranker_model
        # Truyền đường dẫn cache đã resolve thay cho repo id khi model có sẵn: nếu chỉ
        # đưa repo id, thư viện tra cache theo HF_HOME đã đóng băng lúc import (ổ G:\
        # offline trên máy dev) và coi như model chưa tải, dù nó nằm trong repo.
        model_ref = resolve_cached_model(model_name) or model_name
        reranker = SentenceTransformerRerank(
            model=model_ref,
            top_n=top_n,
        )
        logger.info("Cross-encoder reranker loaded: {} (top_n={})", model_name, top_n)
        _reranker_active = True
        return _RerankedRetriever(base_retriever, reranker)

    except Exception as exc:
        logger.warning("Reranker unavailable, skipping: {}", exc)
        _reranker_active = False
        # Same reasoning as the rerank=False branch in build_hybrid_retriever:
        # without a cross-encoder, RRF fusion scores (~0.01-0.03) have no
        # meaningful gap to threshold on, so truncate to rank order instead
        # of returning the full top_k=20 pool.
        return _TruncatedRetriever(base_retriever, top_n=top_n)


class _RerankedRetriever:
    """Thin wrapper: retrieve from hybrid, then rerank."""

    def __init__(self, base_retriever, reranker):
        self._base = base_retriever
        self._reranker = reranker

    async def aretrieve(self, query: str) -> list["NodeWithScore"]:
        nodes = await self._base.aretrieve(query)
        from llama_index.core.schema import QueryBundle

        return self._reranker.postprocess_nodes(nodes, QueryBundle(query_str=query))

    def retrieve(self, query: str) -> list["NodeWithScore"]:
        nodes = self._base.retrieve(query)
        from llama_index.core.schema import QueryBundle

        return self._reranker.postprocess_nodes(nodes, QueryBundle(query_str=query))


class _TruncatedRetriever:
    """Thin wrapper: keep only the top-N of the RRF-fused pool (no cross-encoder
    available to trim by relevance score, so rely on RRF's own rank order)."""

    def __init__(self, base_retriever, top_n: int = 8):
        self._base = base_retriever
        self._top_n = top_n

    async def aretrieve(self, query: str) -> list["NodeWithScore"]:
        nodes = await self._base.aretrieve(query)
        return nodes[: self._top_n]

    def retrieve(self, query: str) -> list["NodeWithScore"]:
        nodes = self._base.retrieve(query)
        return nodes[: self._top_n]


def nodes_to_chunks(nodes: list["NodeWithScore"]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            text=n.node.text or n.node.get_content(),
            score=float(n.score or 0),
            metadata={k: v for k, v in (n.node.metadata or {}).items()
                      if not k.startswith("_")},
        )
        for n in nodes
    ]


def retrieve_direct_chroma(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    """Fallback retrieval path that queries the vector store directly.

    Name kept as "_chroma" for backward compat with existing call sites
    (query.py, agent/graph.py) — but works against whichever provider
    VECTOR_STORE_PROVIDER points to (chroma or qdrant), via vector_backend.
    """
    try:
        from src.rag.embedder import get_embedder
        from src.rag.vector_backend import get_backend, count_chunks, direct_query

        backend = get_backend()
        count = count_chunks(backend)
        if count == 0:
            logger.error("Direct fallback ({}) found empty collection", backend.provider)
            return []

        embedder = get_embedder()
        query_embedding = embedder.get_query_embedding(query)
        results = direct_query(backend, query_embedding, top_k=top_k)

        chunks = [
            RetrievedChunk(text=r["text"], score=r["score"], metadata=r["metadata"])
            for r in results
        ]
        logger.info("Direct fallback ({}) returned {} chunks", backend.provider, len(chunks))
        return chunks
    except Exception as exc:
        logger.error("Direct fallback failed: {}", exc)
        return []
