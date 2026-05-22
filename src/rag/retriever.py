"""
Hybrid retriever: dense vector (bge-m3) + sparse BM25, fused via RRF.
Optional cross-encoder reranker for precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

# Module-level singletons — set by api/main.py at startup
_active_retriever = None
_active_index = None

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
    top_k: int = 10,
    rerank: bool = True,
):
    """
    QueryFusionRetriever: combines dense + BM25 with Reciprocal Rank Fusion.
    Falls back to vector-only if BM25 init fails (e.g. empty corpus).
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
        similarity_top_k=top_k // 2 or 5,
        num_queries=1,           # no query expansion at retriever level
        mode="reciprocal_rerank",
        use_async=True,
    )

    if rerank:
        return _wrap_with_reranker(hybrid, top_n=5)
    return hybrid


def _wrap_with_reranker(base_retriever, top_n: int = 5):
    """
    Adds cross-encoder reranker on top of fusion retriever.
    Uses small but effective MiniLM model — runs locally, no API.
    """
    try:
        from llama_index.core.postprocessor import SentenceTransformerRerank

        reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_n=top_n,
        )
        logger.info("Cross-encoder reranker loaded (top_n={})", top_n)

        # Compose: base retriever + reranker postprocessor
        from llama_index.core.query_engine import RetrieverQueryEngine

        return _RerankedRetriever(base_retriever, reranker)

    except Exception as exc:
        logger.warning("Reranker unavailable, skipping: {}", exc)
        return base_retriever


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
