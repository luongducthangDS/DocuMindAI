"""
Hybrid Retriever: Dense + BM25 + RRF + Reranker

Giải thích kiến trúc trong interview:
- Dense: semantic search qua embedding (hiểu "nghỉ đẻ" = "chế độ thai sản")
- BM25: exact keyword match (tìm đúng "45/2019/QH14", "Điều 54")
- RRF: kết hợp 2 nhánh không cần tune trọng số
- Reranker: cross-encoder chọn top-8 cuối cùng

Production version: src/rag/retriever.py (~500 lines)
"""

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Kết hợp kết quả Dense và BM25 bằng Reciprocal Rank Fusion.

    Công thức: RRF_score(d) = Σ 1 / (k + rank(d))

    Tại sao RRF thay vì weighted sum?
    - Dense score và BM25 score ở 2 scale hoàn toàn khác nhau
    - RRF chỉ dùng rank (thứ hạng), không dùng score tuyệt đối
    - Không cần tune hyperparameter, robust với nhiều loại query
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    # Dense contribution: rank 1 → 1/(60+1) ≈ 0.016, rank 20 → 1/(60+20) ≈ 0.012
    for rank, chunk in enumerate(dense_results, start=1):
        key = chunk.metadata.get("chunk_id", chunk.text[:50])
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
        chunk_map[key] = chunk

    # BM25 contribution: cùng công thức, cộng vào score tích lũy
    for rank, chunk in enumerate(bm25_results, start=1):
        key = chunk.metadata.get("chunk_id", chunk.text[:50])
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
        chunk_map[key] = chunk

    # Sort by fused score, gán lại score
    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    result = []
    for key in sorted_keys:
        chunk = chunk_map[key]
        chunk.score = scores[key]
        result.append(chunk)
    return result


def build_hybrid_retriever(index, nodes, top_k: int = 20, top_n: int = 8):
    """
    Xây dựng retriever hoàn chỉnh:
    1. Dense retriever từ vector index
    2. BM25 retriever từ node list
    3. Fusion qua RRF
    4. Reranker (nếu có) → top_n final chunks

    top_k=20: candidate pool cho reranker
    top_n=8: final chunks vào LLM context
    """
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.retrievers.bm25 import BM25Retriever

    vector_retriever = index.as_retriever(similarity_top_k=top_k)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)

    # QueryFusionRetriever dùng RRF internally
    hybrid = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=top_k,
        num_queries=1,       # không expand query
        mode="reciprocal_rerank",
        use_async=False,
    )

    # Reranker cross-encoder: chấm lại từng (query, chunk) pair
    # Precision cao hơn nhiều so với cosine/BM25 score
    return _add_reranker(hybrid, top_n=top_n)


def _add_reranker(base_retriever, top_n: int = 8):
    """
    Cross-encoder reranker: chấm lại từng (query, chunk) pair.

    Khác với bi-encoder (embedding): cross-encoder nhìn query và chunk cùng lúc
    → hiểu ngữ cảnh tốt hơn, precision cao hơn.
    Trade-off: chậm hơn → chỉ dùng cho top-20, không phải toàn bộ corpus.
    """
    try:
        from llama_index.core.postprocessor import SentenceTransformerRerank
        reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-v2-m3",  # multilingual, handles Vietnamese
            top_n=top_n,
        )
        print(f"✓ Reranker loaded (top_n={top_n})")

        class RerankedRetriever:
            def retrieve(self, query: str):
                from llama_index.core.schema import QueryBundle
                nodes = base_retriever.retrieve(query)
                return reranker.postprocess_nodes(nodes, QueryBundle(query_str=query))

        return RerankedRetriever()

    except Exception as e:
        print(f"⚠ Reranker unavailable ({e}), using RRF rank order")
        return base_retriever


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo_rrf():
    """Minh họa RRF bằng dữ liệu giả để giải thích trong interview."""

    # Dense tìm được: chunk B ở rank 1 (semantic match), chunk A ở rank 2
    dense = [
        RetrievedChunk("Điều 54. Điều kiện hưởng lương hưu: đủ 15 năm đóng BHXH...",
                       score=0.92, metadata={"chunk_id": "B"}),
        RetrievedChunk("Điều 73. Điều kiện hưởng lương hưu (tự nguyện): đủ 15 năm...",
                       score=0.88, metadata={"chunk_id": "A"}),
        RetrievedChunk("Điều 32. Mức đóng bảo hiểm thất nghiệp...",
                       score=0.71, metadata={"chunk_id": "C"}),
    ]

    # BM25 tìm được: chunk A ở rank 1 (keyword "lương hưu tự nguyện"), chunk B ở rank 3
    bm25 = [
        RetrievedChunk("Điều 73. Điều kiện hưởng lương hưu (tự nguyện): đủ 15 năm...",
                       score=12.4, metadata={"chunk_id": "A"}),
        RetrievedChunk("Điều 98. Mức hưởng lương hưu hàng tháng (tự nguyện)...",
                       score=10.1, metadata={"chunk_id": "D"}),
        RetrievedChunk("Điều 54. Điều kiện hưởng lương hưu: đủ 15 năm đóng BHXH...",
                       score=8.2, metadata={"chunk_id": "B"}),
    ]

    print("=== Dense results ===")
    for i, c in enumerate(dense, 1):
        print(f"  Rank {i}: [{c.metadata['chunk_id']}] score={c.score:.2f} | {c.text[:60]}...")

    print("\n=== BM25 results ===")
    for i, c in enumerate(bm25, 1):
        print(f"  Rank {i}: [{c.metadata['chunk_id']}] score={c.score:.1f} | {c.text[:60]}...")

    fused = reciprocal_rank_fusion(dense, bm25)
    print("\n=== RRF Fused results ===")
    for i, c in enumerate(fused, 1):
        print(f"  Rank {i}: [{c.metadata['chunk_id']}] rrf_score={c.score:.4f} | {c.text[:60]}...")

    print("\n--- Nhận xét ---")
    print("Chunk A: Dense rank 2 + BM25 rank 1 → RRF score cao nhất → rank 1 sau fusion")
    print("Chunk B: Dense rank 1 + BM25 rank 3 → RRF score thứ 2")
    print("Điều này đúng: câu hỏi về 'lương hưu tự nguyện' thì Điều 73 (A) phù hợp hơn Điều 54 (B)")


if __name__ == "__main__":
    demo_rrf()
