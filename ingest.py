"""
CLI ingestion pipeline.
Usage:
  python ingest.py --source hf --max-docs 500
  python ingest.py --source crawl --doc-types luat nghi-dinh --max-pages 10
  python ingest.py --source json --dir data/raw
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from loguru import logger

import src.logger  # noqa: F401
from src.config import get_settings
from src.ingestion.chunker import chunk_by_dieu
from src.ingestion.crawler import VbplCrawler
from src.ingestion.loader import iter_hf_dataset, iter_raw_json_dir
from src.rag.embedder import get_chroma_collection, get_embedder


def reset_collection() -> None:
    """Delete and recreate the configured Chroma collection."""
    settings = get_settings()
    client, _ = get_chroma_collection()
    try:
        client.delete_collection(settings.chroma_collection)
        logger.warning("Deleted existing Chroma collection: {}", settings.chroma_collection)
    except Exception as exc:
        logger.info("Collection reset skipped or not needed: {}", exc)


def ingest_documents(docs, embedder, collection, batch_size: int = 32) -> int:
    """Embed and store documents in ChromaDB. Returns total chunks indexed."""
    from llama_index.core import Settings as LlamaSettings, VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import StorageContext
    from llama_index.core.schema import TextNode

    LlamaSettings.embed_model = embedder
    LlamaSettings.llm = None

    chroma_client, chroma_col = get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=chroma_col)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_ctx,
        embed_model=embedder,
    )

    total_chunks = 0
    batch: list[TextNode] = []

    for doc in docs:
        chunks = chunk_by_dieu(doc["content"], doc)
        for chunk in chunks:
            if not chunk.is_valid:
                continue
            node = TextNode(text=chunk.text, metadata=chunk.metadata)
            batch.append(node)

            if len(batch) >= batch_size:
                index.insert_nodes(batch)
                total_chunks += len(batch)
                logger.info("Indexed batch: {} total chunks so far", total_chunks)
                batch = []

    if batch:
        index.insert_nodes(batch)
        total_chunks += len(batch)

    logger.info("Ingestion complete: {} chunks indexed", total_chunks)
    return total_chunks


def _crawl_seed_documents(seed_file: Path, output_dir: Path):
    """Crawl a curated list of important official documents."""
    from src.ingestion.congbao_crawler import CongbaoCrawler, _safe_filename

    if not seed_file.exists():
        raise FileNotFoundError(f"Seed file not found: {seed_file}")

    seeds = json.loads(seed_file.read_text(encoding="utf-8"))
    crawler = CongbaoCrawler(output_dir=output_dir)
    docs = []
    for item in seeds:
        doc = crawler.fetch_download_page(item["url"], metadata=item)
        if not doc:
            logger.warning("Seed document failed: {}", item.get("title") or item["url"])
            continue

        out_path = output_dir / (_safe_filename(doc["title"]) + ".json")
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved seed document: {} ({} chars)", doc["title"], len(doc["content"]))
        docs.append(doc)
    return iter(docs)


def main() -> None:
    parser = argparse.ArgumentParser(description="DocuMind AI — Ingestion Pipeline")
    parser.add_argument("--source", choices=["hf", "crawl", "json", "congbao", "seed"], default="hf")
    parser.add_argument("--max-docs", type=int, default=300)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--max-issues", type=int, default=20,
                        help="Max gazette issues to crawl (congbao source only)")
    parser.add_argument("--doc-types", nargs="+",
                        default=["luat", "bo_luat", "nghi_dinh", "thong_tu",
                                 "quyet_dinh", "nghi_quyet", "phap_lenh"],
                        help="Document types to keep (congbao source)")
    parser.add_argument("--dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--seed-file", type=Path, default=Path("data/seed_documents.json"))
    parser.add_argument("--reset", action="store_true",
                        help="Delete the configured Chroma collection before ingesting")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_dirs()

    logger.info("Starting ingestion | source={} | max_docs={}", args.source, args.max_docs)
    t0 = time.time()

    if args.reset:
        reset_collection()

    # Load embedding model
    embedder = get_embedder()
    _, collection = get_chroma_collection()

    # Select document source
    if args.source == "hf":
        docs = iter_hf_dataset(max_docs=args.max_docs)
    elif args.source == "crawl":
        crawler = VbplCrawler()
        crawler.crawl(
            doc_types=args.doc_types,
            max_pages=args.max_pages,
            max_docs=args.max_docs,
        )
        docs = iter_raw_json_dir(settings.data_dir / "raw")
    elif args.source == "congbao":
        from src.ingestion.congbao_crawler import CongbaoCrawler

        crawler = CongbaoCrawler(output_dir=settings.data_dir / "raw")
        count = crawler.crawl(
            max_issues=args.max_issues,
            doc_types=args.doc_types,
            max_docs=args.max_docs,
        )
        logger.info("Crawled {} documents from congbao.chinhphu.vn", count)
        docs = iter_raw_json_dir(settings.data_dir / "raw")
    elif args.source == "seed":
        docs = _crawl_seed_documents(args.seed_file, settings.data_dir / "raw")
    elif args.source == "json":
        docs = iter_raw_json_dir(args.dir)
    else:
        logger.error("Unknown source: {}", args.source)
        sys.exit(1)

    total = ingest_documents(docs, embedder, collection, batch_size=args.batch_size)

    elapsed = time.time() - t0
    logger.info(
        "Ingestion finished: {} chunks | {:.1f}s | {:.1f} chunks/s",
        total,
        elapsed,
        total / elapsed if elapsed > 0 else 0,
    )


if __name__ == "__main__":
    main()
