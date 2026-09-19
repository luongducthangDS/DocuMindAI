from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hf_env import use_local_hf_cache  # noqa: E402

# Trước mọi import HF. offline=False: ingest được phép tải model chưa cache.
use_local_hf_cache(offline=False, create=True)

import chromadb  # noqa: E402
from loguru import logger  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.rag.embedder import (  # noqa: E402
    STORE_META_DIM,
    STORE_META_MODEL,
    get_embedder,
    get_embedding_dim,
)
from src.ingestion.chunker import chunk_by_dieu
from src.ingestion.cleaner import clean_legal_text


# Model embedding đọc từ settings.embedding_model — xem src/rag/embedder.py.


def iter_json_docs(raw_dir: Path):
    for path in sorted(raw_dir.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipping unreadable JSON {}: {}", path, exc)
            continue
        content = clean_legal_text(doc.get("content", ""), doc_title=doc.get("title", ""))
        if len(content) < 100:
            continue
        doc["content"] = content
        yield doc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    chroma_path = str((settings.data_dir / "chroma_db").resolve())
    client = chromadb.PersistentClient(path=chroma_path)

    if args.reset:
        try:
            client.delete_collection(settings.chroma_collection)
            logger.warning("Deleted collection {}", settings.chroma_collection)
        except Exception as exc:
            logger.info("Collection reset skipped: {}", exc)

    # Dùng chung embedder với đường truy vấn — không tự dựng model ở script.
    model_name = settings.embedding_model
    embedder = get_embedder()
    embedding_dim = get_embedding_dim()

    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={
            "hnsw:space": "cosine",
            STORE_META_MODEL: model_name,
            STORE_META_DIM: embedding_dim,
        },
    )
    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    total = 0

    def flush() -> None:
        nonlocal ids, docs, metas, total
        if not docs:
            return
        embeddings = embedder.get_text_embedding_batch(docs)
        collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        total += len(docs)
        logger.info("Indexed {} chunks", total)
        ids, docs, metas = [], [], []

    for doc in iter_json_docs(args.raw_dir):
        chunks = chunk_by_dieu(doc["content"], doc)
        for idx, chunk in enumerate(chunks):
            if not chunk.is_valid:
                continue
            source_key = doc.get("so_hieu") or doc.get("url") or doc.get("title", "doc")
            ids.append(f"{source_key}:{idx}:{abs(hash(chunk.text))}")
            docs.append(chunk.text)
            metas.append(chunk.metadata)
            if len(docs) >= args.batch_size:
                flush()

    flush()
    logger.info("Done. Collection {} has {} chunks", collection.name, collection.count())


if __name__ == "__main__":
    main()
