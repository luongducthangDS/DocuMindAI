"""
scripts/ingest_uneti.py — ingest pre-chunked UNETI university documents into ChromaDB.

Usage:
    python scripts/ingest_uneti.py --source-dir "D:/Projects for CV/chatbot_uneti_final/source/pdf/md" --reset

Flags:
    --source-dir   Directory containing *_chunks.json files (default: path above)
    --reset        Clear existing collection before ingesting (recommended for full replacement)
    --batch-size   Embedding batch size (default: 16)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import get_settings

INDEXED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Human-readable title map keyed by regulation code
_TITLE_MAP = {
    "QĐ-740": "Quy định về học, kiểm tra và chuẩn đầu ra ngoại ngữ - UNETI",
    "QĐ-747": "Quy chế đánh giá điểm rèn luyện của sinh viên - UNETI",
    "QĐ-748": "Quy định về học bổng khuyến khích học tập - UNETI",
    "QĐ-670": "Quy định về chuẩn đầu ra tin học - UNETI",
}


def _doc_title(regulation: str, filename: str) -> str:
    if regulation in _TITLE_MAP:
        return _TITLE_MAP[regulation]
    # Derive from filename: strip chunk suffix, replace dashes/underscores
    stem = Path(filename).stem.replace("-", " ").replace("_", " ")
    return stem[:80]


def _section_header(metadata: dict) -> str:
    """Pick the most specific header available."""
    for key in ("Header 3", "Header 2", "Header 1"):
        val = metadata.get(key, "").strip()
        if val and len(val) > 5:
            return val[:120]
    return ""


def iter_chunks(source_dir: Path):
    """Yield (chunk_id, text, chroma_metadata) from all *_chunks.json files."""
    files = sorted(source_dir.glob("*_chunks.json"))
    if not files:
        raise FileNotFoundError(f"No *_chunks.json files found in {source_dir}")

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.info("Reading {} — {} chunks", path.name, len(data))

        for item in data:
            meta = item.get("metadata", {})
            content = item.get("content", "").strip()
            if not content or len(content) < 30:
                continue

            regulation = meta.get("regulation", "")
            title = _doc_title(regulation, meta.get("source", path.stem))
            dieu_header = _section_header(meta)

            yield (
                item["chunk_id"],
                content,
                {
                    "title": title,
                    "so_hieu": regulation,
                    "dieu_header": dieu_header,
                    "source_url": "",
                    "doc_type": "Quyết định",
                    "ngay_ban_hanh": "2018-10-20",
                    "institution": "UNETI",
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest UNETI chunks into DocuMind ChromaDB")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("D:/Projects for CV/chatbot_uneti_final/source/pdf/md"),
    )
    parser.add_argument("--reset", action="store_true", help="Clear existing collection first")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not args.source_dir.exists():
        logger.error("Source directory not found: {}", args.source_dir)
        sys.exit(1)

    settings = get_settings()
    chroma_path = str((settings.data_dir / "chroma_db").resolve())
    client = chromadb.PersistentClient(path=chroma_path)

    if args.reset:
        try:
            client.delete_collection(settings.chroma_collection)
            logger.warning("Deleted existing collection '{}'", settings.chroma_collection)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )

    import os
    # Force local HF cache — system HF_HOME may point to Google Drive (G:\) which is offline
    local_hf = Path(__file__).resolve().parents[1] / "data" / "hf_cache"
    local_hf.mkdir(parents=True, exist_ok=True)
    for _k in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
        os.environ[_k] = str(local_hf)

    model = SentenceTransformer(INDEXED_MODEL)
    logger.info("Embedding model loaded: {}", INDEXED_MODEL)

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    total = 0

    def flush() -> None:
        nonlocal ids, docs, metas, total
        if not docs:
            return
        embeddings = model.encode(docs, batch_size=8, normalize_embeddings=True).tolist()
        collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        total += len(docs)
        logger.info("Indexed {} chunks so far", total)
        ids.clear(); docs.clear(); metas.clear()

    for chunk_id, text, meta in iter_chunks(args.source_dir):
        ids.append(chunk_id)
        docs.append(text)
        metas.append(meta)
        if len(docs) >= args.batch_size:
            flush()

    flush()
    logger.success(
        "Done. Collection '{}' now has {} chunks total.",
        settings.chroma_collection,
        collection.count(),
    )


if __name__ == "__main__":
    main()
