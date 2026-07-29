"""
scripts/ingest_documents.py — ingest a folder of banking-document Markdown
files into ChromaDB using DocuMind's chunk_by_dieu pipeline.

Generic replacement for the old UNETI-specific ingest script: point it at any
directory of .md files (quy định, biểu phí, quy trình nghiệp vụ, thông tư/quyết
định NHNN, v.v.) instead of a hardcoded document list. Per-file metadata
(số hiệu, tiêu đề, ngày ban hành...) comes from an optional manifest JSON;
files without a manifest entry get filename-derived defaults so the pipeline
still runs before real metadata is authored.

Manifest format (--manifest path/to/manifest.json):
{
  "ten-file.md": {
    "so_hieu": "TT-01/2024/TT-NHNN",
    "title": "Thông tư quy định về ...",
    "doc_type": "Thông tư",
    "ngay_ban_hanh": "2024-01-15",
    "url": "",
    "institution": "NHNN"
  }
}

Usage:
    python scripts/ingest_documents.py --source-dir data/raw/banking_docs --reset
    python scripts/ingest_documents.py --source-dir data/raw/banking_docs --manifest data/raw/manifest.json
    python scripts/ingest_documents.py --source-dir data/raw/banking_docs --dry-run   # preview chunk counts only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Force local HF cache before any HuggingFace imports
_repo_root = Path(__file__).resolve().parents[1]
_local_hf = _repo_root / "data" / "hf_cache"
_local_hf.mkdir(parents=True, exist_ok=True)
for _k in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
    os.environ[_k] = str(_local_hf)

import chromadb
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import get_settings
from src.ingestion.chunker import chunk_by_dieu

INDEXED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MAX_LINE_CHARS = 2_000  # truncate scan-bloated table cells


def clean_md(text: str) -> str:
    """Remove HTML comments and truncate scan-bloated lines (e.g. 429KB table row)."""
    text = _HTML_COMMENT_RE.sub("", text)
    lines = []
    for line in text.split("\n"):
        if len(line) > _MAX_LINE_CHARS:
            line = line[:_MAX_LINE_CHARS] + "…[cắt bớt]"
        lines.append(line)
    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return text.strip()


def _default_meta(md_path: Path, institution: str) -> dict:
    """Filename-derived placeholder metadata for files with no manifest entry."""
    return {
        "so_hieu": md_path.stem,
        "title": md_path.stem.replace("-", " ").replace("_", " "),
        "doc_type": "Văn bản",
        "ngay_ban_hanh": "",
        "url": "",
        "institution": institution,
    }


def discover_documents(source_dir: Path, manifest_path: Path | None, institution: str) -> list[tuple[Path, dict]]:
    manifest: dict[str, dict] = {}
    if manifest_path is not None:
        if not manifest_path.exists():
            logger.warning("Manifest not found: {}", manifest_path)
        else:
            with manifest_path.open(encoding="utf-8") as f:
                manifest = json.load(f)

    documents: list[tuple[Path, dict]] = []
    for md_path in sorted(source_dir.rglob("*.md")):
        doc_meta = manifest.get(md_path.name) or _default_meta(md_path, institution)
        documents.append((md_path, doc_meta))
    return documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True, help="Directory of .md documents to ingest (searched recursively)")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional JSON file mapping filename -> metadata")
    parser.add_argument("--institution", type=str, default="", help="Default institution tag for files without a manifest entry")
    parser.add_argument("--reset", action="store_true", help="Clear existing collection first")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="Preview chunks without writing to DB")
    args = parser.parse_args()

    settings = get_settings()

    if not args.source_dir.exists():
        logger.error("Source directory not found: {}", args.source_dir)
        sys.exit(1)

    documents = discover_documents(args.source_dir, args.manifest, args.institution)
    if not documents:
        logger.warning("No .md files found under {}", args.source_dir)
        return

    if args.dry_run:
        total = 0
        for md_path, doc_meta in documents:
            text = clean_md(md_path.read_text(encoding="utf-8"))
            chunks = [c for c in chunk_by_dieu(text, doc_meta) if c.is_valid]
            logger.info("{} → {} chunks", doc_meta.get("so_hieu", md_path.name), len(chunks))
            total += len(chunks)
        logger.info("Total chunks (dry-run): {}", total)
        return

    chroma_path = str((_repo_root / "data" / "chroma_db").resolve())
    client = chromadb.PersistentClient(path=chroma_path)

    if args.reset:
        try:
            client.delete_collection(settings.chroma_collection)
            logger.warning("Deleted collection '{}'", settings.chroma_collection)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )

    model = SentenceTransformer(INDEXED_MODEL)
    logger.info("Model loaded: {}", INDEXED_MODEL)

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

    for md_path, doc_meta in documents:
        text = clean_md(md_path.read_text(encoding="utf-8"))
        chunks = [c for c in chunk_by_dieu(text, doc_meta) if c.is_valid]
        doc_key = doc_meta.get("so_hieu") or md_path.stem
        logger.info("{} → {} chunks", doc_key, len(chunks))

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_key.replace('/', '_')}:{idx}:{abs(hash(chunk.text[:100]))}"
            ids.append(chunk_id)
            docs.append(chunk.text)
            metas.append(chunk.metadata)
            if len(docs) >= args.batch_size:
                flush()

    flush()
    logger.success(
        "Done. Collection '{}' has {} chunks total.",
        settings.chroma_collection,
        collection.count(),
    )


if __name__ == "__main__":
    main()
