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
from datetime import date
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root))

from src.rag.context import stamp_access_meta
from src.hf_env import use_local_hf_cache

# Trước mọi import HuggingFace. offline=False: ingest là lúc hợp lệ để tải model
# chưa có trong cache (ví dụ vừa đổi EMBEDDING_MODEL).
use_local_hf_cache(offline=False, create=True)

import chromadb
from loguru import logger

from src.config import get_settings
from src.rag.embedder import STORE_META_DIM, STORE_META_MODEL, get_embedder, get_embedding_dim
from src.ingestion.chunker import LegalChunk, chunk_by_dieu
from src.ingestion.cleaner import strip_consolidated_footnotes, strip_consolidated_quotations
from src.ingestion.manifest import DocEntry, load_manifest
from src.ingestion.versions import build_version_chunks

# Model embedding đọc từ settings.embedding_model — không hard-code ở đây nữa, vì
# một bản sao lệch nghĩa là corpus được index bằng model khác model lúc truy vấn.

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


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def split_frontmatter(raw: str) -> tuple[dict, str]:
    """Return `(frontmatter, body)` for a corpus Markdown file."""
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw
    meta: dict[str, str] = {}
    for line in m.group(1).split("\n"):
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, raw[m.end():]


def _chroma_safe(metadata: dict) -> dict:
    """ChromaDB metadata values must be scalars — normalise empties to ''."""
    out: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        out[key] = "" if value is None else value
    return out


def build_clause_chunks(entry: DocEntry, corpus_dir: Path, as_of: str) -> list[LegalChunk]:
    """Clause-level chunks (current text + superseded versions) for one document."""
    path = corpus_dir / entry.source_file
    if not path.exists():
        logger.error("{}: missing corpus file {}", entry.doc_id, path)
        return []

    frontmatter, body = split_frontmatter(path.read_text(encoding="utf-8"))
    if entry.consolidated_from:
        # VBHN carries editorial footnotes and quotes the amending laws verbatim
        body = strip_consolidated_quotations(body)
        body = strip_consolidated_footnotes(body)
    body = clean_md(body)

    doc_meta = {
        "doc_id": entry.doc_id,
        "so_hieu": entry.so_hieu,
        "title": entry.ten,
        "doc_type": entry.loai,
        "ngay_ban_hanh": entry.ngay_ban_hanh,
        "url": frontmatter.get("nguon", entry.nguon),
        "source": "congbao.chinhphu.vn",
        "effective_from": entry.ngay_hieu_luc,
        "effective_to": entry.effective_to,
        "verify_status": "VERIFIED",
        "consolidated_from": entry.consolidated_from,
        "as_of": as_of,
    }

    chunks = chunk_by_dieu(body, doc_meta, entry.in_place_amended_clauses)
    chunks += build_version_chunks(entry, corpus_dir, doc_meta, as_of=as_of)
    return [c for c in chunks if c.is_valid]


def discover_lao_dong(manifest_path: Path, source_dir: Path, as_of: str) -> list[tuple[str, list[LegalChunk]]]:
    """Chunks per document for the lao động/BHXH corpus, driven by the YAML manifest.

    Only documents listed in `locked_list_v1` are considered, and only VERIFIED
    ones are indexed (CORPUS_SPEC §2) — an unverified document must never reach
    the index, however complete its text looks.
    """
    entries = load_manifest(manifest_path)
    out: list[tuple[str, list[LegalChunk]]] = []
    for doc_id, entry in entries.items():
        if not entry.is_verified:
            logger.warning("Skipping {} — verify_status is not VERIFIED", doc_id)
            continue
        chunks = build_clause_chunks(entry, source_dir, as_of)
        if chunks:
            out.append((doc_id, chunks))
    return out


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
    parser.add_argument("--as-of", type=str, default=date.today().isoformat(),
                        help="Build date used to derive clause status (default: today)")
    args = parser.parse_args()

    settings = get_settings()

    if not args.source_dir.exists():
        logger.error("Source directory not found: {}", args.source_dir)
        sys.exit(1)

    # YAML manifest -> clause-level pipeline (corpus lao động/BHXH);
    # JSON manifest -> legacy per-file pipeline.
    clause_mode = args.manifest is not None and args.manifest.suffix.lower() in {".yaml", ".yml"}

    if clause_mode:
        per_doc = discover_lao_dong(args.manifest, args.source_dir, args.as_of)
        documents = []
    else:
        per_doc = []
        documents = discover_documents(args.source_dir, args.manifest, args.institution)

    if not per_doc and not documents:
        logger.warning("Nothing to ingest from {}", args.source_dir)
        return

    if args.dry_run:
        total = 0
        for doc_id, chunks in per_doc:
            logger.info("{} → {} chunks", doc_id, len(chunks))
            total += len(chunks)
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

    # Dùng chung embedder với đường truy vấn (src.rag.embedder) — trước đây script
    # tự dựng SentenceTransformer, nên khi EMBEDDING_MODEL trỏ tới model API
    # (gemini-embedding-001) nó đi tải "sentence-transformers/gemini-embedding-001"
    # trên HuggingFace và chết với 401.
    model_name = settings.embedding_model
    embedder = get_embedder()
    embedding_dim = get_embedding_dim()

    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={
            "hnsw:space": "cosine",
            # Nhãn để lần mở store sau đối chiếu được với EMBEDDING_MODEL đang cấu hình.
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
        # Stamped here, not in the chunker: `metas` has already been through the
        # clause-version rewrites that close `effective_to` on superseded text,
        # so this is the first point where the integer mirrors are guaranteed to
        # match the strings they mirror (see src/rag/context.stamp_access_meta).
        stamped = [stamp_access_meta(m) for m in metas]
        collection.upsert(ids=ids, documents=docs, metadatas=stamped, embeddings=embeddings)
        total += len(docs)
        logger.info("Indexed {} chunks so far", total)
        ids.clear(); docs.clear(); metas.clear()

    for doc_id, chunks in per_doc:
        logger.info("{} → {} chunks", doc_id, len(chunks))
        for idx, chunk in enumerate(chunks):
            # version_id is unique per clause version; fall back to an index for
            # chunks that carry no clause id (e.g. preamble)
            chunk_id = chunk.metadata.get("version_id") or f"{doc_id}:{idx}"
            ids.append(chunk_id)
            docs.append(chunk.text)
            metas.append(_chroma_safe(chunk.metadata))
            if len(docs) >= args.batch_size:
                flush()

    for md_path, doc_meta in documents:
        text = clean_md(md_path.read_text(encoding="utf-8"))
        chunks = [c for c in chunk_by_dieu(text, doc_meta) if c.is_valid]
        doc_key = doc_meta.get("so_hieu") or md_path.stem
        logger.info("{} → {} chunks", doc_key, len(chunks))

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_key.replace('/', '_')}:{idx}:{abs(hash(chunk.text[:100]))}"
            ids.append(chunk_id)
            docs.append(chunk.text)
            metas.append(_chroma_safe(chunk.metadata))
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
