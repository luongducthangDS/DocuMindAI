"""
scripts/ingest_uneti_md.py — ingest UNETI .md files using DocuMind's chunk_by_dieu pipeline.

Re-chunks all 7 UNETI documents from raw Markdown using the project's own
Điều/Khoản chunker for better semantic boundaries than the old project's
header-based chunks.

Usage:
    python scripts/ingest_uneti_md.py --reset
    python scripts/ingest_uneti_md.py --reset --dry-run   # preview chunk counts only
"""

from __future__ import annotations

import argparse
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

_SOURCE_BASE = Path("D:/Projects for CV/chatbot_uneti_final/source")

# All 7 documents: (md_path, doc_meta)
DOCUMENTS = [
    (
        _SOURCE_BASE / "pdf/md/03.-Bản-chuẩn-QD-740.-20.10.2018-Quy-dinh-ve-hoc-kiem-tra-va-CĐR-ngoai-ngu.md",
        {
            "so_hieu": "QĐ-740/ĐHKTKTCN",
            "title": "Quy định về học, kiểm tra và chuẩn đầu ra ngoại ngữ - UNETI",
            "doc_type": "Quyết định",
            "ngay_ban_hanh": "2018-10-20",
            "url": "",
            "institution": "UNETI",
        },
    ),
    (
        _SOURCE_BASE / "pdf/md/4.-QD-747-20.10.2018.-Quy-chế-đánh-giá-điểm-rèn-luyện-của-sinh-viên.md",
        {
            "so_hieu": "QĐ-747/ĐHKTKTCN",
            "title": "Quy chế đánh giá điểm rèn luyện của sinh viên - UNETI",
            "doc_type": "Quyết định",
            "ngay_ban_hanh": "2018-10-20",
            "url": "",
            "institution": "UNETI",
        },
    ),
    (
        _SOURCE_BASE / "pdf/md/5.QĐ-748-20.10.2018.-Quy-định-về-học-bổng-khuyến-khích-học-tập-đối-với-sinh-viên-Trường-Đại-học-Kinh-tế-Kỹ-thuật-Công-nghiệp.md",
        {
            "so_hieu": "QĐ-748/ĐHKTKTCN",
            "title": "Quy định về học bổng khuyến khích học tập - UNETI",
            "doc_type": "Quyết định",
            "ngay_ban_hanh": "2018-10-20",
            "url": "",
            "institution": "UNETI",
        },
    ),
    (
        _SOURCE_BASE / "pdf/md/qd670-ban-hanh-quy-dinh-ve-cdr-tin-hoc-20250915082744-e.md",
        {
            "so_hieu": "QĐ-670/ĐHKTKTCN",
            "title": "Quy định về chuẩn đầu ra tin học - UNETI",
            "doc_type": "Quyết định",
            "ngay_ban_hanh": "2022-07-15",
            "url": "",
            "institution": "UNETI",
        },
    ),
    (
        _SOURCE_BASE / "scan/md/14.-QD-828-Qui-dinh-ve-viec-huong-dan-va-danh-gia-khoa-luan-tot-nghiep-doi-voi-sinh-vien-dai-hoc-chinh-quy-dao-tao-theo-he-thong-tin-chi.1-đã-chuyển-đổi.md",
        {
            "so_hieu": "QĐ-828/ĐHKTKTCN",
            "title": "Quy định hướng dẫn và đánh giá khóa luận tốt nghiệp - UNETI",
            "doc_type": "Quyết định",
            "ngay_ban_hanh": "2023-09-25",
            "url": "",
            "institution": "UNETI",
        },
    ),
    (
        _SOURCE_BASE / "scan/md/qd-1228-phuong-thuc-danh-gia-cdr-20250106041456-e (1).md",
        {
            "so_hieu": "QĐ-1228/ĐHKTKTCN",
            "title": "Phương thức đánh giá chuẩn đầu ra chương trình đào tạo - UNETI",
            "doc_type": "Quyết định",
            "ngay_ban_hanh": "2024-12-27",
            "url": "",
            "institution": "UNETI",
        },
    ),
    (
        _SOURCE_BASE / "scan/md/qd-853-2592023-20240826115209-e.md",
        {
            "so_hieu": "QĐ-853/ĐHKTKTCN",
            "title": "Quy định ngoại ngữ tiếng Anh cho sinh viên đại học chính quy - UNETI",
            "doc_type": "Quyết định",
            "ngay_ban_hanh": "2023-09-25",
            "url": "",
            "institution": "UNETI",
        },
    ),
]

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Clear existing collection first")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="Preview chunks without writing to DB")
    args = parser.parse_args()

    settings = get_settings()

    if args.dry_run:
        total = 0
        for md_path, doc_meta in DOCUMENTS:
            if not md_path.exists():
                logger.warning("Missing: {}", md_path)
                continue
            text = clean_md(md_path.read_text(encoding="utf-8"))
            chunks = [c for c in chunk_by_dieu(text, doc_meta) if c.is_valid]
            logger.info("{} → {} chunks", doc_meta["so_hieu"], len(chunks))
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

    for md_path, doc_meta in DOCUMENTS:
        if not md_path.exists():
            logger.warning("File not found, skipping: {}", md_path.name)
            continue

        text = clean_md(md_path.read_text(encoding="utf-8"))
        chunks = [c for c in chunk_by_dieu(text, doc_meta) if c.is_valid]
        logger.info("{} → {} chunks", doc_meta["so_hieu"], len(chunks))

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_meta['so_hieu'].replace('/', '_')}:{idx}:{abs(hash(chunk.text[:100]))}"
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
