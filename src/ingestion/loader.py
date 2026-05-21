"""
Loader module: HuggingFace dataset + uploaded PDF/plain-text files.
"""

import io
import json
from pathlib import Path
from typing import Iterator

from loguru import logger

from src.config import get_settings
from src.ingestion.cleaner import clean_legal_text


def iter_hf_dataset(
    dataset_name: str = "th1nhng0/vietnamese_legal_corpus",
    split: str = "train",
    max_docs: int | None = None,
) -> Iterator[dict]:
    """
    Stream documents from HuggingFace dataset to avoid loading 6GB into RAM.
    Yields normalized doc dicts with keys: title, content, url, source, doc_type.
    """
    try:
        from datasets import load_dataset  # lazy import — heavy dependency
    except ImportError as e:
        raise RuntimeError("Install 'datasets' package: pip install datasets") from e

    settings = get_settings()
    logger.info("Loading HF dataset: {} / {}", dataset_name, split)

    ds = load_dataset(
        dataset_name,
        split=split,
        streaming=True,
        token=settings.hf_token or None,
    )

    count = 0
    for row in ds:
        if max_docs and count >= max_docs:
            break

        content = row.get("text") or row.get("content") or row.get("van_ban", "")
        title = row.get("title") or row.get("ten_van_ban", f"doc_{count}")
        url = row.get("url") or row.get("source_url", "")
        doc_type = row.get("loai_van_ban") or row.get("doc_type", "unknown")

        if not content.strip():
            continue

        cleaned = clean_legal_text(content, doc_title=title)
        if not cleaned:
            continue

        yield {
            "title": title,
            "content": cleaned,
            "url": url,
            "doc_type": doc_type,
            "source": "huggingface",
            "so_hieu": row.get("so_hieu", ""),
            "ngay_ban_hanh": str(row.get("ngay_ban_hanh", "")),
        }
        count += 1

    logger.info("HF dataset loaded: {} documents processed", count)


def load_pdf(file_bytes: bytes, filename: str) -> dict | None:
    """
    Extract text from uploaded PDF. Uses pdfplumber (better for structured docs).
    Returns None if extraction fails or result is too short.
    """
    _validate_pdf_bytes(file_bytes)

    try:
        import pdfplumber  # lazy import
    except ImportError as e:
        raise RuntimeError("Install 'pdfplumber': pip install pdfplumber") from e

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if text:
                    pages_text.append(text)
            full_text = "\n".join(pages_text)
    except Exception as exc:
        logger.error("PDF extraction failed for {}: {}", filename, exc)
        return None

    if len(full_text.strip()) < 100:
        logger.warning("PDF '{}' yielded too little text ({} chars)", filename, len(full_text))
        return None

    cleaned = clean_legal_text(full_text, doc_title=filename)
    stem = Path(filename).stem[:100]

    return {
        "title": stem,
        "content": cleaned,
        "url": f"upload://{stem}",
        "doc_type": "uploaded_pdf",
        "source": "user_upload",
        "so_hieu": "",
        "ngay_ban_hanh": "",
    }


def load_json_file(path: Path) -> dict | None:
    """Load a single raw JSON document from disk (crawled data)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load JSON {}: {}", path, exc)
        return None

    content = data.get("content", "")
    if not content.strip():
        return None

    data["content"] = clean_legal_text(content, doc_title=data.get("title", ""))
    return data


def iter_raw_json_dir(raw_dir: Path) -> Iterator[dict]:
    """Yield all docs from data/raw/*.json."""
    json_files = sorted(raw_dir.glob("*.json"))
    logger.info("Found {} JSON files in {}", len(json_files), raw_dir)
    for path in json_files:
        doc = load_json_file(path)
        if doc:
            yield doc


def _validate_pdf_bytes(data: bytes) -> None:
    """Reject non-PDF and oversized uploads."""
    settings = get_settings()

    if len(data) > settings.max_upload_bytes:
        raise ValueError(
            f"File exceeds {settings.max_upload_size_mb} MB limit "
            f"({len(data) / 1_048_576:.1f} MB)"
        )

    # PDF magic bytes: %PDF-
    if not data[:5] == b"%PDF-":
        raise ValueError("Uploaded file is not a valid PDF (wrong magic bytes)")
