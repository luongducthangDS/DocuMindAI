"""
Loader module: HuggingFace dataset + uploaded PDF/plain-text files.
"""

import io
import json
from pathlib import Path
from typing import BinaryIO, Iterator

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


# Thời gian extract tăng theo số trang, và một thread đã quá timeout thì không huỷ
# được (asyncio.to_thread) — giới hạn trang là thứ thật sự chặn trần CPU/RAM.
MAX_PDF_PAGES = 300


class PdfTooManyPages(ValueError):
    """ValueError để route trả 400 như các lỗi validate khác."""


def load_pdf(source: bytes | BinaryIO, filename: str) -> dict | None:
    """
    Extract text from uploaded PDF. Uses pdfplumber (better for structured docs).
    `source` là bytes hoặc file nhị phân seek được — route upload truyền thẳng file
    tạm Starlette đã spool ra disk, không nạp cả file vào RAM.
    Returns None if extraction fails or result is too short.
    Raises PdfTooManyPages above MAX_PDF_PAGES, before any page is parsed.
    """
    if isinstance(source, (bytes, bytearray)):
        _validate_pdf_bytes(source)
        stream: BinaryIO = io.BytesIO(source)
    else:
        stream = source
        size = stream.seek(0, io.SEEK_END)
        stream.seek(0)
        _validate_pdf_bytes(stream.read(5), size=size)
        stream.seek(0)

    try:
        import pdfplumber  # lazy import
    except ImportError as e:
        raise RuntimeError("Install 'pdfplumber': pip install pdfplumber") from e

    try:
        with pdfplumber.open(stream) as pdf:
            if len(pdf.pages) > MAX_PDF_PAGES:
                raise PdfTooManyPages(
                    f"PDF có {len(pdf.pages)} trang, vượt giới hạn {MAX_PDF_PAGES} trang"
                )
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if text:
                    pages_text.append(text)
            full_text = "\n".join(pages_text)
    except PdfTooManyPages:
        raise
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


# ---------- DOCX / XLSX ----------
# Cả hai là file zip. Kiểm magic bytes, rồi chặn zip bomb bằng tổng kích thước GIẢI NÉN
# khai trong central directory trước khi thư viện nào mở file.
UPLOAD_EXTENSIONS = {".pdf", ".docx", ".xlsx"}
MAX_UNZIPPED_BYTES = 100 * 1_048_576
MAX_ZIP_ENTRIES = 5_000
MAX_XLSX_ROWS = 50_000


def _read_all(source: bytes | BinaryIO) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    source.seek(0)
    return source.read()


def _validate_office_bytes(data: bytes) -> None:
    import zipfile

    settings = get_settings()
    if len(data) > settings.max_upload_bytes:
        raise ValueError(f"File exceeds {settings.max_upload_size_mb} MB limit ({len(data) / 1_048_576:.1f} MB)")
    if data[:4] != b"PK\x03\x04":
        raise ValueError("Uploaded file is not a valid DOCX/XLSX (wrong magic bytes)")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            infos = zf.infolist()
    except zipfile.BadZipFile as exc:
        raise ValueError("Uploaded file is not a valid DOCX/XLSX (corrupt zip)") from exc
    if len(infos) > MAX_ZIP_ENTRIES or sum(i.file_size for i in infos) > MAX_UNZIPPED_BYTES:
        raise ValueError("Uploaded file expands beyond the allowed size (possible zip bomb)")


def _upload_doc(filename: str, content: str, kind: str) -> dict:
    stem = Path(filename).stem[:100]
    return {"title": stem, "content": content, "url": f"upload://{stem}", "doc_type": f"uploaded_{kind}",
            "source": "user_upload", "so_hieu": "", "ngay_ban_hanh": ""}


def load_docx(source: bytes | BinaryIO, filename: str) -> dict | None:
    """Paragraphs and tables in document order; table rows become 'a | b | c' lines."""
    data = _read_all(source)
    _validate_office_bytes(data)
    from docx import Document
    from docx.table import Table

    try:
        parts = []
        for block in Document(io.BytesIO(data)).iter_inner_content():
            if isinstance(block, Table):
                for row in block.rows:
                    cells = [c.text.strip() for c in row.cells]
                    if any(cells):
                        parts.append(" | ".join(cells))
            elif block.text.strip():
                parts.append(block.text.strip())
    except Exception as exc:
        logger.error("DOCX extraction failed for {}: {}", filename, exc)
        return None
    text = "\n".join(parts)
    if len(text.strip()) < 100:
        logger.warning("DOCX '{}' yielded too little text ({} chars)", filename, len(text))
        return None
    return _upload_doc(filename, clean_legal_text(text, doc_title=filename), "docx")


def load_xlsx(source: bytes | BinaryIO, filename: str) -> dict | None:
    """One line per row, each value tagged with its column header ('Sheet | Header: value; ...').

    A sliding-window chunk then still says what every number means, which a raw grid dump does not.
    Formulas are read as their cached values (data_only); a formula string is never indexed.
    """
    data = _read_all(source)
    _validate_office_bytes(data)
    from openpyxl import load_workbook

    lines: list[str] = []
    try:
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        for ws in wb.worksheets:
            header: list[str] | None = None
            for row in ws.iter_rows(values_only=True):
                vals = ["" if v is None else str(v).strip() for v in row]
                if not any(vals):
                    continue
                if header is None:
                    header = [h or f"Cột {i + 1}" for i, h in enumerate(vals)]
                    continue
                pairs = [f"{header[i] if i < len(header) else f'Cột {i + 1}'}: {v}" for i, v in enumerate(vals) if v]
                lines.append(f"{ws.title} | " + "; ".join(pairs))
                if len(lines) >= MAX_XLSX_ROWS:
                    raise ValueError(f"Workbook has more than {MAX_XLSX_ROWS} data rows")
        wb.close()
    except ValueError:
        raise
    except Exception as exc:
        logger.error("XLSX extraction failed for {}: {}", filename, exc)
        return None
    text = "\n".join(lines)
    if len(text.strip()) < 100:
        logger.warning("XLSX '{}' yielded too little text ({} chars)", filename, len(text))
        return None
    return _upload_doc(filename, text, "xlsx")


def load_upload(source: bytes | BinaryIO, filename: str) -> dict | None:
    """Dispatch on extension. Content is still verified by magic bytes inside each loader."""
    ext = Path(filename).suffix.lower()
    if ext == ".docx":
        return load_docx(source, filename)
    if ext == ".xlsx":
        return load_xlsx(source, filename)
    if ext == ".pdf":
        return load_pdf(source, filename)
    raise ValueError(f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(UPLOAD_EXTENSIONS))}")


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


def _validate_pdf_bytes(data: bytes, size: int | None = None) -> None:
    """Reject non-PDF and oversized uploads. `size` = tổng kích thước khi `data` chỉ là phần đầu file."""
    settings = get_settings()
    size = len(data) if size is None else size

    if size > settings.max_upload_bytes:
        raise ValueError(
            f"File exceeds {settings.max_upload_size_mb} MB limit "
            f"({size / 1_048_576:.1f} MB)"
        )

    # PDF magic bytes: %PDF-
    if not data[:5] == b"%PDF-":
        raise ValueError("Uploaded file is not a valid PDF (wrong magic bytes)")
