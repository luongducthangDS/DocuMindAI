import re
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class LegalChunk:
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def is_valid(self) -> bool:
        return self.char_count >= 50


# Matches "Điều 1.", "Điều 12a.", "Điều 1:" etc.
_DIEU_RE = re.compile(
    r"(Điều\s+\d+[a-z]?[\.\:]\s+.*?)(?=Điều\s+\d+[a-z]?[\.\:]|$)",
    re.DOTALL,
)

# Matches "Khoản 1.", "1." at line start inside an article
_KHOAN_RE = re.compile(r"(\n\d+\.\s)", re.MULTILINE)


def chunk_by_dieu(text: str, doc_meta: dict) -> list[LegalChunk]:
    """
    Primary strategy: split by Điều (article).
    Each Điều becomes one chunk — preserves full legal semantics.
    Falls back to sliding-window if no Điều structure found.
    """
    chunks: list[LegalChunk] = []

    matches = list(_DIEU_RE.finditer(text))

    if not matches:
        logger.warning(
            "No 'Điều' structure found in '{}', using fallback chunker",
            doc_meta.get("title", "unknown"),
        )
        return _fallback_chunks(text, doc_meta)

    _MAX_DIEU_CHARS = 4_000  # articles longer than this are split further

    for match in matches:
        chunk_text = match.group(1).strip()
        if len(chunk_text) < 50:
            continue

        # Split oversized articles by khoản boundary
        if len(chunk_text) > _MAX_DIEU_CHARS:
            sub_chunks = _split_large_dieu(chunk_text, doc_meta, _MAX_DIEU_CHARS)
            chunks.extend(sub_chunks)
            continue

        first_line = chunk_text.split("\n")[0][:120]
        khoan_count = len(_KHOAN_RE.findall(chunk_text))

        chunk = LegalChunk(
            text=chunk_text,
            metadata={
                "source_url": doc_meta.get("url", ""),
                "title": doc_meta.get("title", ""),
                "doc_type": doc_meta.get("doc_type", "unknown"),
                "so_hieu": doc_meta.get("so_hieu", ""),
                "ngay_ban_hanh": doc_meta.get("ngay_ban_hanh", ""),
                "dieu_header": first_line,
                "khoan_count": khoan_count,
                "char_count": len(chunk_text),
                "source": doc_meta.get("source", ""),
            },
        )
        chunks.append(chunk)

    logger.debug(
        "Chunked '{}' → {} chunks",
        doc_meta.get("title", "")[:40],
        len(chunks),
    )
    return chunks


def _fallback_chunks(
    text: str,
    doc_meta: dict,
    size: int = 800,
    overlap: int = 100,
) -> list[LegalChunk]:
    """
    Sliding-window fallback for documents without Điều structure.
    Splits on paragraph boundaries when possible.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks: list[LegalChunk] = []
    buffer = ""

    for para in paragraphs:
        if len(buffer) + len(para) <= size:
            buffer = buffer + "\n\n" + para if buffer else para
        else:
            if buffer:
                chunks.append(
                    LegalChunk(
                        text=buffer,
                        metadata={
                            **doc_meta,
                            "dieu_header": buffer[:80],
                            "char_count": len(buffer),
                            "chunk_strategy": "fallback_window",
                        },
                    )
                )
            # Keep last paragraph for overlap
            buffer = para

    if buffer:
        chunks.append(
            LegalChunk(
                text=buffer,
                metadata={
                    **doc_meta,
                    "dieu_header": buffer[:80],
                    "char_count": len(buffer),
                    "chunk_strategy": "fallback_window",
                },
            )
        )

    return chunks


def _split_large_dieu(
    dieu_text: str,
    doc_meta: dict,
    max_chars: int,
) -> list[LegalChunk]:
    """Split an oversized Điều into sub-chunks at khoản boundaries."""
    first_line = dieu_text.split("\n")[0][:120]
    base_meta = {
        "source_url": doc_meta.get("url", ""),
        "title": doc_meta.get("title", ""),
        "doc_type": doc_meta.get("doc_type", "unknown"),
        "so_hieu": doc_meta.get("so_hieu", ""),
        "ngay_ban_hanh": doc_meta.get("ngay_ban_hanh", ""),
        "dieu_header": first_line,
        "source": doc_meta.get("source", ""),
    }
    parts = re.split(r"(?=\n\d+\.\s)", dieu_text)
    chunks: list[LegalChunk] = []
    buffer = ""

    for part in parts:
        if len(buffer) + len(part) <= max_chars:
            buffer = buffer + part
        else:
            if buffer.strip():
                chunks.append(LegalChunk(
                    text=buffer.strip()[:max_chars],
                    metadata={**base_meta, "char_count": len(buffer), "khoan_count": 0},
                ))
            buffer = part

    if buffer.strip():
        chunks.append(LegalChunk(
            text=buffer.strip()[:max_chars],
            metadata={**base_meta, "char_count": len(buffer), "khoan_count": 0},
        ))

    return chunks if chunks else [LegalChunk(
        text=dieu_text[:max_chars] + "…",
        metadata={**base_meta, "char_count": max_chars, "khoan_count": 0},
    )]
