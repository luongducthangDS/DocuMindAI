import re
import unicodedata

from loguru import logger

# Vietnamese legal text noise patterns
_NOISE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\s{3,}"),                       # excessive whitespace
    re.compile(r"\n{3,}"),                        # excessive newlines
    re.compile(r"\.{4,}"),                        # dotted leaders (table of contents)
    re.compile(r"[-─]{10,}"),               # horizontal rules
    re.compile(r"Trang\s+\d+\s*/\s*\d+"),        # page numbers
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),    # lone page numbers
]

_UNICODE_REPLACE = str.maketrans({
    "–": "-",   # en-dash
    "—": "-",   # em-dash
    "“": '"',   # left double quote
    "”": '"',   # right double quote
    "‘": "'",   # left single quote
    "’": "'",   # right single quote
    " ": " ",   # non-breaking space
    "​": "",    # zero-width space
    "﻿": "",    # BOM
})


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return text.translate(_UNICODE_REPLACE)


def remove_noise(text: str) -> str:
    for pat in _NOISE_PATTERNS:
        if pat.pattern.startswith("\\s{3"):
            text = pat.sub("  ", text)
        elif pat.pattern.startswith("\\n{3"):
            text = pat.sub("\n\n", text)
        else:
            text = pat.sub("", text)
    return text.strip()


# Consolidated texts (văn bản hợp nhất) carry an inline footnote right after the
# clause number: "1.44 Khoản này được sửa đổi ... năm 2026. Lao động nữ được ...".
# The footnote is editorial, not normative — and it glues the clause number to the
# footnote index, which hides the khoản boundary from the chunker.
_VBHN_FOOTNOTE_KHOAN_RE = re.compile(
    r"^(?P<so>\d+[a-z]?)\.\d{1,3}\s+"
    r"(?:Khoản|Điểm|Điều|Mục|Chương)\s+này được[^.]*?"
    r"có hiệu lực kể từ ngày[^.]*\.\s*",
    re.MULTILINE,
)
_VBHN_FOOTNOTE_DIEM_RE = re.compile(
    r"^(?P<so>[a-zđ])\)\d{1,3}(?:\s+\d{1,3})?\s+"
    r"(?:Khoản|Điểm|Điều|Mục|Chương)\s+này được[^.]*?"
    r"có hiệu lực kể từ ngày[^.]*\.\s*",
    re.MULTILINE,
)


# A VBHN also quotes, inline, the implementation articles of each amending law:
#   ... Khoản 1 Điều 30 của Luật Dân số số 113/2025/QH15 ... quy định như sau:
#   "Điều 30. Hiệu lực thi hành 1. Luật này có hiệu lực ...".
# Those quoted articles belong to the amending law, not to this code — left in,
# they would be chunked as if the host document had a second "Điều 30".
_VBHN_QUOTED_ARTICLE_RE = re.compile(
    r"(?:Khoản\s+\d+\s+)?Điều\s+\d+\s+của\s+(?:Luật|Bộ luật|Nghị định|Pháp lệnh)[^“\"]{0,300}?"
    r"quy định như sau:\s*[“\"].*?[”\"]\s*\.",
    re.DOTALL,
)

# Footnote index glued to the preceding word: "ĐIỀU KHOẢN THI HÀNH66 Điều 50 ..."
_VBHN_GLUED_FOOTNOTE_RE = re.compile(
    r"(?<=[^\W\d_])\d{1,3}(?=\s+Điều\s+\d+\s+của\s+(?:Luật|Bộ luật))"
)


def strip_consolidated_quotations(text: str) -> str:
    """Drop amending-law articles that a VBHN quotes verbatim."""
    text = _VBHN_GLUED_FOOTNOTE_RE.sub("", text)
    return _VBHN_QUOTED_ARTICLE_RE.sub("", text)


def strip_consolidated_footnotes(text: str) -> str:
    """Drop VBHN footnotes, restoring plain "1. " / "a) " clause markers.

    The amendment facts they state (which law, effective when) are already held as
    metadata in `corpus_manifest.yaml`, so keeping them in the clause text would
    only add editorial noise to retrieval and generation.
    """
    text = _VBHN_FOOTNOTE_KHOAN_RE.sub(lambda m: f"{m.group('so')}. ", text)
    text = _VBHN_FOOTNOTE_DIEM_RE.sub(lambda m: f"{m.group('so')}) ", text)
    return text


def clean_legal_text(text: str, doc_title: str = "") -> str:
    """
    Full cleaning pipeline for Vietnamese legal documents.
    Order matters: unicode -> noise -> whitespace -> final strip.
    """
    if not text or not text.strip():
        logger.warning("Empty text received for cleaning: {}", doc_title)
        return ""

    text = normalize_unicode(text)
    text = remove_noise(text)

    # Normalize line breaks around article headers.
    # Use (?<!\n) lookbehind to handle mid-text occurrences (no preceding newline).
    dieu = "Điều"
    chuong = "Chương"
    muc = "Mục"

    text = re.sub(r"(?<!\n)(" + dieu + r"\s+\d+)", r"\n\n\1", text)
    text = re.sub(r"\n{3,}(" + dieu + r"\s+\d+)", r"\n\n\1", text)
    text = re.sub(r"(?<!\n)(" + chuong + r"\s+[IVXLCDM]+)", r"\n\n\1", text)
    text = re.sub(r"\n{3,}(" + chuong + r"\s+[IVXLCDM]+)", r"\n\n\1", text)
    text = re.sub(r"(?<!\n)(" + muc + r"\s+\d+)", r"\n\n\1", text)
    text = re.sub(r"\n{3,}(" + muc + r"\s+\d+)", r"\n\n\1", text)

    # Collapse multiple spaces but preserve indentation structure
    lines = [re.sub(r"[ \t]{2,}", " ", line) for line in text.splitlines()]
    text = "\n".join(lines)

    logger.debug("Cleaned text: {} chars -> {}", len(text), doc_title[:40])
    return text.strip()
