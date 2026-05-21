import re
import unicodedata

from loguru import logger


# Vietnamese legal text noise patterns
_NOISE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\s{3,}"),                       # excessive whitespace
    re.compile(r"\n{3,}"),                        # excessive newlines
    re.compile(r"\.{4,}"),                        # dotted leaders (table of contents)
    re.compile(r"[-─]{10,}"),                     # horizontal rules
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


def clean_legal_text(text: str, doc_title: str = "") -> str:
    """
    Full cleaning pipeline for Vietnamese legal documents.
    Order matters: unicode → noise → whitespace → final strip.
    """
    if not text or not text.strip():
        logger.warning("Empty text received for cleaning: {}", doc_title)
        return ""

    text = normalize_unicode(text)
    text = remove_noise(text)

    # Normalize line breaks around article headers
    text = re.sub(r"\n(Điều\s+\d+)", r"\n\n\1", text)
    text = re.sub(r"\n(Chương\s+[IVXLCDM]+)", r"\n\n\1", text)
    text = re.sub(r"\n(Mục\s+\d+)", r"\n\n\1", text)

    # Collapse multiple spaces but preserve indentation structure
    lines = [re.sub(r"[ \t]{2,}", " ", line) for line in text.splitlines()]
    text = "\n".join(lines)

    logger.debug("Cleaned text: {} chars → {}", len(text), doc_title[:40])
    return text.strip()
