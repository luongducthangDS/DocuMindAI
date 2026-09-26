"""
Guardrails: prompt injection detection + post-hoc citation validation.
Pure stdlib — no project imports, independently testable.

Patterns are matched against a *normalized* form of the query (see `_normalize`):
accents stripped, homoglyphs folded, intra-word separators removed. Writing the
Vietnamese patterns by hand over accented text is what broke the previous
version — `d[aẫ][ấ]n` expects four code points for the three-code-point word
"dẫn", so it never fired on a single real query.

ponytail: normalize + regex has a ceiling — it catches phrasings we listed, not
novel ones. Measured against `data/eval/guardrail_cases.json`; when a family of
attacks starts slipping past, an LLM-based classifier on the query is the next
rung, not more regexes.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass
class GuardResult:
    blocked: bool
    reason: str = ""
    pattern_matched: str = ""
    score: float = 0.0


# ── Normalization ──────────────────────────────────────────────────────────────

# Cyrillic/Greek letters that render like Latin ones. Folding them means an
# attacker swapping "о" for "o" hits the same pattern as plain text.
_HOMOGLYPH_MAP = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y", "і": "i",
    "ѕ": "s", "к": "k", "м": "m", "т": "t", "в": "b", "н": "h", "г": "r", "ԁ": "d",
    "ј": "j", "ӏ": "l", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I",
    "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "ο": "o", "ρ": "p", "α": "a", "ε": "e", "ν": "v", "ι": "i", "κ": "k", "τ": "t",
})
_ZERO_WIDTH = re.compile(r"[​-‏⁠﻿­]")
_INTRAWORD_SEP = re.compile(r"(?<=[a-z0-9])[-_.](?=[a-z])", re.I)
_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercased, accent-free, homoglyph-free view used for pattern matching.

    Punctuation that carries meaning for a pattern (`:`, `[`, `<`) is kept;
    only separators *inside* a word are dropped, so "Ignore-all-previous"
    reads the same as "Ignore all previous".
    """
    text = _ZERO_WIDTH.sub("", unicodedata.normalize("NFC", text)).translate(_HOMOGLYPH_MAP)
    text = text.replace("đ", "d").replace("Đ", "D")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))
    return _WHITESPACE.sub(" ", _INTRAWORD_SEP.sub(" ", text)).strip().lower()


# ── Prompt injection patterns (matched against _normalize output) ──────────────

_INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Role-override — English
    (re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\b"), "role_override"),
    (re.compile(r"\byou\s+are\s+now\b"), "role_override"),
    (re.compile(r"\bact\s+as\s+(if\s+you\s+(are|were)|a\s+)"), "role_override"),
    (re.compile(r"\bpretend\s+(you\s+are|to\s+be)\b"), "role_override"),
    (re.compile(r"\bdan\b.*\bmode\b|\bdo\s+anything\s+now\b"), "role_override"),
    # Role-override — Vietnamese
    (re.compile(r"\b(bay gio|tu gio|tu nay|ke tu gio|gio day) ban (la|se la)\b"), "role_override_vi"),
    (re.compile(r"\bban (khong con la|khong phai la|thoi khong la)\b"), "role_override_vi"),
    (re.compile(r"\bgia vo (nhu |rang )?(ban|minh) (la|dang)\b"), "role_override_vi"),
    # "dong vai tro" ("plays a role in…") is ordinary legal Vietnamese — exclude it.
    (re.compile(r"\b(dong vai(?! tro\b)|nhap vai|vao vai|hoa than thanh)\b"), "role_override_vi"),
    (re.compile(r"\bban la (mot )?(ai|tro ly|he thong|chatbot|luat su) (khong|tu do|moi|rieng)\b"), "role_override_vi"),
    # Instruction override — English
    (re.compile(r"\bdisregard\s+(all\s+)?(your\s+)?(instructions?|rules?|guidelines?)\b"), "instruction_override"),
    (re.compile(r"\bforget\s+(everything|all)\s+(you\s+)?(know|were)\b"), "instruction_override"),
    (re.compile(r"\bnew\s+instructions?\s*:"), "instruction_override"),
    # Instruction override — Vietnamese.
    # "quy dinh moi:" is excluded on purpose: "Quy định mới về lương tối thiểu…"
    # is one of the most common real questions in this domain.
    (re.compile(r"\b(huong dan|chi dan|quy tac|yeu cau|lenh) moi\s*:"), "instruction_override_vi"),
    # "bo qua gioi han/rang buoc" is how people describe an employer breaking the
    # law ("cong ty bo qua gioi han lam them gio") — keep the object list narrow.
    (re.compile(r"\bbo qua (moi |tat ca |het |nhung )?(huong dan|chi dan|quy tac|bo loc)\b"), "instruction_override_vi"),
    (re.compile(r"\b(quen|xoa) (het|di|sach) (moi thu|tat ca|nhung gi)\b"), "instruction_override_vi"),
    (re.compile(r"\bkhong (can |phai )?trich (dan|nguon)\b"), "instruction_override_vi"),
    (re.compile(r"\btu (gio|nay|cau nay) .{0,30}(moi cau tra loi|khong can|ban phai)"), "instruction_override_vi"),
    # "bi cam" = prohibited by law, the single most common verb in this corpus.
    (re.compile(r"(?<!bi )\bcam (tra loi|noi)\b"), "instruction_override_vi"),
    # Prompt leakage — English
    (re.compile(r"\bprint\s+(your\s+)?(system\s+)?prompt\b"), "prompt_leakage"),
    (re.compile(r"\brepeat\s+(after\s+me|your\s+(system|instructions?))\b"), "prompt_leakage"),
    (re.compile(r"\bwhat\s+(are|were)\s+your\s+(instructions?|system\s+prompt|rules?)\b"), "prompt_leakage"),
    (re.compile(r"\bsystem prompt\b|\bprompt he thong\b"), "prompt_leakage"),
    # Prompt leakage — Vietnamese
    (re.compile(r"\bban duoc (cau hinh|huong dan|dan do|lap trinh|gan|day) .{0,30}(gi|nao|the nao|nhung)\b"), "prompt_leakage_vi"),
    (re.compile(r"\b(nhac lai|lap lai|in lai|cho (toi )?xem) .{0,40}(nguyen van|chinh xac) .{0,20}(doan|noi dung|van ban|chi dan|huong dan)"), "prompt_leakage_vi"),
    (re.compile(r"\bten file\b|\bduong dan (tai lieu|file) (ban|cua ban|tren may chu|trong he thong)\b"), "prompt_leakage_vi"),
    # Injection carried in pasted content / markup
    (re.compile(r"\[\s*system\s*[:\]]"), "embedded_injection"),
    (re.compile(r"<!--[^>]{0,200}(instruction|huong dan|tra loi|answer)"), "embedded_injection"),
    (re.compile(r"\bneu ban la (ai|mot ai|chatbot|model|tro ly)\b"), "embedded_injection"),
    # Exfiltration
    (re.compile(r"\b(bien moi truong|environment variable|api[ _]?key|secret key)\b|\.env\b"), "exfiltration"),
    (re.compile(r"\b(toan bo|tat ca) context\b|\bcontext ban duoc\b"), "exfiltration"),
    (re.compile(r"\bgui .{0,40}(toi|den|sang|ve) https?://"), "exfiltration"),
    # Obfuscated payload: an instruction to decode, plus something decodable.
    (re.compile(r"\b(giai ma|decode|base64)\b.{0,40}[a-z0-9+/]{20,}={0,2}"), "obfuscation"),
]

# Heuristic signals
_INSTRUCTION_VERBS = re.compile(
    r"\b(answer|respond|tell\s+me|give\s+me|show\s+me|output|write|generate|produce)\b", re.I
)
_SELF_REF_NOUNS = re.compile(
    r"\b(yourself|your\s+instructions?|your\s+prompt|your\s+system|your\s+rules?|your\s+training)\b", re.I
)
_HOMOGLYPH = re.compile(r"[Ѐ-ӿͰ-Ͽ]")  # Cyrillic/Greek mixed into Latin text


def check_prompt_injection(query: str) -> GuardResult:
    """
    Fail-open: unexpected errors return GuardResult(blocked=False).
    Two-stage check: regex patterns over the normalized query + heuristic scoring.
    """
    try:
        normalized = _normalize(query)

        # Stage 1: exact pattern match
        for pattern, category in _INJECTION_PATTERNS:
            if pattern.search(normalized):
                return GuardResult(
                    blocked=True,
                    reason=f"Detected prompt injection attempt ({category})",
                    pattern_matched=category,
                    score=1.0,
                )

        # Stage 2: heuristic scoring
        score = 0.0
        if _HOMOGLYPH.search(query):
            score += 1.0
        if _INSTRUCTION_VERBS.search(query) and _SELF_REF_NOUNS.search(query):
            score += 1.0
        if len(query) > 600:
            score += 0.5

        if score >= 1.5:
            return GuardResult(
                blocked=True,
                reason="Query matches suspicious heuristic pattern",
                pattern_matched="heuristic",
                score=score,
            )

        return GuardResult(blocked=False, score=score)

    except Exception:
        return GuardResult(blocked=False, reason="guard_error")


# ── Citation validation ────────────────────────────────────────────────────────

# Matches [N] or [NN] but NOT [Khoản 1] or [Điều 5] style text
_CITATION_RE = re.compile(r"(?<!\w)\[(\d{1,2})\](?!\w)")

# Same markers, plus the combined "[1, 2]" form the LLM sometimes emits.
# Kept separate from _CITATION_RE on purpose: validate_citations' stripping
# behaviour is calibrated for single-index markers, and widening it there would
# change what production answers look like. This one is read-only — it reports
# which sources an answer claims, for evaluation.
_CITED_INDICES_RE = re.compile(r"(?<!\w)\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\](?!\w)")

_CITATION_WARNING_VI = (
    "\n\n_(Lưu ý: một số trích dẫn không khớp với nguồn được truy xuất và đã được xóa.)_"
)


def cited_indices(answer: str) -> list[int]:
    """Source indices an answer claims, in order of first appearance.

    Read-only counterpart to validate_citations: it reports what the answer
    cites without judging or rewriting it. Handles both "[1]" and "[1, 2]".
    """
    seen: list[int] = []
    for bracket in _CITED_INDICES_RE.findall(answer or ""):
        for piece in bracket.split(","):
            n = int(piece.strip())
            if n not in seen:
                seen.append(n)
    return seen


def validate_citations(answer: str, chunk_count: int) -> tuple[str, list[int]]:
    """
    Scan answer for [N] citation markers.
    Returns (cleaned_answer, list_of_invalid_indices).
    Invalid = N < 1 or N > chunk_count.
    Strips invalid citations from text and appends a Vietnamese warning note.
    """
    if not answer or chunk_count <= 0:
        return answer, []

    found = [int(m) for m in _CITATION_RE.findall(answer)]
    if not found:
        return answer, []

    invalid = sorted({n for n in found if n < 1 or n > chunk_count})
    if not invalid:
        return answer, []

    cleaned = answer
    for n in invalid:
        cleaned = re.sub(rf"(?<!\w)\[{n}\](?!\w)", "", cleaned)

    cleaned = cleaned.rstrip() + _CITATION_WARNING_VI
    return cleaned, invalid
