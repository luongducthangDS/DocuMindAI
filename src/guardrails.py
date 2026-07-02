"""
Guardrails: prompt injection detection + post-hoc citation validation.
Pure stdlib — no project imports, independently testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class GuardResult:
    blocked: bool
    reason: str = ""
    pattern_matched: str = ""
    score: float = 0.0


# ── Prompt injection patterns ──────────────────────────────────────────────────

_INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Role-override
    (re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\b", re.I), "role_override"),
    (re.compile(r"\byou\s+are\s+now\b", re.I), "role_override"),
    (re.compile(r"\bact\s+as\s+(if\s+you\s+(are|were)|a\s+)", re.I), "role_override"),
    (re.compile(r"\bpretend\s+(you\s+are|to\s+be)\b", re.I), "role_override"),
    (re.compile(r"\bdan\b.*\bmode\b|\bdo\s+anything\s+now\b", re.I), "role_override"),
    # Instruction override
    (re.compile(r"\bdisregard\s+(all\s+)?(your\s+)?(instructions?|rules?|guidelines?)\b", re.I), "instruction_override"),
    (re.compile(r"\bforget\s+(everything|all)\s+(you\s+)?(know|were)\b", re.I), "instruction_override"),
    (re.compile(r"\bnew\s+instructions?\s*:", re.I), "instruction_override"),
    (re.compile(r"\bsystem\s+prompt\b", re.I), "instruction_override"),
    # Prompt leakage
    (re.compile(r"\bprint\s+(your\s+)?(system\s+)?prompt\b", re.I), "prompt_leakage"),
    (re.compile(r"\brepeat\s+(after\s+me|your\s+(system|instructions?))\b", re.I), "prompt_leakage"),
    (re.compile(r"\bwhat\s+(are|were)\s+your\s+(instructions?|system\s+prompt|rules?)\b", re.I), "prompt_leakage"),
    # Vietnamese equivalents
    (re.compile(r"b[oỏ]\s+qua\s+(h[uướ]?[ơờ]?ng\s+d[aẫ][ấ]n|quy\s+t[aắ][cắ])", re.I), "role_override_vi"),
    (re.compile(r"gi[aả]\s+v[oờ][ờ]\s+nh[uư]\s+b[aạ][nạ]\s+l[aà]", re.I), "role_override_vi"),
    (re.compile(r"b[aâ]y\s+gi[oờ]\s+b[aạ][nạ]\s+l[aà]", re.I), "role_override_vi"),
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
    Two-stage check: regex patterns + heuristic scoring.
    """
    try:
        # Stage 1: exact pattern match
        for pattern, category in _INJECTION_PATTERNS:
            if pattern.search(query):
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

_CITATION_WARNING_VI = (
    "\n\n_(Lưu ý: một số trích dẫn không khớp với nguồn được truy xuất và đã được xóa.)_"
)


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
