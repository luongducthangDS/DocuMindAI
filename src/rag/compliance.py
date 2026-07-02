"""
Structured compliance-check ("kiểm định tuân thủ") for narrow, quantifiable
pass/fail regulation lookups (GPA thresholds, điểm rèn luyện brackets, etc.).

This is a companion to the RAG pipeline, NOT a replacement or general rule
engine — it only covers the small set of criteria hand-curated in
data/compliance/criteria.json, each cross-referenced against the source
UNETI document at authoring time. Anything not matched falls back to normal
RAG (see compliance_check_node in src/agent/graph.py).
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache

from loguru import logger

from src.config import get_settings

_OPERATORS = {
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "==": lambda a, b: a == b,
}

# Tried in order: number immediately near a known label keyword, then a number
# followed by a unit marker (điểm/%), then any number at all.
_NUMBER_NEAR_LABEL_RE = re.compile(
    r"(?:điểm rèn luyện|điểm trung bình chung tích lũy|điểm trung bình chung|"
    r"gpa|đtb|điểm)\D{0,20}?(\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)
_TRAILING_UNIT_NUMBER_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:điểm|%)")
_ANY_NUMBER_RE = re.compile(r"(\d+(?:[.,]\d+)?)")

# Below this cosine similarity, an embedding match is considered noise rather
# than a real match — situation is unrelated to any known criterion.
_EMBEDDING_MATCH_THRESHOLD = 0.45


@lru_cache
def load_criteria() -> list[dict]:
    path = get_settings().data_dir / "compliance" / "criteria.json"
    if not path.exists():
        logger.warning("Compliance criteria file not found: {}", path)
        return []
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _keyword_score(situation: str, criterion: dict) -> int:
    situation_lower = situation.lower()
    return sum(1 for kw in criterion.get("keywords", []) if kw.lower() in situation_lower)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _match_by_embedding(situation: str, criteria: list[dict]) -> dict | None:
    try:
        from src.rag.embedder import get_embedder

        embedder = get_embedder()
        situation_vec = embedder.get_query_embedding(situation)
        best_criterion = None
        best_sim = -1.0
        for c in criteria:
            topic_vec = embedder.get_query_embedding(c["topic"])
            sim = _cosine_similarity(situation_vec, topic_vec)
            if sim > best_sim:
                best_sim = sim
                best_criterion = c
        return best_criterion if best_sim >= _EMBEDDING_MATCH_THRESHOLD else None
    except Exception as exc:
        logger.warning("Embedding-based compliance match failed: {}", exc)
        return None


def match_criteria(situation: str, criteria: list[dict]) -> dict | None:
    """Keyword overlap first (cheap, deterministic); falls back to embedding
    similarity against each criterion's `topic` when keywords are ambiguous
    or absent."""
    if not criteria:
        return None

    scored = sorted(
        ((c, _keyword_score(situation, c)) for c in criteria),
        key=lambda pair: pair[1],
        reverse=True,
    )
    best, best_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0

    if best_score > 0 and best_score > second_score:
        return best

    return _match_by_embedding(situation, criteria)


def _extract_value_via_llm(situation: str, condition: dict) -> float | None:
    settings = get_settings()
    if not settings.groq_api_key:
        return None
    try:
        from src.rag.generator import _get_groq_client

        client = _get_groq_client()
        prompt = (
            f"Trích xuất giá trị số liên quan đến '{condition.get('field')}' "
            f"({condition.get('unit', '')}) từ câu sau. Chỉ trả về một số duy nhất, "
            "không giải thích, không kèm chữ nào khác. Nếu không có số nào liên quan, "
            f"trả về đúng chữ 'none'.\n\nCâu: {situation}"
        )
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.lower() == "none":
            return None
        match = _ANY_NUMBER_RE.search(raw)
        return float(match.group(1).replace(",", ".")) if match else None
    except Exception as exc:
        logger.warning("LLM value extraction failed: {}", exc)
        return None


def extract_situation_value(situation: str, condition: dict) -> float | None:
    """Pull the numeric value relevant to `condition['field']` out of
    free-form Vietnamese phrasing. Regex first (deterministic, cheapest);
    escalates to one LLM extraction call only if regex finds nothing."""
    for pattern in (_NUMBER_NEAR_LABEL_RE, _TRAILING_UNIT_NUMBER_RE, _ANY_NUMBER_RE):
        match = pattern.search(situation)
        if match:
            try:
                return float(match.group(1).replace(",", "."))
            except ValueError:
                continue
    return _extract_value_via_llm(situation, condition)


def evaluate_condition(condition: dict, value: float) -> bool:
    op = _OPERATORS.get(condition["operator"])
    if op is None:
        raise ValueError(f"Unsupported operator: {condition['operator']}")
    return op(value, condition["value"])


def _citation(criterion: dict) -> dict:
    return {
        "so_hieu": criterion["so_hieu"],
        "dieu_khoan": criterion["dieu_khoan"],
        "source_url": criterion.get("source_url", ""),
        "title": criterion.get("topic", ""),
    }


def check_compliance(situation: str) -> dict:
    """Orchestrates match -> extract -> evaluate. Returns:
    {matched, criterion_id, verdict, explanation, citation, extracted_value}
    where verdict is one of "pass" | "fail" | "insufficient_info" | "no_match".
    """
    criteria = load_criteria()
    criterion = match_criteria(situation, criteria)
    if criterion is None:
        return {
            "matched": False,
            "criterion_id": "",
            "verdict": "no_match",
            "explanation": "",
            "citation": {},
            "extracted_value": None,
        }

    condition = criterion["condition"]
    value = extract_situation_value(situation, condition)
    if value is None:
        return {
            "matched": True,
            "criterion_id": criterion["id"],
            "verdict": "insufficient_info",
            "explanation": (
                f"Đã xác định tiêu chí liên quan ({criterion['topic']}) nhưng không trích xuất "
                "được số liệu cụ thể từ câu hỏi. Vui lòng nêu rõ con số (ví dụ điểm số, phần trăm)."
            ),
            "citation": _citation(criterion),
            "extracted_value": None,
        }

    passed = evaluate_condition(condition, value)
    template = criterion["verdict_template"]["pass" if passed else "fail"]
    explanation = template.format(dieu_khoan=criterion["dieu_khoan"], so_hieu=criterion["so_hieu"])
    return {
        "matched": True,
        "criterion_id": criterion["id"],
        "verdict": "pass" if passed else "fail",
        "explanation": explanation,
        "citation": _citation(criterion),
        "extracted_value": value,
    }
