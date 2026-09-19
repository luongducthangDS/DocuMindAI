"""
Structured compliance-check ("kiểm định tuân thủ") for narrow, quantifiable
pass/fail regulation lookups (interest-rate thresholds, credit-limit
brackets, income conditions, etc.).

This is a companion to the RAG pipeline, NOT a replacement or general rule
engine — it only covers the small set of criteria hand-curated in
data/compliance/criteria.json, each cross-referenced against the source
labour/social-insurance document at authoring time. Anything not matched falls
back to normal RAG (see compliance_check_node in src/agent/graph.py).
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
# followed by a unit marker (%/triệu/đồng), then any number at all.
# Labels mirror `condition.field` in data/compliance/criteria.json (labour /
# social-insurance domain). Adding a criterion with a new field means adding
# its label here, otherwise extraction falls through to the generic patterns.
_NUMBER_NEAR_LABEL_RE = re.compile(
    r"(?:giờ làm thêm|làm thêm|tăng ca|thử việc|mức lương|tiền lương|lương|"
    r"ngày nghỉ|nghỉ hằng năm|nghỉ phép|thời gian đóng|tỷ lệ đóng|tuổi nghỉ hưu)"
    r"\D{0,20}?(\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)
_TRAILING_UNIT_NUMBER_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:%|giờ|ngày|tháng|năm|triệu|đồng)")
_ANY_NUMBER_RE = re.compile(r"(\d+(?:[.,]\d+)?)")

# Below this cosine similarity, an embedding match is considered noise rather
# than a real match — situation is unrelated to any known criterion.
_EMBEDDING_MATCH_THRESHOLD = 0.45

# Minimum _keyword_score (total matched keyword length) to decide on keywords
# alone. A single generic word — "%", "ngày", "tháng" — is not evidence that the
# situation is about a given criterion, and this engine emits a ✅/❌ verdict with
# a citation, so a weak match must fall through to embeddings rather than guess.
_MIN_KEYWORD_SCORE = 6


@lru_cache
def load_criteria() -> list[dict]:
    path = get_settings().data_dir / "compliance" / "criteria.json"
    if not path.exists():
        logger.warning("Compliance criteria file not found: {}", path)
        return []
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _keyword_score(situation: str, criterion: dict) -> int:
    """Total length of the matched keywords, not their count.

    Criteria in the same family share short keywords ("thử việc", "làm thêm")
    and are told apart by a longer, more specific one ("lương thử việc",
    "trong 01 năm"). Counting matches ties those pairs and drops the decision on
    the embedding fallback; weighting by length lets the specific phrase win.
    """
    situation_lower = situation.lower()
    return sum(len(kw) for kw in criterion.get("keywords", []) if kw.lower() in situation_lower)


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

    if best_score >= _MIN_KEYWORD_SCORE and best_score > second_score:
        return best

    return _match_by_embedding(situation, criteria)


def _extract_value_via_llm(situation: str, condition: dict) -> float | None:
    try:
        from src.rag.generator import gemini_generate

        prompt = (
            f"Trích xuất giá trị số liên quan đến '{condition.get('field')}' "
            f"({condition.get('unit', '')}) từ câu sau. Chỉ trả về một số duy nhất, "
            "không giải thích, không kèm chữ nào khác. Nếu không có số nào liên quan, "
            f"trả về đúng chữ 'none'.\n\nCâu: {situation}"
        )
        raw = (gemini_generate(prompt) or "").strip()
        if raw.lower() == "none":
            return None
        match = _ANY_NUMBER_RE.search(raw)
        return float(match.group(1).replace(",", ".")) if match else None
    except Exception as exc:
        logger.warning("LLM value extraction failed: {}", exc)
        return None


def _to_millions(raw: str) -> float:
    """"5.310.000" / "5,310,000" -> 5.31 (triệu đồng)."""
    return float(re.sub(r"[.,]", "", raw)) / 1_000_000


def extract_situation_value(situation: str, condition: dict) -> float | None:
    """Pull the numeric value relevant to `condition['field']` out of
    free-form Vietnamese phrasing. Regex first (deterministic, cheapest);
    escalates to one LLM extraction call only if regex finds nothing.

    Money criteria are stored in "triệu đồng" but people write the amount both
    ways ("4,5 triệu" and "4.500.000 đồng"), so a full amount is matched first
    and converted — otherwise the generic patterns read "4.500.000 đồng" as the
    trailing group "000".
    """
    money = "đồng" in condition.get("unit", "")
    if money:
        # "5.310.000" / "5310000" — a raw amount in đồng, converted to triệu.
        # Tried before the label patterns, which would otherwise read the "1"
        # out of "lương tối thiểu vùng 1" as the amount.
        full_amount = re.search(r"(\d{1,3}(?:[.,]\d{3}){2,})|(\d{7,})", situation)
        if full_amount:
            return _to_millions(full_amount.group(0))
        in_millions = re.search(
            r"(\d+(?:[.,]\d+)?)\s*(?:triệu|tr(?![a-zà-ỹ]))", situation, re.IGNORECASE
        )
        if in_millions:
            return float(in_millions.group(1).replace(",", "."))

    for pattern in (_NUMBER_NEAR_LABEL_RE, _TRAILING_UNIT_NUMBER_RE, _ANY_NUMBER_RE):
        match = pattern.search(situation)
        if match:
            try:
                value = float(match.group(1).replace(",", "."))
            except ValueError:
                continue
            # "lương 4500000" written without separators — still đồng, not triệu.
            if money and value >= 1000:
                value /= 1_000_000
            return value
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
