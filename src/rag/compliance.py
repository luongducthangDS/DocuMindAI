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

import calendar
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

# Số viết kiểu Việt Nam: dấu chấm tách nghìn ("1.200" = 1200), dấu phẩy thập
# phân ("0,85"). Chỉ nhóm chấm ĐÚNG 3 chữ số mới là tách nghìn — "5.31" vẫn là
# thập phân. Đọc "1.200 giờ" thành 1.2 từng cho ✅ một vụ làm thêm 1200 giờ/năm.
_NUM = r"\d{1,3}(?:\.\d{3})+(?![\d,])|\d+(?:[.,]\d+)?"
_THOUSANDS_RE = re.compile(r"\d{1,3}(?:\.\d{3})+")
# Ngày tháng ("30/4", "01/01/2026") không phải số liệu của tình huống.
_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b")


def _parse_number(raw: str) -> float:
    if _THOUSANDS_RE.fullmatch(raw):
        return float(raw.replace(".", ""))
    return float(raw.replace(",", "."))


# Tried in order: number carrying the criterion's own unit ("320 giờ" for a
# giờ/năm criterion), number near a known label keyword, number followed by any
# unit marker (%/triệu/đồng), then any number at all.
# Labels mirror `condition.field` in data/compliance/criteria.json (labour /
# social-insurance domain). Adding a criterion with a new field means adding
# its label here, otherwise extraction falls through to the generic patterns.
_NUMBER_NEAR_LABEL_RE = re.compile(
    r"(?:giờ làm thêm|làm thêm|tăng ca|thử việc|mức lương|tiền lương|lương|"
    r"ngày nghỉ|nghỉ hằng năm|nghỉ phép|thời gian đóng|tỷ lệ đóng|tuổi nghỉ hưu)"
    rf"\D{{0,20}}?({_NUM})",
    re.IGNORECASE,
)
_TRAILING_UNIT_NUMBER_RE = re.compile(rf"({_NUM})\s*(?:%|giờ|ngày|tháng|năm|triệu|đồng)")
_ANY_NUMBER_RE = re.compile(rf"({_NUM})")

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
    return _match(situation, criteria)[0]


def _match(situation: str, criteria: list[dict]) -> tuple[dict | None, dict]:
    """(tiêu chí, cách khớp). Cách khớp đi vào trace: engine này phát ✅/❌, nên khớp
    nhầm tiêu chí phải thấy được là do từ khoá (điểm bao nhiêu) hay do embedding."""
    if not criteria:
        return None, {"by": "none"}

    scored = sorted(
        ((c, _keyword_score(situation, c)) for c in criteria),
        key=lambda pair: pair[1],
        reverse=True,
    )
    best, best_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0

    if best_score >= _MIN_KEYWORD_SCORE and best_score > second_score:
        return best, {"by": "keyword", "score": best_score, "runner_up": second_score}

    return _match_by_embedding(situation, criteria), {"by": "embedding", "keyword_best": best_score}


def _extract_value_via_llm(situation: str, condition: dict) -> float | None:
    try:
        from src.rag.generator import gemini_generate

        prompt = (
            f"Trích xuất giá trị số liên quan đến '{condition.get('field')}' "
            f"({condition.get('unit', '')}) từ câu sau. Chỉ trả về một số duy nhất, "
            "không giải thích, không kèm chữ nào khác. Nếu không có số nào liên quan, "
            f"trả về đúng chữ 'none'.\n\nCâu: {situation}"
        )
        raw = (gemini_generate(prompt, name="extract-compliance-value") or "").strip()
        if raw.lower() == "none":
            return None
        match = _ANY_NUMBER_RE.search(raw)
        return _parse_number(match.group(1)) if match else None
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
    situation = _DATE_RE.sub(" ", situation)
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

    if condition.get("unit") == "%" and "%" not in situation:
        amounts = _money_amounts(situation)
        if len(amounts) >= 2:
            return _wage_ratio_percent(situation, amounts)
        if amounts:
            # "lương thử việc 9 triệu" đem so với ngưỡng 85% là so tiền với phần trăm.
            raise AmbiguousValue(
                "Câu hỏi chỉ nêu một số tiền. Cần cả lương thử việc và lương chính thức "
                "(hoặc tỷ lệ %) mới so được với ngưỡng."
            )
    if condition.get("unit") == "ngày":
        days = _days_from_other_units(situation, condition)
        if days is not None:
            return days

    unit_word = re.split(r"[\s/]", condition.get("unit", ""), maxsplit=1)[0]
    own_unit_re = re.compile(rf"({_NUM})\s*{re.escape(unit_word)}", re.IGNORECASE) if unit_word else None
    for pattern in (own_unit_re, _NUMBER_NEAR_LABEL_RE, _TRAILING_UNIT_NUMBER_RE, _ANY_NUMBER_RE):
        match = pattern.search(situation) if pattern else None
        if match:
            try:
                value = _parse_number(match.group(1))
            except ValueError:
                continue
            # "lương 4500000" written without separators — still đồng, not triệu.
            if money and value >= 1000:
                value /= 1_000_000
            return value
    return _extract_value_via_llm(situation, condition)


class AmbiguousValue(ValueError):
    """Câu hỏi có số liệu nhưng không đủ để phán — check_compliance trả
    insufficient_info kèm lý do, KHÔNG rơi xuống LLM đoán số."""


# ── F1: thời gian ghi bằng tháng/tuần/năm, tiêu chí tính bằng ngày ──────────

_DAYS_PER_UNIT = {"ngày": 1, "tuần": 7, "tháng": 30, "năm": 365}
# "2 tháng 15 ngày" = một khoảng; phần "ngày" đi liền sau được cộng vào.
_TIME_SPAN_RE = re.compile(
    rf"({_NUM})\s*(tháng|tuần|năm)(?![a-zà-ỹ])(?:\s*(?:và|,)?\s*({_NUM})\s*ngày)?", re.IGNORECASE
)


def normalize_time_unit(value: float, unit: str) -> float:
    """Quy về ngày: 1 tuần = 7, 1 tháng = 30, 1 năm = 365 ngày."""
    return value * _DAYS_PER_UNIT[unit.lower()]


def _calendar_span(value: float, unit: str) -> tuple[float, float]:
    """Ít/nhiều ngày nhất mà `value` tháng/năm dương lịch thực sự dài.

    Luật tính thử việc theo NGÀY (Điều 25), còn "2 tháng" dài 59–62 ngày tuỳ
    tháng bắt đầu — quy ước 30 ngày/tháng không đủ để phán ở sát ngưỡng.
    """
    unit = unit.lower()
    if unit == "năm" and value.is_integer():
        return 365 * value, 366 * value
    if unit != "tháng" or not value.is_integer():
        exact = normalize_time_unit(value, unit)
        return exact, exact
    months = int(value)
    spans = [
        sum(calendar.monthrange(year + (start + k) // 12, (start + k) % 12 + 1)[1] for k in range(months))
        for year in (2023, 2024) for start in range(12)
    ]
    return min(spans), max(spans)


def _days_from_other_units(situation: str, condition: dict) -> float | None:
    """Số ngày khi câu ghi bằng tháng/tuần/năm; None khi câu ghi thẳng bằng ngày."""
    match = _TIME_SPAN_RE.search(situation)
    if not match:
        return None
    value, unit = _parse_number(match.group(1)), match.group(2)
    extra = _parse_number(match.group(3)) if match.group(3) else 0.0
    lo, hi = _calendar_span(value, unit)
    if evaluate_condition(condition, lo + extra) != evaluate_condition(condition, hi + extra):
        raise AmbiguousValue(
            f"{match.group(0).strip()} dài từ {lo + extra:g} đến {hi + extra:g} ngày tuỳ ngày bắt đầu, "
            f"trong khi ngưỡng tính theo ngày ({condition['operator']} {condition['value']:g} ngày). "
            "Vui lòng nêu số ngày cụ thể."
        )
    return normalize_time_unit(value, unit) + extra


# ── F2: lương thử việc cho bằng hai số tiền ───────────────────────────────────

_MONEY_RE = re.compile(
    rf"(?P<dong>\d{{1,3}}(?:[.,]\d{{3}}){{2,}}|\d{{7,}})|(?P<trieu>{_NUM})\s*(?:triệu|tr(?![a-zà-ỹ]))",
    re.IGNORECASE,
)
_WAGE_LABEL_RE = re.compile(r"(thử việc)|(chính thức)", re.IGNORECASE)


def _money_amounts(situation: str) -> list[tuple[int, int, float]]:
    """(start, end, triệu đồng) của mọi số tiền trong câu, theo thứ tự xuất hiện."""
    out = []
    for m in _MONEY_RE.finditer(situation):
        amount = _to_millions(m.group("dong")) if m.group("dong") else _parse_number(m.group("trieu"))
        out.append((m.start(), m.end(), amount))
    return out


def _wage_ratio_percent(situation: str, amounts: list[tuple[int, int, float]]) -> float:
    """Lương thử việc / lương chính thức, theo %.

    Mỗi số tiền nhận vai của nhãn ("thử việc"/"chính thức") gần nó nhất, ghép
    tham lam theo khoảng cách — đúng cả "thử việc 9 triệu, chính thức 10 triệu",
    "10 triệu lương chính thức, thử việc 9 triệu" lẫn "thử việc 9tr / 10tr"
    (số còn lại nhận vai còn thiếu). Không xác định được thì không đoán.
    """
    labels = [(m.start(), m.end(), "probation" if m.group(1) else "official")
              for m in _WAGE_LABEL_RE.finditer(situation)]
    pairs = sorted(
        (max(ls - ae, as_ - le), i, role)
        for i, (as_, ae, _) in enumerate(amounts) for ls, le, role in labels
    )
    role_of: dict[int, str] = {}
    for _, i, role in pairs:
        if i not in role_of and role not in role_of.values():
            role_of[i] = role
    if len(role_of) == 1 and len(amounts) == 2:
        (other,) = {0, 1} - set(role_of)
        role_of[other] = ({"probation", "official"} - set(role_of.values())).pop()
    by_role = {role: amounts[i][2] for i, role in role_of.items()}
    if set(by_role) != {"probation", "official"} or by_role["official"] <= 0:
        raise AmbiguousValue(
            "Không xác định được đâu là lương thử việc, đâu là lương chính thức. "
            "Vui lòng nêu rõ, ví dụ: lương thử việc 9 triệu, lương chính thức 10 triệu."
        )
    return round(by_role["probation"] * 100 / by_role["official"], 6)


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


def resolve_version(criterion: dict, as_of: str) -> dict | None:
    """The criterion as it applies on `as_of` (ISO date): the top-level entry or
    one of its `versions`, each valid on [effective_from, effective_to).
    None when no version covers that date.

    A version overrides the top-level fields it lists; `value` replaces the
    threshold in `condition`. Without this the engine judged a 2025 wage
    against the 2026 minimum (DATA_QC C-01).
    """
    for version in (criterion, *criterion.get("versions", [])):
        if version.get("effective_from", "") <= as_of < (version.get("effective_to") or "9999-12-31"):
            if version is criterion:
                return criterion
            merged = {**criterion, **{k: v for k, v in version.items() if k != "value"}}
            merged["condition"] = {**criterion["condition"], "value": version["value"]}
            return merged
    return None


def check_compliance(situation: str, as_of_date: str | None = None) -> dict:
    """Orchestrates match -> extract -> evaluate. Returns:
    {matched, criterion_id, verdict, explanation, citation, extracted_value, match}
    where verdict is one of "pass" | "fail" | "insufficient_info" | "no_match",
    and `match` says how the criterion was picked (by, score, keywords) — for the trace.
    `as_of_date` (ISO) picks the criterion version in force; default today.
    """
    from src.rag.temporal import today_iso

    as_of = as_of_date or today_iso()
    criterion, how = _match(situation, load_criteria())
    if criterion is not None:
        situation_lower = situation.lower()
        how["keywords"] = [kw for kw in criterion.get("keywords", []) if kw.lower() in situation_lower]
    return {**_verdict(situation, criterion, as_of), "match": how}


def _verdict(situation: str, criterion: dict | None, as_of: str) -> dict:
    if criterion is not None:
        in_force = resolve_version(criterion, as_of)
        if in_force is None:
            return {
                "matched": True,
                "criterion_id": criterion["id"],
                "verdict": "insufficient_info",
                "explanation": (
                    f"Dữ liệu hiện có không có mức áp dụng cho tiêu chí ({criterion['topic']}) "
                    f"tại thời điểm {as_of}, nên không kết luận được."
                ),
                "citation": {},
                "extracted_value": None,
            }
        criterion = in_force
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
    try:
        value = extract_situation_value(situation, condition)
    except AmbiguousValue as exc:
        return {
            "matched": True,
            "criterion_id": criterion["id"],
            "verdict": "insufficient_info",
            "explanation": str(exc),
            "citation": _citation(criterion),
            "extracted_value": None,
        }
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

    # "bằng 0,85 lương chính thức" là tỷ lệ 85%, không phải 0,85%.
    # ponytail: chỉ quy đổi khi câu không có "%" và số ≤ 1 (hai số tiền đã được
    # _wage_ratio_percent quy ra % trước đó).
    if condition.get("unit") == "%" and value <= 1 and "%" not in situation:
        value = round(value * 100, 6)

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
