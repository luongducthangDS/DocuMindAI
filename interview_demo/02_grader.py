"""
Self-Correction Grader: LLM-as-judge relevance check

Giải thích trong interview:
- Sau khi retrieve, không phải lúc nào chunk cũng relevant
- Two-tier: heuristic (rẻ) → LLM judge (chỉ khi cần)
- Fail open: lỗi LLM → coi là relevant (không bao giờ block câu trả lời)

Production version: src/rag/grader.py (~200 lines)
"""

import json
import os
import re

import google.generativeai as genai


def grade_chunks(query: str, chunks: list[dict], best_score: float) -> dict:
    """
    Đánh giá chunks có đủ relevant để trả lời query không.

    Returns: {"relevant": bool, "reason": str}

    Two-tier approach để tối ưu chi phí LLM:
    1. Nếu best_score rất cao → skip LLM, trả lời ngay
    2. Nếu score mơ hồ → hỏi LLM phán xét
    """
    if not chunks:
        return {"relevant": False, "reason": "Không tìm thấy chunk nào"}

    # Tier 1: Heuristic — score cao rõ ràng → relevant, không cần LLM
    CONFIDENT_THRESHOLD = 0.05 * 2.0  # 2× abstain threshold
    if best_score >= CONFIDENT_THRESHOLD:
        return {"relevant": True, "reason": f"Score cao ({best_score:.3f}), không cần LLM judge"}

    # Tier 2: LLM-as-judge — score mơ hồ, nhờ LLM phán xét
    return _llm_judge(query, chunks)


def _llm_judge(query: str, chunks: list[dict]) -> dict:
    """
    Nhờ LLM phán xét: chunks có đủ trả lời query không?
    Fail open: lỗi → trả {"relevant": True} để không block câu trả lời.
    """
    excerpts = "\n\n".join(
        f"[{i+1}] {c['text'][:300]}" for i, c in enumerate(chunks[:5])
    )

    prompt = f"""Câu hỏi: {query}

Các đoạn văn bản tìm được:
{excerpts}

Các đoạn trên có đủ thông tin để trả lời câu hỏi không?
Trả về JSON một dòng: {{"relevant": true/false, "reason": "lý do ngắn"}}"""

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        raw = response.text or ""

        # Parse JSON từ response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return {
                "relevant": bool(data.get("relevant")),
                "reason": str(data.get("reason", ""))[:200],
            }
    except Exception as e:
        # Fail open: LLM lỗi → không block câu trả lời
        print(f"⚠ LLM judge failed ({e}), failing open")

    return {"relevant": True, "reason": "LLM judge unavailable — fail open"}


def demo_grader():
    """Demo self-correction loop trong interview."""
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        print("Set GOOGLE_API_KEY để chạy demo LLM judge")
        print("Chạy demo heuristic tier thôi:\n")
        _demo_heuristic()
        return

    genai.configure(api_key=api_key)

    print("=== Demo Self-Correction Grader ===\n")

    # Case 1: Score cao → heuristic tier, skip LLM
    print("Case 1: Score cao (reranker trả 0.95)")
    result = grade_chunks(
        query="Điều kiện hưởng lương hưu là gì?",
        chunks=[{"text": "Điều 54. Điều kiện hưởng lương hưu: đủ 15 năm đóng BHXH..."}],
        best_score=0.95,
    )
    print(f"  → relevant={result['relevant']}, reason='{result['reason']}'")
    print(f"  → Tier: HEURISTIC (không tốn LLM call)\n")

    # Case 2: Score thấp → LLM judge
    print("Case 2: Score thấp (RRF score ~0.016, không có reranker)")
    result = grade_chunks(
        query="Mức phạt khi trả lương chậm là bao nhiêu?",
        chunks=[
            {"text": "Điều 94. Nguyên tắc trả lương: Người sử dụng lao động phải trả lương trực tiếp..."},
            {"text": "Điều 97. Tiền lương làm thêm giờ: Ít nhất bằng 150% tiền lương giờ bình thường..."},
        ],
        best_score=0.016,  # RRF score khi không có reranker
    )
    print(f"  → relevant={result['relevant']}, reason='{result['reason']}'")
    tier = "LLM JUDGE" if 0.016 < 0.05 * 2.0 else "HEURISTIC"
    print(f"  → Tier: {tier}")


def _demo_heuristic():
    """Demo không cần API key."""
    test_cases = [
        (0.95, True, "reranker score cao"),
        (0.016, False, "RRF score thấp, cần LLM judge"),
        (0.0, False, "không có reranker, score = 0"),
    ]
    for score, expected_heuristic, desc in test_cases:
        threshold = 0.05 * 2.0
        heuristic_ok = score >= threshold
        print(f"  score={score:.3f} ({desc})")
        print(f"    → heuristic: {'PASS' if heuristic_ok else 'FAIL → cần LLM judge'}")
        print()


if __name__ == "__main__":
    demo_grader()
