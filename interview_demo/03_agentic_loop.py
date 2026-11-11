"""
Agentic RAG Loop: LLM quyết định đủ hay tìm thêm

Đây là điểm khác biệt chính của DocuMind so với RAG thông thường.

Vấn đề RAG thông thường gặp với pháp luật VN:
  Query: "Điều kiện điều chỉnh lương hưu dựa trên cơ sở nào?"
  Retrieve: Điều 23 — "điều chỉnh lương hưu theo Điều 67..."
  Problem: Điều 23 chỉ dẫn chiếu, không có câu trả lời thực sự.
           Cần lấy thêm Điều 67 mới có nội dung.

Giải pháp: LLM đọc những gì đã có, quyết định:
  - "enough": đủ → trả lời
  - "fetch": lấy điều luật được dẫn chiếu
  - "search": tìm kiếm thêm với sub-query
  - "versions": lấy phiên bản cũ/mới để so sánh

Production version: src/agent/research.py (~300 lines)
"""

import json
import os
import re

import google.generativeai as genai


DECISION_PROMPT = """Bạn là trợ lý tra cứu pháp luật. Nhiệm vụ: quyết định các đoạn đã tìm được
có ĐỦ để trả lời câu hỏi chưa. Chưa trả lời câu hỏi.

Câu hỏi: {query}

Các đoạn đã có:
{excerpts}

Các điều được dẫn chiếu nhưng chưa có nội dung:
{candidates}

Quy tắc:
- Nếu câu trả lời phụ thuộc nội dung một điều trong danh sách dẫn chiếu → chọn vào "fetch"
- Nếu còn thiếu thông tin không có trong dẫn chiếu → viết câu tìm kiếm vào "search"
- Nếu đã đủ → "enough": true

Trả về JSON một dòng:
{{"enough": true/false, "fetch": ["R1"], "search": ["câu tìm kiếm"], "reason": "lý do"}}"""


def agentic_research(
    query: str,
    chunks: list[dict],
    referenced_articles: list[dict],
    max_rounds: int = 2,
) -> tuple[list[dict], list[str]]:
    """
    Vòng lặp agentic: LLM quyết định đủ hay tìm thêm.

    Args:
        query: câu hỏi gốc
        chunks: các chunk đã retrieve được
        referenced_articles: điều luật được dẫn chiếu trong chunks nhưng chưa có nội dung
        max_rounds: số vòng tối đa (tránh infinite loop)

    Returns:
        (chunks cuối cùng, các bước đã thực hiện)
    """
    steps = []

    for round_num in range(max_rounds):
        print(f"\n--- Round {round_num + 1} ---")

        # LLM quyết định
        decision = _decide(query, chunks, referenced_articles)
        if decision is None:
            print("  LLM unavailable, dùng những gì đã có")
            break

        print(f"  Decision: enough={decision['enough']}, "
              f"fetch={decision['fetch']}, search={decision['search']}")
        print(f"  Reason: {decision['reason']}")

        # Nếu đủ và không có hành động → dừng
        if decision["enough"] and not decision["fetch"] and not decision["search"]:
            steps.append(f"Round {round_num+1}: Đánh giá đủ thông tin")
            break

        added = []

        # Fetch: lấy điều luật được dẫn chiếu
        for code in decision.get("fetch", []):
            article = next((a for a in referenced_articles if a["code"] == code), None)
            if article:
                new_chunk = _fetch_article(article)
                chunks.append(new_chunk)
                added.append(f"Lấy {article['name']}")
                print(f"  ✓ Fetched: {article['name']}")

        # Search: tìm kiếm thêm
        for sub_query in decision.get("search", [])[:1]:  # max 1 sub-query
            new_chunks = _search(sub_query)
            chunks.extend(new_chunks)
            added.append(f"Tìm thêm: '{sub_query}'")
            print(f"  ✓ Searched: '{sub_query}' → {len(new_chunks)} chunks")

        if added:
            steps.append(f"Round {round_num+1}: {'; '.join(added)}")

        if not added or decision["enough"]:
            break

    return chunks, steps


def _decide(query: str, chunks: list[dict], candidates: list[dict]) -> dict | None:
    """Nhờ LLM quyết định đủ chưa. Fail open nếu lỗi."""
    excerpts = "\n".join(
        f"[{i+1}] {c['text'][:300]}" for i, c in enumerate(chunks)
    )
    cand_lines = "\n".join(
        f"{a['code']}. {a['name']} (được dẫn từ [{a['cited_from']}])"
        for a in candidates
    )

    prompt = DECISION_PROMPT.format(
        query=query,
        excerpts=excerpts or "(không có)",
        candidates=cand_lines or "(không có)",
    )

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        raw = response.text or ""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            as_list = lambda v: [str(x).strip() for x in (v if isinstance(v, list) else [v] if v else [])]
            return {
                "enough": bool(data.get("enough")),
                "fetch": as_list(data.get("fetch")),
                "search": as_list(data.get("search"))[:1],
                "reason": str(data.get("reason", ""))[:200],
            }
    except Exception as e:
        print(f"  ⚠ LLM decision failed: {e}")

    return None


def _fetch_article(article: dict) -> dict:
    """Lấy nội dung điều luật từ vector store (giả lập)."""
    # Production: query vector store với clause_uid cụ thể
    return {
        "text": article.get("content", f"Nội dung của {article['name']}"),
        "metadata": {"source": article["name"], "agent_step": "fetch"},
    }


def _search(sub_query: str) -> list[dict]:
    """Tìm kiếm thêm với sub-query (giả lập)."""
    # Production: gọi retrieve_with_context(sub_query, ctx)
    return [{"text": f"Kết quả tìm kiếm cho: {sub_query}", "metadata": {"agent_query": sub_query}}]


def demo_agentic_loop():
    """
    Demo vòng lặp agentic với bài toán multi-hop thực tế.

    Câu hỏi: "Điều kiện điều chỉnh lương hưu dựa trên cơ sở nào?"
    Điều 23 chỉ nói "theo Điều 67" → cần fetch Điều 67
    """
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        print("Set GOOGLE_API_KEY để chạy demo")
        _demo_without_api()
        return

    genai.configure(api_key=api_key)

    print("=== Demo Agentic RAG Loop ===")
    print("Bài toán: multi-hop reasoning trong pháp luật VN\n")

    query = "Điều kiện điều chỉnh lương hưu dựa trên cơ sở nào?"

    # Lượt retrieve đầu: chỉ lấy được Điều 23 (dẫn chiếu, không có nội dung)
    initial_chunks = [
        {
            "text": (
                "Điều 23. Điều chỉnh lương hưu\n"
                "1. Lương hưu được điều chỉnh theo quy định tại Điều 67 của Luật này.\n"
                "2. Chính phủ quy định chi tiết điều này."
            ),
            "metadata": {"dieu": 23, "clause_uid": "bhxh_41_2024__d23"},
        }
    ]

    # Đồ thị dẫn chiếu: Điều 23 dẫn tới Điều 67
    referenced = [
        {
            "code": "R1",
            "name": "Điều 67. Điều chỉnh lương hưu",
            "cited_from": 1,  # từ chunk [1]
            "content": (
                "Điều 67. Điều chỉnh lương hưu\n"
                "Lương hưu được điều chỉnh trên cơ sở:\n"
                "a) Chỉ số giá tiêu dùng;\n"
                "b) Tăng trưởng kinh tế;\n"
                "c) Khả năng của quỹ bảo hiểm xã hội;\n"
                "d) Ngân sách nhà nước."
            ),
        }
    ]

    print(f"Query: '{query}'\n")
    print("Lượt retrieve đầu:")
    print(f"  → {len(initial_chunks)} chunk: Điều 23 (dẫn chiếu Điều 67, không có nội dung)")
    print(f"  → Đồ thị dẫn chiếu: Điều 23 → Điều 67\n")

    final_chunks, steps = agentic_research(
        query=query,
        chunks=initial_chunks,
        referenced_articles=referenced,
        max_rounds=2,
    )

    print(f"\n=== Kết quả ===")
    print(f"Chunks ban đầu: {len(initial_chunks)} → Chunks cuối: {len(final_chunks)}")
    print(f"Các bước agent thực hiện: {steps}")
    print(f"\nChunks trong context khi trả lời:")
    for i, c in enumerate(final_chunks, 1):
        agent_step = c["metadata"].get("agent_step", "retrieve")
        print(f"  [{i}] ({agent_step}) {c['text'][:80]}...")


def _demo_without_api():
    """Giải thích logic mà không cần API key."""
    print("=== Agentic RAG Logic (không cần API) ===\n")

    print("Vấn đề RAG thông thường với pháp luật VN:")
    print("  Query: 'Điều kiện điều chỉnh lương hưu dựa trên cơ sở nào?'")
    print("  Retrieve: Điều 23 — 'theo quy định tại Điều 67...'")
    print("  → Chunk này KHÔNG có câu trả lời, chỉ có dẫn chiếu")
    print("  → LLM sẽ trả lời 'không tìm thấy' hoặc hallucinate\n")

    print("Giải pháp Agentic RAG:")
    print("  Sau retrieve, LLM đọc Điều 23 và thấy 'theo Điều 67'")
    print("  LLM quyết định: {'enough': false, 'fetch': ['R1'], 'reason': 'cần nội dung Điều 67'}")
    print("  System fetch Điều 67: 'trên cơ sở chỉ số giá tiêu dùng, tăng trưởng kinh tế...'")
    print("  LLM quyết định: {'enough': true}")
    print("  → Câu trả lời đúng và đầy đủ\n")

    print("Kết quả đo được:")
    print("  Multi-hop accuracy: 0.913 → 1.000 sau khi thêm agentic loop")


if __name__ == "__main__":
    demo_agentic_loop()
