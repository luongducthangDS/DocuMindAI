"""
LangGraph Agent Flow — Full workflow của DocuMind AI

Giải thích trong interview:
- Mỗi "node" là một bước xử lý rõ ràng, dễ debug
- "edge" định nghĩa điều kiện chuyển giữa các node
- State (AgentState) chứa toàn bộ dữ liệu xuyên suốt workflow
- Vòng lặp grade → reformulate → retrieve bị giới hạn (MAX_RETRIES=1)

Production version: src/agent/graph.py (~1100 lines)
Demo này giữ đúng architecture nhưng bỏ: temporal filter, compliance check,
report generation, FPT AI fallback, Langfuse tracing.
"""

import os
from typing import Annotated, Literal, TypedDict

import google.generativeai as genai
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

MAX_RETRIES = 1  # giới hạn vòng lặp grade → reformulate


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Toàn bộ dữ liệu xuyên suốt workflow. Mỗi node chỉ update field nó cần."""
    messages: Annotated[list, add_messages]
    query: str
    intent: Literal["simple_qa", "compare", "summarize", "unknown"]
    retrieved_chunks: list[dict]
    answer: str
    sources: list[dict]
    retry_count: int
    grade: Literal["relevant", "irrelevant", "unknown"]
    grade_reason: str
    steps: list[dict]  # [{label, detail}] — visible reasoning steps


# ── Nodes ─────────────────────────────────────────────────────────────────────

def contextualize_node(state: AgentState) -> dict:
    """
    Rewrite câu hỏi follow-up thành câu độc lập.

    Ví dụ:
      Lượt 1: "Điều kiện hưởng lương hưu là gì?"
      Lượt 2: "Còn BHXH tự nguyện thì sao?" → "Điều kiện hưởng lương hưu BHXH tự nguyện là gì?"

    Nếu là lượt đầu (không có history) → skip LLM, trả nguyên.
    """
    messages = state.get("messages", [])
    query = state["query"]
    history = messages[:-1]  # bỏ message hiện tại

    if not history:
        # Lượt đầu: không cần rewrite
        return {"steps": [{"label": "Phân tích câu hỏi", "detail": "Câu hỏi độc lập"}]}

    # Có history: nhờ LLM rewrite
    history_text = "\n".join(
        f"{'User' if getattr(m, 'type', '') == 'human' else 'AI'}: {getattr(m, 'content', '')[:200]}"
        for m in history[-4:]  # 2 lượt gần nhất
    )
    prompt = f"""Lịch sử hội thoại:
{history_text}

Câu hỏi cuối: {query}

Viết lại câu hỏi thành câu độc lập, đầy đủ ý nghĩa. Chỉ trả về câu đã viết lại."""

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        rewritten = model.generate_content(prompt).text.strip()
        if rewritten and rewritten != query:
            return {
                "query": rewritten,
                "steps": [{"label": "Diễn giải câu hỏi", "detail": rewritten}],
            }
    except Exception:
        pass  # fallback: dùng câu gốc

    return {"steps": []}


def router_node(state: AgentState) -> dict:
    """
    Phân loại intent: simple_qa / compare / summarize / unknown.

    Fast-path: keyword matching trước, chỉ gọi LLM khi cần.
    """
    query = state["query"].lower()

    # Fast-path keyword classification
    if any(w in query for w in ["so sánh", "khác nhau", "giống nhau"]):
        return {"intent": "compare", "steps": state.get("steps", []) + [
            {"label": "Phân loại câu hỏi", "detail": "So sánh văn bản"}
        ]}
    if any(w in query for w in ["tóm tắt", "nội dung chính"]):
        return {"intent": "summarize", "steps": state.get("steps", []) + [
            {"label": "Phân loại câu hỏi", "detail": "Tóm tắt văn bản"}
        ]}

    # Default: simple_qa
    return {"intent": "simple_qa", "steps": state.get("steps", []) + [
        {"label": "Phân loại câu hỏi", "detail": "Tra cứu quy định"}
    ]}


def retrieve_node(state: AgentState) -> dict:
    """
    Hybrid retrieval: Dense + BM25 + RRF + Reranker.
    Demo: giả lập kết quả để không cần setup vector store.
    """
    query = state["query"]
    print(f"  [Retrieve] Query: '{query[:60]}...'")

    # Production: hybrid_retriever.retrieve(query)
    # Demo: giả lập chunks
    mock_chunks = [
        {
            "text": f"Điều 54. Điều kiện hưởng lương hưu: Người lao động được hưởng lương hưu khi đủ điều kiện về tuổi và thời gian đóng bảo hiểm xã hội theo quy định...",
            "score": 0.92,
            "metadata": {"title": "Luật BHXH 41/2024", "dieu_header": "Điều 54"},
        },
        {
            "text": f"Điều 64. Mức hưởng lương hưu hàng tháng: Mức hưởng lương hưu hàng tháng được tính theo tỷ lệ % tương ứng với số năm đóng bảo hiểm xã hội...",
            "score": 0.85,
            "metadata": {"title": "Luật BHXH 41/2024", "dieu_header": "Điều 64"},
        },
    ]

    return {
        "retrieved_chunks": mock_chunks,
        "steps": state.get("steps", []) + [
            {"label": "Tìm kiếm tài liệu", "detail": f"{len(mock_chunks)} đoạn liên quan"}
        ],
    }


def grade_node(state: AgentState) -> dict:
    """
    Đánh giá chunks có relevant không.
    Two-tier: heuristic → LLM judge.
    """
    chunks = state.get("retrieved_chunks", [])
    query = state["query"]

    if not chunks:
        return {"grade": "irrelevant", "grade_reason": "Không tìm thấy chunk nào"}

    best_score = max(c.get("score", 0) for c in chunks)

    # Tier 1: score cao → relevant ngay
    if best_score >= 0.5:
        return {
            "grade": "relevant",
            "grade_reason": f"Score cao ({best_score:.2f})",
            "steps": state.get("steps", []) + [
                {"label": "Đánh giá tài liệu", "detail": "Tài liệu phù hợp"}
            ],
        }

    # Tier 2: LLM judge
    try:
        excerpts = "\n".join(f"[{i+1}] {c['text'][:200]}" for i, c in enumerate(chunks[:3]))
        prompt = f"Câu hỏi: {query}\n\nChunks:\n{excerpts}\n\nChunks có đủ trả lời không? Trả về 'yes' hoặc 'no'."
        model = genai.GenerativeModel("gemini-2.0-flash")
        result = model.generate_content(prompt).text.strip().lower()
        grade = "relevant" if "yes" in result else "irrelevant"
    except Exception:
        grade = "relevant"  # fail open

    return {
        "grade": grade,
        "grade_reason": "LLM judge",
        "steps": state.get("steps", []) + [
            {"label": "Đánh giá tài liệu", "detail": f"Grade: {grade}"}
        ],
    }


def reformulate_node(state: AgentState) -> dict:
    """
    Rewrite query khi chunks không relevant, retry retrieve.
    Max 1 lần (MAX_RETRIES=1).

    Tại sao cần node này?
    - Người dùng hay dùng từ dân dã: "rút BHXH" → thuật ngữ: "hưởng BHXH một lần"
    - Retrieval lần 1 trả về chunk lệch → rewrite → retrieve lại
    """
    query = state["query"]
    tried = state.get("tried_queries", [query])

    # Rewrite bằng thuật ngữ pháp lý hơn
    reformulated = query + " (quy định pháp luật, điều khoản cụ thể)"

    return {
        "query": reformulated,
        "retry_count": state.get("retry_count", 0) + 1,
        "tried_queries": tried + [reformulated],
        "steps": state.get("steps", []) + [
            {"label": "Diễn giải lại câu hỏi", "detail": reformulated}
        ],
    }


def answer_node(state: AgentState) -> dict:
    """
    Generate câu trả lời với mandatory citations [N].
    """
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        return {
            "answer": "Tôi không tìm thấy văn bản pháp luật liên quan.",
            "sources": [],
        }

    # Build context từ chunks
    context = "\n\n".join(
        f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)
    )

    prompt = f"""Bạn là trợ lý tra cứu pháp luật lao động VN.
Trả lời câu hỏi dựa trên các đoạn văn bản bên dưới.
Mỗi câu trả lời PHẢI có trích dẫn [số] tương ứng.

{context}

Câu hỏi: {query}"""

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        answer = model.generate_content(prompt).text.strip()
    except Exception as e:
        # Fallback: trích nguyên văn khi LLM không dùng được
        answer = f"[1] {chunks[0]['text'][:500]}..."

    sources = [
        {"index": i+1, **c["metadata"]}
        for i, c in enumerate(chunks)
    ]

    return {
        "answer": answer,
        "sources": sources,
        "steps": state.get("steps", []) + [
            {"label": "Tổng hợp câu trả lời", "detail": f"{len(sources)} trích dẫn"}
        ],
    }


# ── Conditional edges ─────────────────────────────────────────────────────────

def route_by_intent(state: AgentState) -> str:
    """Sau router_node: chuyển tới retrieve hoặc answer trực tiếp."""
    intent = state.get("intent", "simple_qa")
    if intent in ("simple_qa", "compare"):
        return "retrieve"
    return "answer"  # summarize, unknown: answer thẳng


def route_after_grade(state: AgentState) -> str:
    """
    Sau grade_node: quyết định có retry không.
    - relevant → answer
    - irrelevant + retry < MAX_RETRIES → reformulate → retrieve lại
    - irrelevant + retry >= MAX_RETRIES → answer (với những gì đã có)
    """
    if state.get("grade") == "relevant":
        return "answer"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "reformulate"
    return "answer"  # hết retry, trả lời với những gì có


# ── Build Graph ───────────────────────────────────────────────────────────────

def build_graph():
    """
    Xây dựng LangGraph StateGraph.

    Topology:
        START → contextualize → router
                                  │
                        ┌─────────┴──────────┐
                      retrieve            answer (summarize)
                        │
                      grade
                        │
                ┌───────┴────────┐
              answer          reformulate
                                  │
                               retrieve (retry)
    """
    graph = StateGraph(AgentState)

    # Thêm nodes
    graph.add_node("contextualize", contextualize_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("reformulate", reformulate_node)
    graph.add_node("answer", answer_node)

    # Edges tuyến tính
    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "router")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("answer", END)

    # Conditional edges
    graph.add_conditional_edges("router", route_by_intent, {
        "retrieve": "retrieve",
        "answer": "answer",
    })
    graph.add_conditional_edges("grade", route_after_grade, {
        "answer": "answer",
        "reformulate": "reformulate",
    })

    return graph.compile()


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo_full_flow():
    """Chạy full LangGraph flow."""
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        print("Set GOOGLE_API_KEY để chạy demo\n")
        _explain_graph()
        return

    genai.configure(api_key=api_key)

    app = build_graph()

    from langchain_core.messages import HumanMessage

    test_queries = [
        "Điều kiện hưởng lương hưu là gì?",
        "So sánh BHXH bắt buộc và tự nguyện",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        result = app.invoke({
            "messages": [HumanMessage(content=query)],
            "query": query,
            "intent": "simple_qa",
            "retrieved_chunks": [],
            "answer": "",
            "sources": [],
            "retry_count": 0,
            "grade": "unknown",
            "grade_reason": "",
            "steps": [],
        })

        print("\n--- Steps ---")
        for step in result.get("steps", []):
            print(f"  [{step['label']}] {step['detail']}")

        print(f"\n--- Answer (first 300 chars) ---")
        print(result.get("answer", "")[:300])


def _explain_graph():
    """Giải thích kiến trúc LangGraph không cần API."""
    print("=== LangGraph Flow Architecture ===\n")
    print("Nodes (mỗi node = 1 step rõ ràng):")
    print("  1. contextualize_node  → rewrite follow-up query")
    print("  2. router_node         → classify: simple_qa / compare / summarize")
    print("  3. retrieve_node       → Hybrid Search (Dense + BM25 + RRF)")
    print("  4. grade_node          → LLM-as-judge: relevant?")
    print("  5. reformulate_node    → rewrite query nếu chunks không relevant")
    print("  6. answer_node         → generate với mandatory citations")

    print("\nConditional Edges:")
    print("  router → 'retrieve' nếu simple_qa/compare")
    print("  router → 'answer' nếu summarize/unknown")
    print("  grade → 'answer' nếu relevant")
    print("  grade → 'reformulate' nếu irrelevant + retry < 1")
    print("  grade → 'answer' nếu irrelevant + retry >= 1 (hết cơ hội)")

    print("\nTại sao LangGraph?")
    print("  - State machine rõ ràng: dễ debug, dễ trace từng bước")
    print("  - Vòng lặp có giới hạn: MAX_RETRIES=1 đảm bảo terminate")
    print("  - Mỗi node độc lập: test riêng từng node dễ dàng")


if __name__ == "__main__":
    demo_full_flow()
