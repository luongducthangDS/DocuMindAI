"""
LangGraph workflow for DocuMind AI agent.

Graph topology:
  User Query → Router → [simple_qa | compare | summarize | report] → Response

Each node is a pure function operating on AgentState (TypedDict).
The router classifies intent and routes to the appropriate sub-graph.
"""

from __future__ import annotations

import re
import time
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger

from src.agent.memory import LongTermMemory, ShortTermMemory
from src.agent.tools import ALL_TOOLS
from src.config import get_settings
from src.rag.generator import generate_answer, stream_answer
from src.rag.grader import grade_chunks
from src.rag.retriever import nodes_to_chunks, retrieve_direct_chroma

# Hard cap on retrieval retries — bounds the only cycle in the graph so
# genuinely out-of-corpus questions still terminate instead of looping.
MAX_RETRIES = 2

# LangSmith tracing — optional; no-ops gracefully if langsmith not installed
# or LANGCHAIN_TRACING_V2 is not set.
try:
    from langsmith import traceable as _traceable  # type: ignore
    _LANGSMITH_AVAILABLE = True
except ImportError:
    # Fallback: identity decorator when langsmith is not installed
    def _traceable(**kwargs):
        def decorator(fn):
            return fn
        return decorator
    _LANGSMITH_AVAILABLE = False


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    original_query: str
    intent: Literal["simple_qa", "compare", "summarize", "report", "compliance_check", "unknown"]
    retrieved_chunks: list
    answer: str
    sources: list
    used_llm: str
    session_id: str
    latency_ms: int
    error: str | None
    steps: list  # [{label, detail, ms}] — visible reasoning steps for UI
    retry_count: int
    tried_queries: list
    grade: Literal["relevant", "irrelevant", "unknown"]
    grade_reason: str
    compliance_result: dict | None


# ── LLM Setup ─────────────────────────────────────────────────────────────────

def _get_llm():
    """Return a LangChain-compatible LLM bound with tools."""
    settings = get_settings()

    if settings.groq_api_key:
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=settings.groq_api_key,
            temperature=0.1,
        )
        return llm.bind_tools(ALL_TOOLS)

    if settings.google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=settings.google_api_key,
            temperature=0.1,
        )
        return llm.bind_tools(ALL_TOOLS)

    raise RuntimeError("No LLM API key configured. Set GROQ_API_KEY or GOOGLE_API_KEY.")


_ROUTER_PROMPT = """Phân loại ý định câu hỏi sau vào MỘT trong các loại:
- simple_qa: câu hỏi đơn giản, tra cứu một quy định
- compare: so sánh hai văn bản hoặc hai quy định
- summarize: yêu cầu tóm tắt một văn bản
- report: yêu cầu tạo báo cáo PDF hoặc tổng hợp nhiều văn bản
- compliance_check: kiểm tra một tình huống cụ thể có đáp ứng điều kiện/quy định hay không (ví dụ: "sinh viên có điểm rèn luyện 60 có đạt loại khá không?", "GPA 2.3 có đủ điều kiện làm khóa luận không?")
- unknown: không liên quan đến pháp luật

Chỉ trả về một từ duy nhất (không giải thích)."""


# ── Nodes ──────────────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> dict:
    """Classify query intent."""
    t0 = time.time()
    settings = get_settings()
    query = state["query"]

    keyword_intent = _keyword_classify(query)
    word_count = len(query.split())
    if keyword_intent != "simple_qa" or word_count <= 6:
        logger.info("Router (keyword fast-path): {} | '{}'", keyword_intent, query[:60])
        intent = keyword_intent
    else:
        try:
            if settings.groq_api_key:
                from groq import Groq
                client = Groq(api_key=settings.groq_api_key)
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": _ROUTER_PROMPT},
                        {"role": "user", "content": query},
                    ],
                    temperature=0,
                    max_tokens=10,
                )
                intent_raw = resp.choices[0].message.content.strip().lower()
            else:
                intent_raw = keyword_intent
        except Exception as exc:
            logger.warning("Router LLM failed, using keyword classify: {}", exc)
            intent_raw = keyword_intent

        valid = {"simple_qa", "compare", "summarize", "report", "compliance_check", "unknown"}
        intent = intent_raw if intent_raw in valid else "simple_qa"

    _INTENT_LABEL = {
        "simple_qa": "Tra cứu quy định",
        "compare": "So sánh văn bản",
        "summarize": "Tóm tắt văn bản",
        "report": "Tạo báo cáo",
        "compliance_check": "Kiểm định tuân thủ",
        "unknown": "Câu hỏi ngoài phạm vi",
    }
    ms = int((time.time() - t0) * 1000)
    logger.info("Query classified as: {} | '{}'", intent, query[:60])
    steps = state.get("steps") or []
    return {
        "intent": intent,
        "steps": steps + [{"label": "Phân loại câu hỏi", "detail": _INTENT_LABEL.get(intent, intent), "ms": ms}],
    }


_COMPLIANCE_PHRASES = ("có được", "có đủ điều kiện", "có bị", "có đạt", "quá", "vượt quá")
_COMPLIANCE_NUMBER_RE = re.compile(r"\d")


def _keyword_classify(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["so sánh", "khác nhau", "giống nhau", "phân biệt"]):
        return "compare"
    if any(w in q for w in ["tóm tắt", "tóm lược", "nội dung chính"]):
        return "summarize"
    if any(w in q for w in ["báo cáo", "xuất pdf", "tổng hợp"]):
        return "report"
    # Conservative fast-path: only fires when the query both contains an
    # eligibility phrase AND a number — avoids stealing generic Q&A.
    if any(p in q for p in _COMPLIANCE_PHRASES) and _COMPLIANCE_NUMBER_RE.search(q):
        return "compliance_check"
    return "simple_qa"


def retrieve_node(state: AgentState) -> dict:
    """Run hybrid retrieval for any intent that needs context."""
    import src.rag.retriever as r_module

    t0 = time.time()  # fix: must be defined before try/except
    query = state["query"]

    try:
        retriever = getattr(r_module, "_active_retriever", None)
        if retriever is None:
            logger.warning("No active retriever found, returning empty chunks")
            return {"retrieved_chunks": [], "sources": []}

        # retrieve_node runs in a thread (via LangGraph's run_in_executor),
        # so we use the synchronous retrieve() method to avoid nested event loops.
        # If only aretrieve() exists, use asyncio.run() which creates a fresh loop
        # in the worker thread (safe because threads don't have a running loop).
        if hasattr(retriever, "retrieve"):
            nodes = retriever.retrieve(query)
        elif hasattr(retriever, "aretrieve"):
            # asyncio.run() would fail here if there's already a running loop (FastAPI/LangGraph).
            # Use nest_asyncio if available, otherwise fall back to a new thread.
            import asyncio
            try:
                import nest_asyncio  # type: ignore
                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                nodes = loop.run_until_complete(retriever.aretrieve(query))
            except ImportError:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    nodes = ex.submit(asyncio.run, retriever.aretrieve(query)).result()
        else:
            nodes = retriever.retrieve(query)

        chunks = nodes_to_chunks(nodes)
        if not chunks:
            logger.warning("Retriever returned no chunks, trying direct Chroma fallback")
            chunks = retrieve_direct_chroma(query)

        ms = int((time.time() - t0) * 1000)
        seen: dict[str, int] = {}
        for c in chunks:
            key = c.metadata.get("so_hieu") or c.metadata.get("title", "?")
            seen[key] = seen.get(key, 0) + 1
        doc_summary = ", ".join(f"{k} ({v} đoạn)" for k, v in list(seen.items())[:3])
        detail = f"Tìm thấy {len(chunks)} đoạn" + (f" — {doc_summary}" if doc_summary else "")

        steps = state.get("steps") or []
        return {
            "retrieved_chunks": chunks,
            "steps": steps + [{"label": "Tìm kiếm tài liệu", "detail": detail, "ms": ms}],
        }

    except Exception as exc:
        logger.error("retrieve_node failed: {}", exc)
        chunks = retrieve_direct_chroma(query)
        ms = int((time.time() - t0) * 1000)
        steps = state.get("steps") or []
        return {
            "retrieved_chunks": chunks,
            "error": str(exc) if not chunks else None,
            "steps": steps + [{"label": "Tìm kiếm tài liệu", "detail": f"Fallback: {len(chunks)} đoạn", "ms": ms}],
        }


def grade_node(state: AgentState) -> dict:
    """Assess whether retrieved chunks are actually relevant to the query."""
    t0 = time.time()
    chunks = state.get("retrieved_chunks", [])
    result = grade_chunks(state["query"], chunks)
    ms = int((time.time() - t0) * 1000)
    steps = state.get("steps") or []
    return {
        "grade": "relevant" if result["relevant"] else "irrelevant",
        "grade_reason": result["reason"],
        "steps": steps + [{"label": "Đánh giá độ liên quan", "detail": result["reason"], "ms": ms}],
    }


_REFORMULATE_PROMPT = """Câu hỏi gốc: {query}
Các cách hỏi đã thử (không lặp lại): {tried}

Viết lại câu hỏi trên bằng từ khóa/thuật ngữ khác (ưu tiên thuật ngữ pháp lý, Điều/Khoản) \
nhưng giữ nguyên ý nghĩa, để tìm kiếm dễ khớp với văn bản quy định hơn. Chỉ trả về câu hỏi \
viết lại, không giải thích."""

_STOPWORDS = {"là", "của", "và", "có", "được", "cho", "trong", "một", "các", "những", "này"}


def _cheap_reformulate(query: str, tried: list[str]) -> str:
    """Deterministic fallback rewrite when the LLM reformulation call fails —
    strips common stopwords so the retriever sees a different token set."""
    words = [w for w in query.split() if w.lower() not in _STOPWORDS]
    rewritten = " ".join(words) or query
    if rewritten in tried:
        rewritten = query  # nothing left to vary — retry with original as last resort
    return rewritten


def reformulate_node(state: AgentState) -> dict:
    """Rewrite the query with different keywords before retrying retrieval."""
    t0 = time.time()
    query = state["query"]
    tried = state.get("tried_queries", [])
    settings = get_settings()

    new_query = None
    try:
        if settings.groq_api_key:
            from groq import Groq
            client = Groq(api_key=settings.groq_api_key)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": _REFORMULATE_PROMPT.format(
                    query=query, tried="; ".join(tried),
                )}],
                temperature=0.0,
                max_tokens=100,
            )
            candidate = (resp.choices[0].message.content or "").strip().strip('"')
            if candidate and candidate not in tried:
                new_query = candidate
    except Exception as exc:
        logger.warning("reformulate_node LLM failed, using cheap rewrite: {}", exc)

    if not new_query:
        new_query = _cheap_reformulate(query, tried)

    ms = int((time.time() - t0) * 1000)
    steps = state.get("steps") or []
    return {
        "query": new_query,
        "retry_count": state.get("retry_count", 0) + 1,
        "tried_queries": tried + [new_query],
        "steps": steps + [{"label": "Tìm lại với truy vấn khác", "detail": new_query, "ms": ms}],
    }


def answer_node(state: AgentState) -> dict:
    """Generate answer with citations from retrieved chunks."""
    t0 = time.time()
    chunks = state.get("retrieved_chunks", [])

    # Build conversation history from LangGraph messages (exclude current query = last HumanMessage)
    raw_messages = state.get("messages", [])
    history: list[dict] = []
    for msg in raw_messages[:-1]:  # skip last = current query
        role = "user" if getattr(msg, "type", "") == "human" else "assistant"
        content = getattr(msg, "content", "")
        if content:
            history.append({"role": role, "content": str(content)[:800]})
    history = history[-6:]  # keep last 3 turns (6 messages)

    result = generate_answer(state["query"], chunks, history=history or None)
    latency = int((time.time() - t0) * 1000)

    llm_label = {"groq": "Llama 3.3 70B", "gemini": "Gemini", "none": "Không cần LLM"}.get(
        result["used_llm"], result["used_llm"]
    )
    steps = state.get("steps") or []
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "used_llm": result["used_llm"],
        "latency_ms": latency,
        "messages": [AIMessage(content=result["answer"])],
        "steps": steps + [{"label": "Tổng hợp câu trả lời", "detail": llm_label, "ms": latency}],
    }


async def answer_node_async(state: AgentState) -> dict:
    """Async version — collects full streamed answer before returning."""
    t0 = time.time()
    chunks = state.get("retrieved_chunks", [])
    parts = []
    async for token in stream_answer(state["query"], chunks):
        parts.append(token)
    answer = "".join(parts)
    latency = int((time.time() - t0) * 1000)

    return {
        "answer": answer,
        "latency_ms": latency,
        "messages": [AIMessage(content=answer)],
    }


def _parse_compare_args(query: str) -> tuple[str, str, str]:
    """
    Extract (doc_a, doc_b, aspect) from a comparison query.
    Heuristic: look for 'và'/'with'/'vs' between two document identifiers.
    Falls back to (query, query, query) so compare_documents still gets called
    but with the full query — better than passing identical nonsense before.
    """
    import re
    # Pattern: "so sánh QĐ-740 và QĐ-747 về học bổng"
    m = re.search(
        r"(QĐ-\w+|QĐ\s*\d+|\d{2,4}/\w+[-/]\w+)",
        query, re.IGNORECASE
    )
    parts = re.split(r"\s+(?:và|with|vs\.?)\s+", query, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        doc_a = parts[0].strip()
        # aspect is everything after the second doc token (e.g., "về học bổng")
        right = parts[1].strip()
        aspect_match = re.split(r"\s+(?:về|on|regarding)\s+", right, maxsplit=1, flags=re.IGNORECASE)
        doc_b = aspect_match[0].strip()
        aspect = aspect_match[1].strip() if len(aspect_match) > 1 else ""
        return doc_a, doc_b, aspect or query
    # No clear split found — use full query for both sides; compare_documents
    # will retrieve from the same corpus but the prompt will still be coherent
    return query, query, query


async def compare_node(state: AgentState) -> dict:
    """Retrieve extra context for compare intent."""
    from src.agent.tools import compare_documents

    query = state["query"]
    doc_a, doc_b, aspect = _parse_compare_args(query)
    logger.info("compare_node: doc_a='{}', doc_b='{}', aspect='{}'", doc_a[:40], doc_b[:40], aspect[:40])
    try:
        result = await compare_documents.ainvoke({
            "doc_a": doc_a,
            "doc_b": doc_b,
            "aspect": aspect,
        })
        return {
            "answer": result,
            "messages": [AIMessage(content=result)],
        }
    except Exception as exc:
        logger.error("compare_node failed: {}", exc)
        return await answer_node_async(state)


async def summarize_node(state: AgentState) -> dict:
    """Summarize document."""
    from src.agent.tools import summarize_document

    query = state["query"]
    try:
        result = await summarize_document.ainvoke({"doc_title": query, "focus": ""})
        return {
            "answer": result,
            "messages": [AIMessage(content=result)],
        }
    except Exception as exc:
        logger.error("summarize_node failed: {}", exc)
        return await answer_node_async(state)


_COMPLIANCE_VERDICT_PREFIX = {
    "pass": "✅ **Đạt điều kiện.**",
    "fail": "❌ **Không đạt điều kiện.**",
}


def _render_compliance_answer(result: dict) -> str:
    verdict = result.get("verdict")
    if verdict in ("pass", "fail"):
        prefix = _COMPLIANCE_VERDICT_PREFIX[verdict]
        citation = result.get("citation") or {}
        source_line = f"\n\n**Nguồn:** {citation.get('so_hieu', '')} — {citation.get('dieu_khoan', '')}"
        return f"{prefix} {result.get('explanation', '')}{source_line}"
    if verdict == "insufficient_info":
        return result.get("explanation", "Không đủ thông tin để kết luận.")
    return ""


async def compliance_check_node(state: AgentState) -> dict:
    """Match the situation against the hand-curated criteria table
    (data/compliance/criteria.json) and return a pass/fail verdict with
    citation. Falls back to the normal retrieve→answer flow when no
    criterion matches or the situation lacks a usable number — same
    graceful-degrade pattern as compare_node/summarize_node/report_node."""
    t0 = time.time()
    from src.rag.compliance import check_compliance

    try:
        result = check_compliance(state["query"])
    except Exception as exc:
        logger.error("compliance_check_node failed: {}", exc)
        return await answer_node_async(state)

    if result["verdict"] == "no_match":
        # No curated criterion matches this situation at all — normal RAG
        # retrieval has a better shot than a bare "no match" message.
        return await answer_node_async(state)

    answer = _render_compliance_answer(result)
    ms = int((time.time() - t0) * 1000)
    steps = state.get("steps") or []
    citation = result.get("citation") or {}
    sources = (
        [{
            "index": 1,
            "title": citation.get("title", ""),
            "dieu_header": citation.get("dieu_khoan", ""),
            "source_url": citation.get("source_url", ""),
            "score": 1.0,
        }]
        if result["matched"] and citation
        else []
    )
    return {
        "answer": answer,
        "sources": sources,
        "used_llm": "compliance_rule_engine",
        "compliance_result": result,
        "latency_ms": ms,
        "messages": [AIMessage(content=answer)],
        "steps": steps + [{"label": "Kiểm định tuân thủ", "detail": result["verdict"], "ms": ms}],
    }


async def report_node(state: AgentState) -> dict:
    """Generate PDF report."""
    from src.agent.tools import generate_pdf_report

    query = state["query"]
    try:
        result = await generate_pdf_report.ainvoke({
            "title": f"Báo cáo: {query[:60]}",
            "query": query,
            "output_filename": "auto_report",
        })
        return {
            "answer": result,
            "messages": [AIMessage(content=result)],
        }
    except Exception as exc:
        logger.error("report_node failed: {}", exc)
        return await answer_node_async(state)


def persist_node(state: AgentState) -> dict:
    """Log query to SQLite for long-term memory."""
    try:
        mem = LongTermMemory()
        mem.log_query(
            session_id=state.get("session_id", "default"),
            query=state.get("original_query") or state["query"],
            answer_snippet=state.get("answer", "")[:200],
            latency_ms=state.get("latency_ms", 0),
            used_llm=state.get("used_llm", "unknown"),
        )
    except Exception as exc:
        logger.warning("persist_node failed (non-fatal): {}", exc)
    # LangGraph requires at least one key returned — pass error through unchanged.
    return {"error": state.get("error")}


# ── Routing function ───────────────────────────────────────────────────────────

def route_by_intent(state: AgentState) -> str:
    intent = state.get("intent", "simple_qa")
    # "unknown" still goes through retrieval — better to attempt an answer than silently fail.
    # The generator will say "Tôi không tìm thấy quy định này" if context is irrelevant.
    mapping = {
        "simple_qa": "do_retrieve",
        "compare": "do_compare",
        "summarize": "do_summarize",
        "report": "do_report",
        "compliance_check": "do_compliance",
        "unknown": "do_retrieve",
    }
    return mapping.get(intent, "do_retrieve")


def route_after_grade(state: AgentState) -> str:
    """Cycle back to retrieval with a reformulated query if chunks graded
    irrelevant, up to MAX_RETRIES — otherwise proceed to answer (the
    generator's own score threshold will abstain gracefully if context is
    still weak after exhausting retries)."""
    if state.get("grade") == "relevant":
        return "do_answer"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "do_answer"
    return "do_reformulate"


# ── Graph Builder ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    # Node names must not conflict with AgentState keys.
    # "answer", "sources", "query" etc. are state keys — use prefixed names.
    g.add_node("router", router_node)
    g.add_node("do_retrieve", retrieve_node)
    g.add_node("do_grade", grade_node)
    g.add_node("do_reformulate", reformulate_node)
    g.add_node("do_answer", answer_node)
    g.add_node("do_compare", compare_node)
    g.add_node("do_summarize", summarize_node)
    g.add_node("do_report", report_node)
    g.add_node("do_compliance", compliance_check_node)
    g.add_node("do_persist", persist_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_by_intent)
    g.add_edge("do_retrieve", "do_grade")
    g.add_conditional_edges(
        "do_grade", route_after_grade, {"do_answer": "do_answer", "do_reformulate": "do_reformulate"}
    )
    g.add_edge("do_reformulate", "do_retrieve")
    g.add_edge("do_answer", "do_persist")
    g.add_edge("do_compare", "do_persist")
    g.add_edge("do_summarize", "do_persist")
    g.add_edge("do_report", "do_persist")
    g.add_edge("do_compliance", "do_persist")
    g.add_edge("do_persist", END)

    return g.compile()


# Singleton compiled graph — built lazily
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
        logger.info("LangGraph compiled successfully")
    return _graph


@_traceable(
    name="documind-agent",
    run_type="chain",
    tags=["rag", "langgraph", "legal-qa"],
)
async def run_agent(
    query: str,
    session_id: str = "default",
    history: list[dict] | None = None,
) -> AgentState:
    """Main entry point for agent invocation.

    Decorated with @traceable so every invocation is visible in LangSmith as
    a top-level run named 'documind-agent', containing all child LLM calls
    (router classification, answer generation) and tool invocations.
    """
    graph = get_graph()

    messages = [HumanMessage(content=query)]
    if history:
        messages = [
            HumanMessage(content=m["content"]) if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in history[-6:]  # last 3 turns
        ] + messages

    initial_state: AgentState = {
        "messages": messages,
        "query": query,
        "original_query": query,
        "intent": "simple_qa",
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "used_llm": "",
        "session_id": session_id,
        "latency_ms": 0,
        "error": None,
        "steps": [],
        "retry_count": 0,
        "tried_queries": [query],
        "grade": "unknown",
        "grade_reason": "",
        "compliance_result": None,
    }

    result = await graph.ainvoke(initial_state)
    return result
