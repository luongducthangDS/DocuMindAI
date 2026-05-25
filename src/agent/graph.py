"""
LangGraph workflow for DocuMind AI agent.

Graph topology:
  User Query → Router → [simple_qa | compare | summarize | report] → Response

Each node is a pure function operating on AgentState (TypedDict).
The router classifies intent and routes to the appropriate sub-graph.
"""

from __future__ import annotations

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
from src.rag.retriever import nodes_to_chunks, retrieve_direct_chroma

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
    intent: Literal["simple_qa", "compare", "summarize", "report", "unknown"]
    retrieved_chunks: list
    answer: str
    sources: list
    used_llm: str
    session_id: str
    latency_ms: int
    error: str | None


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
- unknown: không liên quan đến pháp luật

Chỉ trả về một từ duy nhất (không giải thích)."""


# ── Nodes ──────────────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> dict:
    """Classify query intent using LLM."""
    settings = get_settings()
    query = state["query"]

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
            # Simple keyword-based fallback
            intent_raw = _keyword_classify(query)

    except Exception as exc:
        logger.warning("Router LLM failed, using keyword classify: {}", exc)
        intent_raw = _keyword_classify(query)

    valid = {"simple_qa", "compare", "summarize", "report", "unknown"}
    intent = intent_raw if intent_raw in valid else "simple_qa"

    logger.info("Query classified as: {} | '{}'", intent, query[:60])
    return {"intent": intent}


def _keyword_classify(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["so sánh", "khác nhau", "giống nhau", "phân biệt"]):
        return "compare"
    if any(w in q for w in ["tóm tắt", "tóm lược", "nội dung chính"]):
        return "summarize"
    if any(w in q for w in ["báo cáo", "xuất pdf", "tổng hợp"]):
        return "report"
    return "simple_qa"


def retrieve_node(state: AgentState) -> dict:
    """Run hybrid retrieval for any intent that needs context."""
    import src.rag.retriever as r_module

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
            import asyncio
            nodes = asyncio.run(retriever.aretrieve(query))
        else:
            nodes = retriever.retrieve(query)

        chunks = nodes_to_chunks(nodes)
        if not chunks:
            logger.warning("Retriever returned no chunks, trying direct Chroma fallback")
            chunks = retrieve_direct_chroma(query)
        return {"retrieved_chunks": chunks}

    except Exception as exc:
        logger.error("retrieve_node failed: {}", exc)
        chunks = retrieve_direct_chroma(query)
        return {"retrieved_chunks": chunks, "error": str(exc) if not chunks else None}


def answer_node(state: AgentState) -> dict:
    """Generate answer with citations from retrieved chunks."""
    t0 = time.time()
    chunks = state.get("retrieved_chunks", [])
    result = generate_answer(state["query"], chunks)
    latency = int((time.time() - t0) * 1000)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "used_llm": result["used_llm"],
        "latency_ms": latency,
        "messages": [AIMessage(content=result["answer"])],
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


async def compare_node(state: AgentState) -> dict:
    """Retrieve extra context for compare intent."""
    from src.agent.tools import compare_documents

    query = state["query"]
    try:
        result = await compare_documents.ainvoke({
            "doc_a": query,
            "doc_b": query,
            "aspect": query,
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
            query=state["query"],
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
    mapping = {
        "simple_qa": "do_retrieve",
        "compare": "do_compare",
        "summarize": "do_summarize",
        "report": "do_report",
        "unknown": "do_retrieve",
    }
    return mapping.get(intent, "do_retrieve")


# ── Graph Builder ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    # Node names must not conflict with AgentState keys.
    # "answer", "sources", "query" etc. are state keys — use prefixed names.
    g.add_node("router", router_node)
    g.add_node("do_retrieve", retrieve_node)
    g.add_node("do_answer", answer_node)
    g.add_node("do_compare", compare_node)
    g.add_node("do_summarize", summarize_node)
    g.add_node("do_report", report_node)
    g.add_node("do_persist", persist_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_by_intent)
    g.add_edge("do_retrieve", "do_answer")
    g.add_edge("do_answer", "do_persist")
    g.add_edge("do_compare", "do_persist")
    g.add_edge("do_summarize", "do_persist")
    g.add_edge("do_report", "do_persist")
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
        "intent": "simple_qa",
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "used_llm": "",
        "session_id": session_id,
        "latency_ms": 0,
        "error": None,
    }

    result = await graph.ainvoke(initial_state)
    return result
