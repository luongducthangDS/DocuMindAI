"""
LangGraph Agent Tools — 6 core tools for legal document reasoning.
Each tool is a plain async function decorated for LangGraph/LangChain.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from langchain_core.tools import tool
from loguru import logger

from src.rag.retriever import RetrievedChunk


# ── Shared state injected at runtime ──────────────────────────────────────────
_retriever = None
_index = None


def configure_tools(retriever, index) -> None:
    """Called once during app startup to inject retriever and index."""
    global _retriever, _index
    _retriever = retriever
    _index = index


# ── Tool 1: Search ─────────────────────────────────────────────────────────────

@tool
async def search_legal_docs(
    query: Annotated[str, "Câu hỏi hoặc từ khóa pháp luật cần tìm kiếm"],
    top_k: Annotated[int, "Số lượng kết quả tối đa"] = 5,
) -> list[dict]:
    """Tìm kiếm văn bản pháp luật liên quan trong cơ sở dữ liệu."""
    if _retriever is None:
        return [{"error": "Retriever chưa được khởi tạo"}]

    try:
        if hasattr(_retriever, "aretrieve"):
            nodes = await _retriever.aretrieve(query)
        else:
            nodes = _retriever.retrieve(query)

        from src.rag.retriever import nodes_to_chunks

        chunks = nodes_to_chunks(nodes[:top_k])
        return [
            {
                "text": c.text[:500],
                "score": round(c.score, 4),
                "title": c.metadata.get("title", ""),
                "dieu": c.metadata.get("dieu_header", ""),
                "url": c.metadata.get("source_url", ""),
            }
            for c in chunks
        ]
    except Exception as exc:
        logger.error("search_legal_docs failed: {}", exc)
        return [{"error": str(exc)}]


# ── Tool 2: Summarize ──────────────────────────────────────────────────────────

@tool
async def summarize_document(
    doc_title: Annotated[str, "Tên văn bản cần tóm tắt"],
    focus: Annotated[str, "Khía cạnh cần nhấn mạnh (để trống nếu tóm tắt toàn bộ)"] = "",
) -> str:
    """Tóm tắt nội dung chính của một văn bản pháp luật cụ thể."""
    results = await search_legal_docs.ainvoke({"query": doc_title, "top_k": 10})
    if not results or "error" in results[0]:
        return f"Không tìm thấy văn bản: {doc_title}"

    context = "\n\n".join(r.get("text", "") for r in results)
    focus_note = f" Tập trung vào: {focus}." if focus else ""

    from src.rag.generator import _get_groq_client, _SYSTEM_PROMPT

    try:
        client = _get_groq_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Tóm tắt văn bản sau.{focus_note}\n\n{context}",
                },
            ],
            temperature=0.1,
            max_tokens=512,
        )
        return resp.choices[0].message.content
    except Exception as exc:
        logger.error("summarize_document LLM call failed: {}", exc)
        return f"Lỗi khi tóm tắt: {exc}"


# ── Tool 3: Compare ────────────────────────────────────────────────────────────

@tool
async def compare_documents(
    doc_a: Annotated[str, "Tên hoặc số hiệu văn bản thứ nhất"],
    doc_b: Annotated[str, "Tên hoặc số hiệu văn bản thứ hai"],
    aspect: Annotated[str, "Khía cạnh cần so sánh (ví dụ: mức phạt, quy định thành lập...)"],
) -> str:
    """So sánh hai văn bản pháp luật theo một khía cạnh cụ thể."""
    results_a, results_b = await asyncio.gather(
        search_legal_docs.ainvoke({"query": f"{doc_a} {aspect}", "top_k": 5}),
        search_legal_docs.ainvoke({"query": f"{doc_b} {aspect}", "top_k": 5}),
    )

    ctx_a = "\n".join(r.get("text", "") for r in results_a if "error" not in r)
    ctx_b = "\n".join(r.get("text", "") for r in results_b if "error" not in r)

    if not ctx_a and not ctx_b:
        return "Không tìm thấy nội dung để so sánh."

    from src.rag.generator import _get_groq_client, _SYSTEM_PROMPT

    prompt = (
        f"So sánh {doc_a} và {doc_b} về khía cạnh: {aspect}\n\n"
        f"**{doc_a}:**\n{ctx_a[:1500]}\n\n"
        f"**{doc_b}:**\n{ctx_b[:1500]}\n\n"
        "Trình bày dưới dạng bảng so sánh nếu có thể."
    )

    try:
        client = _get_groq_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=768,
        )
        return resp.choices[0].message.content
    except Exception as exc:
        logger.error("compare_documents failed: {}", exc)
        return f"Lỗi khi so sánh: {exc}"


# ── Tool 4: Extract Articles ───────────────────────────────────────────────────

@tool
async def extract_articles(
    query: Annotated[str, "Chủ đề hoặc điều khoản cần trích xuất"],
    doc_filter: Annotated[str, "Lọc theo tên văn bản cụ thể (để trống nếu tìm tất cả)"] = "",
) -> list[dict]:
    """Trích xuất danh sách các Điều/Khoản liên quan đến chủ đề."""
    search_q = f"{query} {doc_filter}".strip()
    results = await search_legal_docs.ainvoke({"query": search_q, "top_k": 8})

    articles = []
    for r in results:
        if "error" in r:
            continue
        dieu = r.get("dieu", "")
        if dieu:
            articles.append({
                "dieu": dieu,
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "relevance_score": r.get("score", 0),
                "excerpt": r.get("text", "")[:200],
            })
    return articles


# ── Tool 5: Generate PDF Report ────────────────────────────────────────────────

@tool
async def generate_pdf_report(
    title: Annotated[str, "Tiêu đề báo cáo"],
    query: Annotated[str, "Nội dung/chủ đề chính của báo cáo"],
    output_filename: Annotated[str, "Tên file đầu ra (không cần .pdf)"] = "report",
) -> str:
    """Tạo báo cáo PDF tổng hợp từ nhiều văn bản pháp luật liên quan."""
    from src.report.generator import create_legal_report

    results = await search_legal_docs.ainvoke({"query": query, "top_k": 8})
    summary = await summarize_document.ainvoke({"doc_title": query, "focus": ""})

    # Sanitize filename — prevent path traversal
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in output_filename)
    safe_name = safe_name[:60]

    chunks = [
        RetrievedChunk(
            text=r.get("text", ""),
            score=r.get("score", 0),
            metadata={
                "title": r.get("title", ""),
                "dieu_header": r.get("dieu", ""),
                "source_url": r.get("url", ""),
            },
        )
        for r in results if "error" not in r
    ]

    path = await asyncio.to_thread(
        create_legal_report,
        title=title,
        summary=summary,
        chunks=chunks,
        filename=safe_name,
    )
    return f"Báo cáo đã tạo: {path}"


# ── Tool 6: Document Metadata ─────────────────────────────────────────────────

@tool
async def get_document_metadata(
    doc_name: Annotated[str, "Tên hoặc số hiệu văn bản pháp luật"],
) -> dict:
    """Lấy thông tin metadata của văn bản: số hiệu, ngày ban hành, cơ quan ban hành."""
    results = await search_legal_docs.ainvoke({"query": doc_name, "top_k": 1})

    if not results or "error" in results[0]:
        return {"error": f"Không tìm thấy thông tin về: {doc_name}"}

    r = results[0]
    return {
        "title": r.get("title", ""),
        "url": r.get("url", ""),
        "dieu_header": r.get("dieu", ""),
        "relevance_score": r.get("score", 0),
    }


ALL_TOOLS = [
    search_legal_docs,
    summarize_document,
    compare_documents,
    extract_articles,
    generate_pdf_report,
    get_document_metadata,
]
