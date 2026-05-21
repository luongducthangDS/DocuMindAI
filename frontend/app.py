"""
Streamlit frontend for DocuMind AI.
Connects to FastAPI backend via REST + WebSocket.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import httpx
import streamlit as st
import websockets

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8080")
WS_BASE = API_BASE.replace("http", "ws").replace("https", "wss")


# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind AI — Pháp luật Việt Nam",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session State Init ─────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    import uuid

    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "stream_mode" not in st.session_state:
    st.session_state.stream_mode = True


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ DocuMind AI")
    st.caption("Hỏi đáp pháp luật Việt Nam với trích dẫn nguồn")

    st.divider()

    st.subheader("Cài đặt")
    st.session_state.stream_mode = st.toggle("Streaming response", value=True)

    st.divider()

    st.subheader("Upload văn bản")
    uploaded = st.file_uploader("Thêm văn bản PDF", type=["pdf"], accept_multiple_files=False)
    if uploaded and st.button("Ingest PDF"):
        with st.spinner("Đang xử lý..."):
            result = _upload_doc(uploaded)
        if result:
            st.success(f"✅ {result.get('message', 'Thành công')}")
        else:
            st.error("Upload thất bại")

    st.divider()

    st.subheader("Tạo báo cáo")
    report_query = st.text_input("Chủ đề báo cáo", placeholder="Luật doanh nghiệp 2020...")
    report_title = st.text_input("Tiêu đề", placeholder="Báo cáo tổng hợp...")
    if st.button("📄 Tạo PDF", disabled=not report_query):
        with st.spinner("Đang tạo báo cáo..."):
            url = _create_report(report_title or f"Báo cáo: {report_query}", report_query)
        if url:
            st.success("Báo cáo đã sẵn sàng!")
            st.markdown(f"[📥 Tải về]({API_BASE}{url})")

    st.divider()

    st.caption(f"Session: `{st.session_state.session_id}`")

    if st.button("🗑️ Xóa lịch sử"):
        st.session_state.messages = []
        st.rerun()

    # Health status
    with st.expander("Trạng thái hệ thống"):
        status_data = _get_health()
        if status_data:
            for svc in status_data.get("services", []):
                icon = "🟢" if svc.get("healthy") else "🔴"
                st.write(f"{icon} {svc.get('name')}: {svc.get('detail', '')[:40]}")


# ── Main Chat UI ───────────────────────────────────────────────────────────────
st.title("💬 Hỏi đáp Pháp luật Việt Nam")
st.caption("Hệ thống RAG + AI Agent — câu trả lời kèm điều khoản cụ thể")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Nguồn ({len(msg['sources'])} điều khoản)"):
                for src in msg["sources"]:
                    title = src.get("title", "")
                    dieu = src.get("dieu_header", src.get("dieu", ""))
                    url = src.get("source_url", src.get("url", ""))
                    score = src.get("score", src.get("relevance_score", 0))
                    line = f"**{title}**"
                    if dieu:
                        line += f" — {dieu[:80]}"
                    st.markdown(line)
                    if url:
                        st.caption(f"🔗 [{url[:60]}]({url})")
                    if score:
                        st.caption(f"Độ liên quan: {score:.2f}")

# Query input
if prompt := st.chat_input("Nhập câu hỏi pháp luật... (ví dụ: Điều kiện thành lập công ty TNHH?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.stream_mode:
            answer, sources = _query_stream(prompt)
        else:
            answer, sources = _query_rest(prompt)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })


# ── API Helper Functions ───────────────────────────────────────────────────────

def _query_rest(query: str) -> tuple[str, list]:
    """Non-streaming REST query."""
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{API_BASE}/api/v1/query",
                json={"query": query, "session_id": st.session_state.session_id},
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("answer", "Không có phản hồi")
            sources = data.get("sources", [])
            latency = data.get("latency_ms", 0)

        st.markdown(answer)
        st.caption(f"⏱ {latency}ms | LLM: {data.get('used_llm', '?')} | {data.get('chunk_count', 0)} chunks")
        return answer, sources

    except httpx.HTTPStatusError as exc:
        err = f"Lỗi API: {exc.response.status_code}"
        st.error(err)
        return err, []
    except Exception as exc:
        err = f"Lỗi kết nối: {exc}"
        st.error(err)
        return err, []


def _query_stream(query: str) -> tuple[str, list]:
    """Streaming via WebSocket."""
    placeholder = st.empty()
    full_text = ""
    sources = []

    async def _stream():
        nonlocal full_text, sources
        uri = f"{WS_BASE}/api/v1/ws/{st.session_state.session_id}"
        try:
            async with websockets.connect(uri, ping_interval=20) as ws:
                await ws.send(json.dumps({"query": query}))
                async for msg in ws:
                    if isinstance(msg, str):
                        try:
                            data = json.loads(msg)
                            if data.get("done"):
                                sources = data.get("sources", [])
                                break
                            elif data.get("error"):
                                full_text += f"\n⚠️ {data['error']}"
                        except json.JSONDecodeError:
                            # Plain text token
                            full_text += msg
                            placeholder.markdown(full_text + "▌")
        except Exception as exc:
            full_text = f"Lỗi streaming: {exc}. Đang thử lại với REST..."
            _, rest_sources = _query_rest(query)
            sources.extend(rest_sources)

    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(_stream())
        loop.close()
    except Exception as exc:
        full_text = f"⚠️ Không thể kết nối streaming: {exc}"

    placeholder.markdown(full_text)

    if sources:
        with st.expander(f"📚 Nguồn ({len(sources)} điều khoản)"):
            for src in sources:
                st.markdown(f"**{src.get('title', '')}** — {src.get('dieu', '')[:80]}")

    return full_text, sources


def _upload_doc(file) -> dict | None:
    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{API_BASE}/api/v1/upload",
                files={"file": (file.name, file.getvalue(), "application/pdf")},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        st.error(f"Upload lỗi: {exc}")
        return None


def _create_report(title: str, query: str) -> str | None:
    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{API_BASE}/api/v1/report/create",
                json={"title": title, "query": query, "filename": "user_report"},
            )
            resp.raise_for_status()
            return resp.json().get("download_url")
    except Exception as exc:
        st.error(f"Tạo báo cáo lỗi: {exc}")
        return None


def _get_health() -> dict | None:
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{API_BASE}/api/v1/health")
            return resp.json()
    except Exception:
        return None
