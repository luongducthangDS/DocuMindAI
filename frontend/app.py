"""
DocuMind AI — Streamlit frontend
Hỏi đáp văn bản pháp luật Việt Nam với trích dẫn nguồn.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid

import httpx
import streamlit as st
import websockets

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE = os.getenv(
    "API_BASE_URL",
    f"http://{os.getenv('API_HOST', '127.0.0.1')}:{os.getenv('API_PORT', '9000')}",
)
WS_BASE  = API_BASE.replace("http", "ws").replace("https", "wss")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind AI — Pháp luật Việt Nam",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State ──────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []


# ══════════════════════════════════════════════════════════════════════════════
# API HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _is_online() -> bool:
    """Kiểm tra backend có hoạt động không."""
    try:
        r = httpx.get(f"{API_BASE}/api/v1/health", timeout=30)
        if r.status_code >= 500:
            return False
        data = r.json()
        services = {s["name"]: s["healthy"] for s in data.get("services", [])}
        # Accept any configured LLM (Groq primary or Gemini fallback) + ChromaDB
        llm_ok = services.get("llm_groq", False) or services.get("llm_gemini", False)
        return llm_ok or services.get("chromadb", False)
    except Exception:
        return False


def _query_rest(query: str) -> tuple[str, list]:
    """Gửi câu hỏi qua REST, trả về (answer, sources)."""
    try:
        with httpx.Client(timeout=90) as client:
            resp = client.post(
                f"{API_BASE}/api/v1/query",
                json={"query": query, "session_id": st.session_state.session_id},
            )
            resp.raise_for_status()
            data = resp.json()
        return data.get("answer", "Không có phản hồi."), data.get("sources", [])
    except httpx.TimeoutException:
        return "⏳ Hệ thống mất nhiều thời gian xử lý hơn dự kiến. Vui lòng thử lại.", []
    except Exception:
        return "⚠️ Không thể kết nối đến hệ thống. Vui lòng thử lại sau.", []


def _query_stream(query: str, placeholder) -> tuple[str, list]:
    """Streaming qua WebSocket, cập nhật placeholder theo từng token."""
    full_text = ""
    sources: list = []

    async def _run():
        nonlocal full_text, sources
        uri = f"{WS_BASE}/api/v1/ws/{st.session_state.session_id}"
        try:
            async with websockets.connect(uri, ping_interval=20, open_timeout=10) as ws:
                await ws.send(json.dumps({"query": query}))
                async for raw in ws:
                    if not isinstance(raw, str):
                        continue
                    try:
                        pkt = json.loads(raw)
                        if pkt.get("done"):
                            sources = pkt.get("sources", [])
                            break
                        if pkt.get("error"):
                            full_text += f"\n\n⚠️ {pkt['error']}"
                    except json.JSONDecodeError:
                        full_text += raw
                        placeholder.markdown(full_text + "▌")
        except Exception:
            # Fallback sang REST khi WebSocket không dùng được
            answer, rest_sources = _query_rest(query)
            full_text = answer
            sources.extend(rest_sources)

    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(_run())
        loop.close()
    except Exception:
        full_text, sources = _query_rest(query)

    placeholder.markdown(full_text)
    return full_text, sources


def _upload_pdf(file) -> str | None:
    """Upload PDF, trả về thông báo thành công hoặc None."""
    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{API_BASE}/api/v1/upload",
                files={"file": (file.name, file.getvalue(), "application/pdf")},
            )
            resp.raise_for_status()
            return resp.json().get("message", "Tài liệu đã được thêm thành công.")
    except Exception:
        return None


def _create_report(title: str, query: str) -> tuple[str, bytes] | None:
    """Tạo báo cáo PDF, trả về URL tải về."""
    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{API_BASE}/api/v1/report/create",
                json={"title": title, "query": query, "filename": "bao_cao"},
            )
            resp.raise_for_status()
            data = resp.json()
            download_url = data.get("download_url")
            filename = data.get("filename", "bao_cao.pdf")
            if not download_url:
                return None

            file_resp = client.get(f"{API_BASE}{download_url}")
            file_resp.raise_for_status()
            return filename, file_resp.content
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # ── Logo & tiêu đề ──────────────────────────────────────────────────────
    st.markdown("## ⚖️ DocuMind AI")
    st.caption("Tra cứu pháp luật Việt Nam — câu trả lời có trích dẫn nguồn")

    # ── Trạng thái kết nối ──────────────────────────────────────────────────
    online = _is_online()
    if online:
        st.success("🟢 Hệ thống sẵn sàng", icon=None)
    else:
        st.warning("🔴 Đang kiểm tra kết nối…")

    st.divider()

    # ── Gợi ý câu hỏi ───────────────────────────────────────────────────────
    st.markdown("**💡 Câu hỏi thường gặp**")
    suggestions = [
        "Điều kiện thành lập công ty TNHH?",
        "Mức lương tối thiểu vùng hiện tại?",
        "Thủ tục đăng ký nhãn hiệu hàng hóa?",
        "Quyền và nghĩa vụ của người lao động?",
    ]
    for q in suggestions:
        if st.button(q, use_container_width=True, key=f"sug_{q[:20]}"):
            st.session_state["_pending_query"] = q
            st.rerun()

    st.divider()

    # ── Upload tài liệu ─────────────────────────────────────────────────────
    with st.expander("📎 Thêm tài liệu PDF"):
        uploaded = st.file_uploader(
            "Chọn file PDF",
            type=["pdf"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
        if uploaded:
            st.caption(f"📄 {uploaded.name} ({uploaded.size // 1024} KB)")
            if st.button("Tải lên & lập chỉ mục", use_container_width=True):
                with st.spinner("Đang xử lý tài liệu…"):
                    msg = _upload_pdf(uploaded)
                if msg:
                    st.success(msg)
                else:
                    st.error("Tải lên thất bại. Vui lòng thử lại.")

    # ── Tạo báo cáo ─────────────────────────────────────────────────────────
    with st.expander("📄 Xuất báo cáo PDF"):
        report_query = st.text_input(
            "Nội dung cần tổng hợp",
            placeholder="Ví dụ: Luật Doanh nghiệp 2020…",
        )
        report_title = st.text_input(
            "Tiêu đề báo cáo",
            placeholder="Báo cáo pháp lý…",
        )
        if st.button("📥 Tạo báo cáo", disabled=not report_query, use_container_width=True):
            with st.spinner("Đang tổng hợp nội dung…"):
                report = _create_report(
                    report_title or f"Báo cáo: {report_query}",
                    report_query,
                )
            if report:
                filename, content = report
                st.success("Báo cáo đã sẵn sàng!")
                st.download_button("Tải về báo cáo", data=content, file_name=filename, mime="application/pdf", use_container_width=True)
            else:
                st.error("Không thể tạo báo cáo. Vui lòng thử lại.")

    st.divider()

    # ── Xóa lịch sử ─────────────────────────────────────────────────────────
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — CHAT INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("## 💬 Hỏi đáp Pháp luật Việt Nam")


# ── Hàm hiển thị nguồn trích dẫn ────────────────────────────────────────────
def _render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"📚 {len(sources)} văn bản được trích dẫn"):
        for i, src in enumerate(sources, 1):
            title = src.get("title", "Văn bản pháp luật")
            dieu  = src.get("dieu_header", src.get("dieu", ""))
            url   = src.get("source_url", src.get("url", ""))

            col1, col2 = st.columns([8, 2])
            with col1:
                label = f"**{i}. {title}**"
                if dieu:
                    label += f"\n\n_{dieu[:120]}_"
                st.markdown(label)
            with col2:
                if url:
                    st.markdown(f"[🔗 Xem]({url})")
            if i < len(sources):
                st.divider()


# ── Màn hình chào khi chưa có tin nhắn ─────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        """
        <div style="text-align:center; padding: 3rem 1rem; color: #888;">
            <div style="font-size: 3rem">⚖️</div>
            <h3 style="color:#444">Chào mừng đến với DocuMind AI</h3>
            <p>Đặt câu hỏi về pháp luật Việt Nam — tôi sẽ trả lời kèm trích dẫn văn bản cụ thể.</p>
            <p style="font-size:0.85rem">Nguồn: Cổng Công báo Chính phủ · Cập nhật liên tục</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Render lịch sử hội thoại ────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        _render_sources(msg.get("sources", []))


# ── Xử lý câu hỏi từ gợi ý (sidebar buttons) ────────────────────────────────
if "_pending_query" in st.session_state:
    prompt = st.session_state.pop("_pending_query")
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        ph = st.empty()
        with st.spinner("Đang tìm kiếm văn bản pháp luật…"):
            answer, sources = _query_stream(prompt, ph)
        _render_sources(sources)
    st.session_state.messages.append({
        "role": "assistant", "content": answer, "sources": sources,
    })
    st.rerun()

# ── Chat input ───────────────────────────────────────────────────────────────
if prompt := st.chat_input("Nhập câu hỏi pháp luật…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Đang tìm kiếm văn bản pháp luật…"):
            answer, sources = _query_stream(prompt, placeholder)
        _render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
