"""WS /ws/{session_id}: bước tiến trình chỉ gửi khi client bật "progress"."""
import uuid
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _run_turn(payload: dict) -> list:
    from src.api.routes import query as query_route

    async def noop():
        return None

    async def fake_stream(query, chunks, as_of_date=None):
        for tok in ("Mức lương ", "5.310.000 đồng [1]"):
            yield tok

    app = FastAPI()
    app.include_router(query_route.router)
    frames = []
    with patch("src.api.main.ensure_rag_initialized", noop), \
         patch("src.rag.retriever.retrieve_with_context", return_value=["c1", "c2"]), \
         patch.object(query_route, "stream_answer", fake_stream), \
         patch.object(query_route, "_cited_sources", return_value=[{"index": 1}]):
        with TestClient(app).websocket_connect(f"/api/v1/ws/t-{uuid.uuid4().hex[:12]}") as ws:
            ws.send_json(payload)
            while True:
                msg = ws.receive()
                text = msg.get("text")
                frames.append(text)
                if text and '"done"' in text:
                    break
    return frames


def test_progress_steps_are_sent_in_order_when_requested():
    import json

    frames = _run_turn({"query": "Mức lương tối thiểu vùng I?", "progress": True, "as_of_date": "2025-10-01"})
    steps = [json.loads(f)["step"] for f in frames if f.startswith('{"step"')]
    assert [s["label"] for s in steps] == [
        "Phân tích câu hỏi", "Tìm kiếm tài liệu", "Lọc theo hiệu lực", "Tổng hợp câu trả lời",
    ]
    assert steps[1]["detail"] == "2 đoạn liên quan"
    assert steps[2]["detail"] == "Áp dụng tại 01/10/2025"
    assert steps[3]["detail"] == "1 trích dẫn"
    # Bước tổng hợp đến sau toàn bộ token, trước "done".
    synth = next(i for i, f in enumerate(frames) if "Tổng hợp câu trả lời" in f)
    assert frames.index("5.310.000 đồng [1]") < synth == len(frames) - 2
    assert json.loads(frames[-1]) == {"done": True, "sources": [{"index": 1}], "degraded": []}


def test_agent_research_steps_and_passages_reach_the_stream():
    """Agentic RAG trên WS: bước "Tìm thêm" nằm giữa lọc hiệu lực và tổng hợp; đoạn agent tìm thêm vào stream_answer."""
    import json

    from src.api.routes import query as query_route

    seen = {}

    async def fake_stream(query, chunks, as_of_date=None):
        seen["chunks"] = chunks
        yield "ok [1]"

    def fake_research(query, chunks, ctx):
        seen["ctx_as_of"] = ctx.as_of_date
        return chunks + ["c67"], [{"label": "Tìm thêm", "detail": "Điều 67 41/2024/QH15 — thiếu cơ sở", "ms": 1200}]

    app = FastAPI()
    app.include_router(query_route.router)

    async def noop():
        return None

    frames = []
    with patch("src.api.main.ensure_rag_initialized", noop), \
         patch("src.rag.retriever.retrieve_with_context", return_value=["c1"]), \
         patch("src.agent.research.research", fake_research), \
         patch.object(query_route, "stream_answer", fake_stream), \
         patch.object(query_route, "_cited_sources", return_value=[]):
        with TestClient(app).websocket_connect(f"/api/v1/ws/t-{uuid.uuid4().hex[:12]}") as ws:
            ws.send_json({"query": "Trợ cấp hằng tháng điều chỉnh thế nào?", "progress": True,
                          "as_of_date": "2025-10-01"})
            while True:
                text = ws.receive().get("text")
                frames.append(text)
                if text and '"done"' in text:
                    break
    steps = [json.loads(f)["step"] for f in frames if f.startswith('{"step"')]
    assert [s["label"] for s in steps] == [
        "Phân tích câu hỏi", "Tìm kiếm tài liệu", "Lọc theo hiệu lực", "Tìm thêm", "Tổng hợp câu trả lời"]
    assert steps[3]["ms"] >= 1200  # thời gian của bước agent, không phải 0
    assert seen["chunks"] == ["c1", "c67"] and seen["ctx_as_of"] == "2025-10-01"


def test_no_step_frames_for_legacy_clients():
    frames = _run_turn({"query": "Mức lương tối thiểu vùng I?"})
    assert not any(f.startswith('{"step"') for f in frames)
    assert frames[:2] == ["Mức lương ", "5.310.000 đồng [1]"]


def test_ws_smalltalk_greeting():
    """Lời chào 'hi' trên WS: trả lời thân thiện tức thì, không gọi retrieval hay stream_answer."""
    import json

    frames = _run_turn({"query": "hi", "progress": True})
    steps = [json.loads(f)["step"] for f in frames if f.startswith('{"step"')]
    assert len(steps) == 1
    assert steps[0]["label"] == "Phân loại câu hỏi"
    assert steps[0]["detail"] == "Giao tiếp thân thiện"
    # Toàn bộ nội dung trả lời ghép lại
    text_content = "".join(f for f in frames if not f.startswith("{"))
    assert "DocuMind AI" in text_content
    assert json.loads(frames[-1]) == {"done": True, "sources": [], "degraded": []}


def test_ws_smalltalk_chat_first():
    """Người dùng nói muốn nói chuyện trước: phản hồi ấm áp, không bị ép vào khuôn 'chưa tìm thấy quy định'."""
    import json

    frames = _run_turn({"query": "tôi chưa hỏi về công việc, tôi muốn nói chuyện trước", "progress": True})
    steps = [json.loads(f)["step"] for f in frames if f.startswith('{"step"')]
    assert len(steps) == 1
    assert steps[0]["label"] == "Phân loại câu hỏi"
    text_content = "".join(f for f in frames if not f.startswith("{"))
    assert "trò chuyện" in text_content.lower() or "thư thả" in text_content.lower()
    assert json.loads(frames[-1]) == {"done": True, "sources": [], "degraded": []}

