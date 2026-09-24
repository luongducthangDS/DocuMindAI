"""WS /ws/{session_id}: bước tiến trình chỉ gửi khi client bật "progress"."""
import uuid
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _run_turn(payload: dict) -> list:
    from src.api.routes import query as query_route

    async def noop():
        return None

    async def fake_stream(query, chunks):
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
    assert json.loads(frames[-1]) == {"done": True, "sources": [{"index": 1}]}


def test_no_step_frames_for_legacy_clients():
    frames = _run_turn({"query": "Mức lương tối thiểu vùng I?"})
    assert not any(f.startswith('{"step"') for f in frames)
    assert frames[:2] == ["Mức lương ", "5.310.000 đồng [1]"]
