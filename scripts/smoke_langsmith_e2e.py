"""E2E: câu hỏi thật qua POST /api/v1/query → LangSmith phải nhận trace từng câu.

Gọi Gemini + LangSmith thật (tốn quota), nên là script chứ không nằm trong pytest
(conftest đè GOOGLE_API_KEY bằng key giả). Cần LANGCHAIN_API_KEY trong .env.

    python scripts/smoke_langsmith_e2e.py            # 1 câu
    python scripts/smoke_langsmith_e2e.py --n 15     # 15 câu, mỗi chủ đề một câu

Câu hỏi lấy từ gold set (data/eval/legal_qa_200.json), kèm as_of_date của câu.
Đạt khi: mỗi câu trả 200, LangSmith có đủ N run gốc `documind-agent` mới, và mỗi
trace có span `gemini-generate` (trừ câu không cần LLM, vd compliance-check).
Exit 0 = đạt, 1 = trượt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import app  # noqa: E402
from src.config import get_settings  # noqa: E402

GOLD = REPO / "data/eval/legal_qa_200.json"
ROOT, CHILD = "documind-agent", "gemini-generate"


def pick_questions(n: int) -> list[dict]:
    """Mỗi category một câu (theo thứ tự trong file) cho đa dạng, đủ n thì dừng."""
    seen, out = set(), []
    for item in json.loads(GOLD.read_text(encoding="utf-8")):
        if item["category"] not in seen:
            seen.add(item["category"])
            out.append(item)
        if len(out) == n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1, help="số câu hỏi (mặc định 1)")
    args = ap.parse_args()

    s = get_settings()
    if not s.langchain_api_key:
        print("FAIL: LANGCHAIN_API_KEY trống trong .env")
        return 1

    questions = pick_questions(args.n)
    t0 = datetime.now(timezone.utc)
    failed = 0
    with TestClient(app) as client:  # chạy lifespan → set env LANGCHAIN_*
        for i, q in enumerate(questions, 1):
            body = {"query": q["question"], "session_id": f"ls-e2e-{uuid.uuid4().hex[:8]}"}
            if q.get("as_of_date"):
                body["as_of_date"] = q["as_of_date"]
            if i > 1:  # API giới hạn 10 req/phút/IP → giãn ≥6.1s giữa các lần gửi
                time.sleep(max(0.0, 6.1 - (time.time() - t)))
            t = time.time()
            r = client.post("/api/v1/query", json=body)
            ok = r.status_code == 200
            failed += not ok
            answer = r.json().get("answer", "") if ok else r.text
            print(f"\n[{i:02d}] {q['id']} ({q['category']}, {q['expected_behavior']}) "
                  f"HTTP {r.status_code} {time.time() - t:.1f}s")
            print(f"  Q: {q['question']}")
            print(f"  A: {answer[:220].replace(chr(10), ' ')}")
    if failed:
        print(f"\nFAIL: {failed}/{len(questions)} câu không trả 200")
        return 1

    from langsmith import Client

    ls = Client()
    # ponytail: poll thưa, 2 truy vấn/lượt — /runs/query của free tier trả 429 nếu gọi dày
    for attempt in range(5):
        time.sleep(15)
        try:
            roots = [r for r in ls.list_runs(project_name=s.langchain_project, is_root=True, start_time=t0)
                     if r.name == ROOT]
            gen_traces = {r.trace_id for r in ls.list_runs(project_name=s.langchain_project, start_time=t0)
                          if r.name == CHILD}
        except Exception as exc:  # 429, mạng chập chờn — thử lại vòng sau
            print(f"[{attempt + 1}/5] LangSmith lỗi: {exc.__class__.__name__}")
            continue
        if len(roots) < len(questions):
            print(f"[{attempt + 1}/5] mới thấy {len(roots)}/{len(questions)} run {ROOT}")
            continue
        no_gen = [r for r in roots if r.trace_id not in gen_traces]
        print(f"\nLangSmith: {len(roots)} run {ROOT} "
              f"({sum(r.status == 'success' for r in roots)} success), "
              f"{len(roots) - len(no_gen)} trace có span {CHILD}")
        for r in no_gen:  # từ chối/smalltalk có thể không gọi LLM — liệt kê để người xem
            print(f"  không có {CHILD}: {str((r.inputs or {}).get('query', ''))[:80]}")
        if len(no_gen) == len(roots):
            print(f"FAIL: không trace nào có {CHILD} — context tracing không đi qua LangGraph")
            return 1
        print("PASS")
        return 0

    print("FAIL: hết thời gian chờ, LangSmith không đủ trace")
    return 1


if __name__ == "__main__":
    sys.exit(main())
