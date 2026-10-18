"""
scripts/generate_doc2query.py — sinh câu hỏi đời thường cho từng đoạn của index (doc2query).

Nguyên liệu cho bộ câu hỏi tổng hợp, CHƯA AI DUYỆT. Không phải gold set, và không dùng trong
truy hồi: làm nhánh truy hồi thứ ba đã đo và loại (reports/retrieval_experiments_2026-10-18.json).

  python scripts/generate_doc2query.py                       # mọi đoạn chưa có / đã đổi nội dung
  python scripts/generate_doc2query.py --rpm 60 --workers 8

Mỗi dòng của data/eval/synthetic/doc2query_questions.jsonl ứng với một phiên bản đoạn
(`version_id`) và mang `text_sha1` của nội dung lúc sinh — chunk đổi thì hash lệch, dòng đó cũ.
Chạy lại chỉ sinh cho đoạn mới hoặc đã đổi; dòng cũ không tự xoá.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_settings  # noqa: E402
from src.rag.generator import _fpt_call  # noqa: E402

OUT = ROOT / "data" / "eval" / "synthetic" / "doc2query_questions.jsonl"
PROMPT = ("Đọc đoạn văn bản pháp luật dưới đây và viết 4 câu hỏi khác nhau mà người dân hoặc doanh nghiệp "
          "có thể hỏi và đoạn này trả lời được. Dùng lời lẽ đời thường, có thể kể tình huống. Mỗi câu một dòng, "
          "không đánh số, không giải thích.\n\n{text}")

# FPT 100 RPM mỗi model, key dùng chung với production (xem eval/corpus_llm_audit.py).
_LOCK, _NEXT = threading.Lock(), [0.0]


def _throttle(rpm: float) -> None:
    with _LOCK:
        wait = _NEXT[0] - time.monotonic()
        _NEXT[0] = max(_NEXT[0], time.monotonic()) + 60.0 / rpm
    if wait > 0:
        time.sleep(wait)


def text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def generate(text: str, models: list[str], rpm: float) -> tuple[list[str], str]:
    for attempt in range(5):
        for model in models:
            _throttle(rpm)
            try:
                lines = [ln.strip(" -•*\t") for ln in _fpt_call(model, PROMPT.format(text=text[:3500]), timeout=90).text.splitlines()]
                questions = [ln for ln in lines if len(ln) > 15 and not ln.startswith("<")][:4]
                if questions:
                    return questions, model
            except Exception:  # noqa: BLE001 — thử model kế; hết vòng thì chờ cửa sổ RPM trôi
                pass
        time.sleep(15 * (attempt + 1))
    return [], ""


def row(cid: str, text: str, meta: dict, questions: list[str], model: str) -> dict:
    return {
        "version_id": cid, "clause_uid": meta.get("clause_uid", ""), "doc_id": meta.get("doc_id", ""),
        "so_hieu": meta.get("so_hieu", ""), "dieu": meta.get("dieu"),
        "effective_from": meta.get("effective_from", ""), "effective_to": meta.get("effective_to", ""),
        "text_sha1": text_sha1(text), "model": model, "questions": questions,
    }


def main() -> None:
    import chromadb

    ap = argparse.ArgumentParser()
    ap.add_argument("--rpm", type=float, default=60)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    s = get_settings()
    models = [m.strip() for m in s.fpt_models.split(",") if m.strip()]
    if not s.fpt_api_key or not models:
        sys.exit("Thiếu FPT_API_KEY / FPT_MODELS")
    got = chromadb.PersistentClient(str(s.data_dir / "chroma_db")).get_collection(s.chroma_collection).get(
        include=["documents", "metadatas"])
    done = set()
    if OUT.exists():
        done = {(r["version_id"], r["text_sha1"]) for r in map(json.loads, OUT.read_text(encoding="utf-8").splitlines())}
    todo = [(i, d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"]) if (i, text_sha1(d)) not in done]
    print(f"{len(got['ids'])} đoạn, sinh cho {len(todo)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        results = pool.map(lambda t: generate(t[1][1], models[t[0] % len(models):] + models[:t[0] % len(models)], args.rpm),
                           enumerate(todo))
        for (cid, text, meta), (questions, model) in zip(todo, results):
            if questions:
                f.write(json.dumps(row(cid, text, meta, questions, model), ensure_ascii=False) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
