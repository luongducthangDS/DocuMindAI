"""
eval/corpus_llm_audit.py — model FPT đọc từng đoạn corpus và xác định nội dung:
loại đoạn (quy phạm / bảng tra cứu / biểu mẫu / hành chính) + lỗi trích xuất.

Bổ sung cho `scripts/validate_corpus.py` (chỉ kiểm metadata) và
`tests/test_ingestion.py::TestCorpusTextIntegrity` (lỗi bắt được bằng regex): đây là
lớp đọc hiểu — câu đứt, bảng vỡ, lẫn nội dung — mà regex không thấy. Không phán xét
nội dung pháp lý đúng/sai.

Đoạn dựng đúng như lúc ingest (`discover_lao_dong`), không đọc vector store. Kết quả
cache theo hash nội dung đoạn trong `reports/corpus_llm_audit.jsonl`: chạy lại chỉ gọi
model cho đoạn mới/đổi.

Chạy:
    python eval/corpus_llm_audit.py                      # toàn corpus
    python eval/corpus_llm_audit.py --docs 135-2020-ND-CP --limit 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from scripts.ingest_documents import discover_lao_dong  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.rag.generator import _fpt_call  # noqa: E402

CACHE = ROOT / "reports" / "corpus_llm_audit.jsonl"
REPORT = ROOT / "reports" / "corpus_llm_audit.json"
LOAI = ("quy_pham", "bang_tra_cuu", "bieu_mau", "hanh_chinh")
LOI = ("cau_dut", "bang_vo", "lan_noi_dung", "ky_tu_loi", "trung_lap", "thieu_noi_dung")

PROMPT = """Bạn kiểm tra chất lượng TRÍCH XUẤT văn bản pháp luật Việt Nam (từ PDF/HTML sang text).
KHÔNG đánh giá nội dung pháp lý đúng hay sai. Đoạn dưới đây là một đơn vị truy hồi: một Điều,
một phần của Điều dài (đã tách theo khoản — mở/kết thúc ở ranh giới khoản là BÌNH THƯỜNG),
hoặc một phần phụ lục. Dòng đầu có thể là tiêu đề Chương/Mục do hệ thống gắn — bình thường.

Văn bản: {so_hieu} — {title}
Vị trí: {where}

<doan>
{text}
</doan>

Trả về DUY NHẤT một JSON:
{{"loai": "<quy_pham|bang_tra_cuu|bieu_mau|hanh_chinh>", "loi": [<0 hoặc nhiều mã lỗi>], "ghi_chu": "<tối đa 25 từ, nêu chỗ lỗi cụ thể>"}}

loai:
- quy_pham: nội dung điều, khoản, điểm có tính quy phạm
- bang_tra_cuu: bảng/danh mục số liệu để tra (mức lương vùng, tuổi nghỉ hưu, danh mục địa bàn, công việc)
- bieu_mau: mẫu văn bản để điền (nhiều chỗ trống "....", chú thích (1), (2)...)
- hanh_chinh: chỉ có nơi nhận, chữ ký, quốc hiệu — không có nội dung pháp lý

loi (chỉ báo khi thấy rõ trong đoạn):
- cau_dut: câu bị cắt cụt giữa chừng ở đầu/cuối đoạn hoặc giữa đoạn (không tính ranh giới khoản)
- bang_vo: bảng bị dàn phẳng, không xác định được giá trị nào thuộc hàng/cột nào
- lan_noi_dung: lẫn chữ không thuộc văn bản (header/footer trang, dấu chữ ký số, đoạn của văn bản khác,
  ghi chú của nơi đăng như "(Xem tiếp theo Công báo số ...)" — dấu hiệu văn bản bị cắt giữa hai số Công báo)
- ky_tu_loi: lỗi mã hoá, chữ dính nhau, mất dấu, ký tự lạ
- trung_lap: một đoạn văn bị lặp lại
- thieu_noi_dung: câu dẫn "như sau:" / bảng / danh sách được báo trước NGAY TRONG đoạn nhưng không có nội dung"""


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _parse(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    out = json.loads(m.group(0)) if m else {}
    # Model đôi khi trả "quy_pham|bang_tra_cuu" — lấy loại đầu tiên hợp lệ.
    loai = next((x for x in re.findall(r"[a-z_]+", str(out.get("loai", ""))) if x in LOAI), "?")
    loi = sorted({x for x in out.get("loi") or [] if x in LOI})
    return {"loai": loai, "loi": loi, "ghi_chu": str(out.get("ghi_chu", ""))[:300]}


# FPT giới hạn 100 RPM mỗi model (đo 2026-10-17: 6 luồng không giới hạn → 1004/1248 lời gọi 429)
# và production dùng CHUNG key này làm LLM chính — chạy sát trần là đẩy app thật sang Gemini.
# ponytail: một khoảng cách tối thiểu toàn cục giữa hai lời gọi, đủ cho một script chạy tay.
_RATE_LOCK = threading.Lock()
_NEXT_CALL = [0.0]


def _throttle(rpm: float) -> None:
    with _RATE_LOCK:
        wait = _NEXT_CALL[0] - time.monotonic()
        _NEXT_CALL[0] = max(_NEXT_CALL[0], time.monotonic()) + 60.0 / rpm
    if wait > 0:
        time.sleep(wait)


def audit_one(chunk, models: list[str], rpm: float = 80) -> dict:
    m = chunk.metadata
    where = (f"Điều {m['dieu']}" + (f" (đoạn tách {m['clause_uid'].rsplit('_s', 1)[-1]})" if "_s" in m.get("clause_uid", "") else "")
             if isinstance(m.get("dieu"), int) else "Phụ lục / biểu mẫu")
    prompt = PROMPT.format(so_hieu=m.get("so_hieu", ""), title=m.get("title", ""), where=where, text=chunk.text)
    err = ""
    for attempt in range(4):
        for model in models:  # FPT-only: lời gọi này là để model FPT đọc, không rơi về Gemini
            _throttle(rpm)
            try:
                return {**_parse(_fpt_call(model, prompt, timeout=90).text), "model": model}
            except Exception as exc:  # noqa: BLE001 — thử model kế, ghi lỗi nếu hết
                err = f"{model}: {str(exc)[:150]}"
        if not err.split(": ", 1)[-1].startswith("429"):
            break
        time.sleep(15 * (attempt + 1))  # cả hai model đều 429: chờ cửa sổ RPM trôi
    return {"loai": "?", "loi": [], "ghi_chu": "", "model": "", "error": err}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="*", help="chỉ các doc_id này")
    ap.add_argument("--limit", type=int, default=0, help="tối đa N đoạn mỗi văn bản")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rpm", type=float, default=80, help="tổng lời gọi/phút (FPT: 100 RPM mỗi model, dùng chung key với production)")
    ap.add_argument("--as-of", default="2026-10-17")
    args = ap.parse_args()

    s = get_settings()
    models = [x.strip() for x in s.fpt_models.split(",") if x.strip()]
    if not s.fpt_api_key or not models:
        logger.error("Thiếu FPT_API_KEY / FPT_MODELS")
        return 2

    per_doc = discover_lao_dong(ROOT / "docs/corpus/corpus_manifest.yaml", ROOT / "data/raw/lao_dong", args.as_of)
    items = []
    for doc_id, chunks in per_doc:
        if args.docs and doc_id not in args.docs:
            continue
        items += [(doc_id, c) for c in (chunks[: args.limit] if args.limit else chunks)]

    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not row.get("error"):
                cache[row["key"]] = row
    todo = [(d, c) for d, c in items if _key(c.text) not in cache]
    logger.info("{} đoạn, {} đã có trong cache, gọi FPT cho {}", len(items), len(items) - len(todo), len(todo))

    CACHE.parent.mkdir(exist_ok=True)
    with CACHE.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        for (doc_id, c), res in zip(todo, pool.map(lambda ic: audit_one(ic[1][1], models[ic[0] % len(models):] + models[:ic[0] % len(models)], args.rpm), enumerate(todo))):
            row = {"key": _key(c.text), "doc_id": doc_id, "dieu": c.metadata.get("dieu"),
                   "clause_uid": c.metadata.get("clause_uid", ""), **res}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            cache[row["key"]] = row

    rows = [{**cache[_key(c.text)], "doc_id": d, "clause_uid": c.metadata.get("clause_uid", "")} for d, c in items]
    flagged = [r for r in rows if r["loi"]]
    summary = {
        "doan": len(rows),
        "loi_goi_model": sum(1 for r in rows if r.get("error")),
        "loai": dict(Counter(r["loai"] for r in rows)),
        "loi": dict(Counter(x for r in rows for x in r["loi"])),
        "doan_co_loi": len(flagged),
        "theo_van_ban": {d: {"doan": n, "co_loi": sum(1 for r in flagged if r["doc_id"] == d)}
                         for d, n in Counter(r["doc_id"] for r in rows).items()},
        "model": dict(Counter(r.get("model", "") for r in rows)),
    }
    REPORT.write_text(json.dumps({"summary": summary, "flagged": flagged}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
