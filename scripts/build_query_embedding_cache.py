"""
scripts/build_query_embedding_cache.py — embed câu hỏi gold MỘT LẦN ở nơi đủ RAM.

Chạy ở đâu: Colab / Kaggle / VM FPT — bất kỳ máy nào load nổi
`AITeamVN/Vietnamese_Embedding` (~2.2GB trọng số + torch). KHÔNG chạy được trên
máy dev đang thiếu RAM, và đó chính là lý do script này tồn tại (xem DEC-0006).

Sau khi chạy, commit file cache; `eval/scoring_ab.py --query-embeddings <file>`
chạy được toàn bộ eval retrieval mà không load model nào.

Dùng trên Colab (chép nguyên 4 ô dưới đây):

    # [1] cài đặt — chỉ cần sentence-transformers; truyền --model tường minh thì
    #     script không import src.config, nên không cần .env và không cần pydantic
    !pip install -q sentence-transformers

    # [2] clone ĐÚNG nhánh chứa script này (nó chưa có trên main/develop)
    !git clone -b feature/eval-metric-clause-uid --depth 1 https://github.com/luongducthangDS/DocuMindAI.git
    %cd DocuMindAI

    # [3] embed — ghi ra /content cho dễ tải. Thêm --dtype fp16 khi đo phương án F1
    #     của DEC-0006 (chạy lần nữa, đổi tên file output).
    !python scripts/build_query_embedding_cache.py --model AITeamVN/Vietnamese_Embedding --dtype fp32 --output /content/temporal_30q_fp32.json

    # [4] tải file về máy
    from google.colab import files
    files.download('/content/temporal_30q_fp32.json')

Sau đó **commit từ máy local**: chép file đã tải vào `data/eval/query_embeddings/`
rồi commit như bình thường. KHÔNG push từ Colab.

Dùng tại chỗ (máy đủ RAM):

    python scripts/build_query_embedding_cache.py

Script này KHÔNG đọc ChromaDB, KHÔNG ghi gì vào corpus, KHÔNG push gì lên git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.query_cache import build_payload  # noqa: E402

DEFAULT_GOLD = _REPO_ROOT / "data" / "eval" / "temporal_questions.json"
DEFAULT_OUTPUT = _REPO_ROOT / "data" / "eval" / "query_embeddings" / "temporal_30q.json"


def _git_commit() -> str:
    """Commit đang checkout, kèm cờ nếu working tree bẩn — để sau này biết cache
    sinh ra từ trạng thái repo nào."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=_REPO_ROOT, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            cwd=_REPO_ROOT, check=True,
        ).stdout.strip()
        return f"{head}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def _default_model_name() -> str:
    """Tên model khi không truyền `--model`.

    Thứ tự: biến môi trường → `.env` của repo → `src.config` (pydantic).

    Không đi thẳng vào `src.config`: nó import `pydantic_settings`, thứ KHÔNG có
    sẵn trên Colab/Kaggle (ở đó chỉ cài `sentence-transformers`). Trước đây bỏ
    `--model` trên Colab là gặp `ModuleNotFoundError: No module named
    'pydantic_settings'` — một lỗi chẳng nói gì về việc thiếu tên model.

    Vẫn không hard-code tên model ở đây: `EMBEDDING_MODEL` là nguồn sự thật duy
    nhất, nên nếu không đọc được ở đâu cả thì báo lỗi và yêu cầu `--model`.
    """
    from_env = os.getenv("EMBEDDING_MODEL", "").strip()
    if from_env:
        return from_env

    dotenv = _REPO_ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "EMBEDDING_MODEL" and value.strip():
                return value.strip().strip('"').strip("'")

    try:
        from src.config import get_settings

        return get_settings().embedding_model
    except Exception as exc:
        raise SystemExit(
            f"Không xác định được model ({type(exc).__name__}). Truyền tường minh:\n"
            f"  python {Path(__file__).name} --model AITeamVN/Vietnamese_Embedding ..."
        ) from None


def _weights_fingerprint(model) -> str:
    """SHA-256 rút gọn của trọng số đã load.

    Tên model trên Hub không cố định nội dung — repo có thể được đẩy bản mới. Vân
    tay này cho biết cache sinh từ đúng bộ trọng số nào, độc lập với tên.
    """
    try:
        import torch

        h = hashlib.sha256()
        for _, tensor in sorted(model.state_dict().items()):
            h.update(tensor.detach().to(torch.float32).cpu().numpy().tobytes())
        return h.hexdigest()[:32]
    except Exception:
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model", default="",
        help="mặc định đọc EMBEDDING_MODEL từ biến môi trường / .env / settings. "
             "Trên Colab nên truyền tường minh cho chắc",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--dtype", choices=("fp32", "fp16"), default="fp32",
        help="độ chính xác trọng số lúc embed. fp16 phục vụ đo phương án F1 trong "
             "DEC-0006 (~1.1GB thay vì ~2.2GB); vector fp16 phải đối chiếu cosine "
             "với fp32 trước khi tin, vì corpus đã index bằng fp32",
    )
    args = parser.parse_args()

    model_name = args.model or _default_model_name()

    questions = [q["question"] for q in json.loads(args.gold.read_text(encoding="utf-8"))]
    print(f"Gold: {args.gold.name} — {len(questions)} câu")
    print(f"Model: {model_name}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    if args.dtype == "fp16":
        # Nửa độ chính xác: trọng số ~1.1GB thay vì ~2.2GB. Vector sinh ra KHÔNG
        # bit-identical với fp32 — đó chính là thứ phép đo F1 cần định lượng, nên
        # dtype được ghi vào cache để không ai trộn hai loại vào cùng một báo cáo.
        model = model.half()
    embeddings = model.encode(
        questions,
        batch_size=args.batch_size,
        normalize_embeddings=False,
        show_progress_bar=True,
    ).tolist()

    payload = build_payload(
        questions,
        embeddings,
        model_name,
        extra={
            "created_at": date.today().isoformat(),
            "gold_set": str(args.gold.relative_to(_REPO_ROOT)).replace("\\", "/"),
            "git_commit": _git_commit(),
            "weights_sha256_32": _weights_fingerprint(model),
            "dtype": args.dtype,
            "normalize_embeddings": False,
        },
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    size_kb = args.output.stat().st_size // 1024
    print()
    print(f"Đã ghi {len(questions)} vector {payload['dim']} chiều ({args.dtype})")
    print(f"  đường dẫn : {args.output.resolve()}")
    print(f"  kích thước: {size_kb} KB")
    print(f"  model     : {model_name}")
    print(f"  vân tay   : {payload['weights_sha256_32']}")
    print()
    print("Tải file này về máy rồi commit từ local — đừng push từ Colab.")


if __name__ == "__main__":
    main()
