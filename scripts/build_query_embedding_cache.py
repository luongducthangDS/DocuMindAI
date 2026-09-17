"""
scripts/build_query_embedding_cache.py — embed câu hỏi gold MỘT LẦN ở nơi đủ RAM.

Chạy ở đâu: Colab / Kaggle / VM FPT — bất kỳ máy nào load nổi
`AITeamVN/Vietnamese_Embedding` (~2.2GB trọng số + torch). KHÔNG chạy được trên
máy dev đang thiếu RAM, và đó chính là lý do script này tồn tại (xem DEC-0006).

Sau khi chạy, commit file cache; `eval/scoring_ab.py --query-embeddings <file>`
chạy được toàn bộ eval retrieval mà không load model nào.

Dùng trên Colab:

    !git clone https://github.com/luongducthangDS/DocuMindAI.git
    %cd DocuMindAI
    !pip install -q sentence-transformers
    !python scripts/build_query_embedding_cache.py \
        --gold data/eval/temporal_questions.json \
        --output data/eval/query_embeddings/temporal_30q.json

Dùng tại chỗ (máy đủ RAM):

    python scripts/build_query_embedding_cache.py

Script này KHÔNG đọc ChromaDB và KHÔNG ghi gì vào corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
        help="mặc định lấy từ EMBEDDING_MODEL trong .env / settings",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    model_name = args.model
    if not model_name:
        from src.config import get_settings
        model_name = get_settings().embedding_model

    questions = [q["question"] for q in json.loads(args.gold.read_text(encoding="utf-8"))]
    print(f"Gold: {args.gold.name} — {len(questions)} câu")
    print(f"Model: {model_name}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
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
            "normalize_embeddings": False,
        },
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    size_kb = args.output.stat().st_size // 1024
    print(f"Đã ghi {len(questions)} vector {payload['dim']} chiều → {args.output} ({size_kb} KB)")


if __name__ == "__main__":
    main()
