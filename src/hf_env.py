"""
Trỏ các biến môi trường HuggingFace về cache trong repo.

Vì sao cần: HF_HOME của máy dev trỏ tới `G:\\My Drive\\HF_Cache_Models` (Google
Drive, thường không mount). Khi ổ đó offline, hf_xet ghi log vào HF_HOME và gây
crash native (0xC0000005) trên Windows, hoặc model im lặng tải lại từ đầu.

Trước đây mỗi entrypoint tự lặp lại đoạn set env này — 8 bản sao, 4 biến thể khác
nhau (gán đè vs setdefault, có/không HF_HUB_OFFLINE), và ít nhất một lần gây lỗi
thật: `from eval.temporal_eval import score_context` kéo theo HF_HUB_OFFLINE=1 đặt
ở module level, khiến harness khác không tải được model mới.

Nguyên tắc ở đây: module này KHÔNG tự chạy gì lúc import. Entrypoint nào cần thì
gọi tường minh, và gọi TRƯỚC khi import sentence_transformers/transformers.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE_DIR = REPO_ROOT / "data" / "hf_cache"

# Mọi biến HF đều trỏ về cùng một thư mục. Đây là layout mà cache trên đĩa đang
# dùng (data/hf_cache/models--*), không phải layout chuẩn HF_HOME/hub — đổi sang
# chuẩn sẽ khiến các model đã tải thành "không tìm thấy" và tải lại vài GB.
_HF_VARS = ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "SENTENCE_TRANSFORMERS_HOME")


def use_local_hf_cache(*, offline: bool = True, create: bool = False) -> Path:
    """Trỏ mọi biến môi trường HF về `data/hf_cache` trong repo.

    Gọi trước khi import sentence_transformers / transformers.

    offline=True  — cấm gọi ra Hub (mặc định: đường chạy production/eval chỉ dùng
                    model đã cache, và cấm ra mạng thì lỗi thiếu cache lộ ngay
                    thay vì âm thầm tải vài GB).
    offline=False — cho phép tải model chưa có (ingest lần đầu, thử model mới).
    create=True   — tạo thư mục cache nếu chưa có.

    Trả về đường dẫn cache.
    """
    if create:
        HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for key in _HF_VARS:
        os.environ[key] = str(HF_CACHE_DIR)

    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        # Gỡ hẳn, không để giá trị từ lần gọi trước (hoặc từ module khác) chặn tải.
        os.environ.pop("HF_HUB_OFFLINE", None)

    return HF_CACHE_DIR


def resolve_cached_model(repo_id: str) -> str | None:
    """Đường dẫn thư mục model đã nằm trong cache của repo, hoặc None nếu chưa có.

    Dùng khi thư viện KHÔNG cho truyền cache_folder (ví dụ SentenceTransformerRerank):
    truyền thẳng đường dẫn thay cho repo id thì không còn phụ thuộc biến môi trường.

    Cần thiết vì `huggingface_hub` đọc HF_HOME/HF_HUB_CACHE đúng một lần lúc import
    và giữ nguyên giá trị đó; mọi lần set os.environ sau đó đều vô tác dụng, và cache
    vẫn bị tra ở HF_HOME của máy (ở đây là một ổ Google Drive thường offline).
    """
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo_id, cache_dir=str(HF_CACHE_DIR), local_files_only=True)
    except Exception:
        return None
