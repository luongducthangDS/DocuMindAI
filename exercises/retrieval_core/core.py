"""Lõi retrieval tự viết — xem README.md trong cùng thư mục.

KHÔNG dùng agent sinh code cho file này. KHÔNG import từ src/ hay llama_index.
Chỉ stdlib. Mỗi `raise NotImplementedError` là một phần việc phải tự làm.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field

K1 = 1.5
B = 0.75
RRF_K = 60

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


# ---------------------------------------------------------------------------
# Phần 1 — ĐÃ CHO SẴN. Đây là mẫu phong cách: hằng số có lý do, docstring nói
# *vì sao* chứ không nhắc lại tên hàm.
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Tách token cho BM25 và cho việc đếm độ dài chunk.

    NFC trước khi tách: cùng một chữ "ệ" có hai cách mã hoá (dựng sẵn / tổ hợp),
    tuỳ nguồn văn bản mà ra cách nào. Không chuẩn hoá thì "nghỉ" từ file crawl về
    và "nghỉ" người dùng gõ vào là hai token khác nhau, df đếm sai, BM25 sai theo.

    Giữ nguyên dấu tiếng Việt (không strip accent): "lãi" và "lai" là hai từ khác
    nghĩa, gộp lại là mất recall đổi lấy nhiễu.

    Không tách từ ghép ("người lao động" -> 3 token). Đây là một đánh đổi có ý
    thức: BM25 trên unigram tiếng Việt kém hơn trên từ ghép, nhưng tách từ cần
    dependency ngoài, và nhánh dense trong hybrid đã gánh phần ngữ nghĩa.
    """
    return _TOKEN_RE.findall(unicodedata.normalize("NFC", text).lower())


# ---------------------------------------------------------------------------
# Phần 2 — Chunking. 5 bất biến ở README §Phần 2.
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    text: str           # đã bao gồm dòng header ở đầu (bất biến 2)
    dieu_header: str    # ví dụ "Điều 12."
    body_tokens: list[str] = field(default_factory=list)  # phần thân, không gồm header


_DIEU_RE = re.compile(r"^\s*(Điều\s+\d+\s*[.:])", re.MULTILINE)


def split_dieu(text: str) -> list[tuple[str, str]]:
    """Trả về [(header, body), ...] theo thứ tự xuất hiện.

    Văn bản trước `Điều` đầu tiên bị bỏ (thường là quốc hiệu, nơi ban hành).
    """
    raise NotImplementedError("Phần 2")


def chunk_by_dieu(
    text: str,
    max_tokens: int = 220,
    overlap_tokens: int = 40,
) -> list[Chunk]:
    """Cắt văn bản thành chunk, không để một chunk trộn nội dung hai Điều.

    Ghi lý do chọn max_tokens/overlap_tokens của mình vào notes.md — hai con số
    này sau sẽ bị hỏi.
    """
    raise NotImplementedError("Phần 2")


# ---------------------------------------------------------------------------
# Phần 3 — BM25 Okapi. Công thức ở README §Phần 3, test neo số thật.
# ---------------------------------------------------------------------------
class BM25Index:
    """Sparse index. Tính sẵn idf + độ dài doc lúc build; query đi qua inverted index."""

    def __init__(self, docs: list[str], k1: float = K1, b: float = B) -> None:
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.doc_tokens: list[list[str]] = [tokenize(d) for d in docs]
        self.n = len(docs)
        # TODO Phần 3: self.avgdl, self.doc_len, self.df, self.idf, self.inverted
        raise NotImplementedError("Phần 3")

    def idf_of(self, term: str) -> float:
        """ln(1 + (N - df + 0.5) / (df + 0.5)). Tách riêng để test neo được."""
        raise NotImplementedError("Phần 3")

    def score(self, query: str, doc_id: int) -> float:
        raise NotImplementedError("Phần 3")

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """[(doc_id, score)] giảm dần. Doc điểm 0 không cần trả về."""
        raise NotImplementedError("Phần 3")


# ---------------------------------------------------------------------------
# Phần 4 — RRF.
# ---------------------------------------------------------------------------
def rrf_fuse(
    rankings: list[list[str]],
    k: int = RRF_K,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Hợp nhất nhiều danh sách xếp hạng bằng Reciprocal Rank Fusion.

    rankings: mỗi phần tử là một list doc_id đã xếp hạng (tốt nhất đứng đầu).
    Trả về [(doc_id, rrf_score)] giảm dần theo điểm.
    """
    raise NotImplementedError("Phần 4")


# ---------------------------------------------------------------------------
# Phần 5 — Rerank bằng cross-encoder (scorer tiêm từ ngoài).
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    doc_id: str
    text: str
    score: float = 0.0


def rerank(
    query: str,
    candidates: list[Candidate],
    scorer,
    top_n: int = 8,
) -> list[Candidate]:
    """Chấm lại ứng viên bằng cross-encoder, trả top_n.

    scorer: callable nhận list[(query, doc_text)] và trả list[float] cùng độ dài.
    Gọi scorer đúng MỘT lần cho cả batch — lý do vì sao, ghi vào notes.md.
    """
    raise NotImplementedError("Phần 5")
