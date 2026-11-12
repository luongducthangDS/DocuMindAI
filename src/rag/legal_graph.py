"""
Đồ thị dẫn chiếu giữa các điều luật trong corpus — dựng từ chính văn bản các chunk.

Nút là một điều (doc_id, số Điều); cạnh là câu "… quy định tại Điều 98 của Luật Bảo hiểm xã hội"
trong một điều khác. Dùng để kéo vào ngữ cảnh điều mà các đoạn đã truy hồi được dẫn chiếu tới:
g2_b031 — đoạn hạng 3 (NĐ 159/2025 Điều 14) dẫn "Điều 98 của Luật Bảo hiểm xã hội", nhưng chính
Điều 98 (điều kiện hưởng lương hưu, 15 năm) đứng hạng 28 nên không tới LLM (đo 2026-10-19).

Tên văn bản ("Luật Bảo hiểm xã hội") ứng với nhiều doc_id theo thời kỳ (58/2014 → 41/2024), nên
dẫn chiếu được giải theo ngày hiệu lực của chính đoạn dẫn chiếu: văn bản cùng tên có hiệu lực
muộn nhất mà không muộn hơn đoạn đó. Văn bản không có trong corpus (Bộ luật Lao động 2012 mà các
văn bản 2015 dẫn tới, Bộ luật Tố tụng hình sự…) thì không giải được và bị bỏ qua.

ponytail: regex trên văn bản, không phải parser pháp lý — bắt "Điều N [, M và K] [của <văn bản>]";
dẫn chiếu kiểu "điều này", "chương này", "mục 2" không bắt. Đủ cho dẫn chiếu tới cấp Điều.

Chấm tay trên corpus thật (2026-10-19, mẫu ngẫu nhiên): precision cạnh cùng văn bản 25/25, khác văn bản
24/25 (cạnh sai đã sửa — _DESCRIBED_RE), cạnh do _SENTENCE_NAME_RE 24/24; recall dẫn chiếu tới điều CÓ trong
corpus 32/32 (20 đoạn, 10 luật + 10 NĐ/TT). Bản trước đó: ~120/1170 cạnh sai (10%), toàn là "Điều N" trơn bị
gán cho chính văn bản. Sửa regex thì chấm lại — cạnh sai không báo lỗi, chỉ kéo điều lạc đề vào ngữ cảnh.
"""

from __future__ import annotations

import re
from collections import defaultdict

# "Điều 35, 36 và 37", "Điều 98 và Điều 64", "Điều 5 và khoản 3 Điều 7" — "của <văn bản>" phía sau áp
# cho cả danh sách (bản đầu chỉ bắt "Điều 98", rồi coi "và Điều 64 của Luật …" là tự dẫn chiếu).
_REF_RE = re.compile(
    r"\b[Đđ]iều\s+(?P<list>\d+(?:\s*(?:,|và|hoặc)\s*(?:(?:điểm\s+[a-zđ]\s+)?khoản\s+\d+\s+)?(?:[Đđ]iều\s+)?\d+)*)",
    re.IGNORECASE)
_ITEM_SPLIT = re.compile(r"\s*(?:,|\bvà\b|\bhoặc\b)\s*", re.IGNORECASE)
# Văn bản đứng sau danh sách: "Điều 98 của Luật …", "Điều 4 Nghị định số 135/2020/NĐ-CP", và cả khi cách bằng
# phần danh sách mà _REF_RE không bắt — "a) Điều 6; khoản 3 Điều 7 và khoản 2 Điều 11 của Nghị định số
# 115/2015/NĐ-CP", "khoản 1 Điều 35 và điểm đ, e khoản 2 Điều 35 Bộ luật Lao động". Ở giữa chỉ được có từ của
# danh sách dẫn chiếu (mỗi nhánh một ký tự/một từ, loại trừ nhau — không backtrack bùng nổ).
_NAME_AFTER_RE = re.compile(
    r"(?:\s|[,;]|\d|\b(?:và|hoặc|các|điểm|khoản|[Đđ]iều)\b|\b[a-zđ]\b)*"
    # tên phân biệt hoa/thường: "quyết định tại Điều 35", "pháp luật về …" không phải tên văn bản
    r"(?:của\s+)?(?P<name>(?-i:Bộ luật|Luật|Nghị định|Thông tư|Quyết định)\b[^,.;()\n]*)", re.IGNORECASE)
_SO_HIEU_RE = re.compile(r"\bsố\s+(\d{1,3}/\d{4}/[A-ZĐa-zđ0-9\-]+)")
_THIS_RE = re.compile(r"^(?:Bộ luật|Luật|Nghị định|Thông tư|Quyết định)\s+này\b", re.IGNORECASE)
# Câu dẫn mà các mục sau nói về văn bản KHÁC: "các điều, khoản sau đây của Bộ luật Lao động:" (NĐ 145/2020 Điều
# 1), "Sửa đổi, bổ sung một số điều của Luật An toàn, vệ sinh lao động số 84/2015/QH13 như sau:" (Luật BHXH 2024
# Điều 139), "Sửa đổi, bổ sung Điều 32 của Bộ luật Tố tụng dân sự …:" (BLLĐ Điều 219). "khoản 3 Điều 49" ở mục
# bên dưới là của văn bản trong câu dẫn — chủ thể ở xa.
_LIST_LEAD_RE = re.compile(
    r"(?:sau đây|một số điều|sửa đổi, bổ sung\b[^:\n]*?)\s+của\s+(?:Bộ luật|Luật|Nghị định|Thông tư)\b[^:\n]*:",
    re.IGNORECASE)
# Nghị định/thông tư liệt kê điều của luật mẹ, tên luật ở CUỐI câu: "nghỉ hằng tuần theo Điều 111, … Điều 114,
# khoản 1 Điều 115; ⏎ thời gian … theo Điều 128 của Bộ luật Lao động." (NĐ 145/2020 Điều 8) — "Điều N" trơn lấy
# tên văn bản đầu tiên phía sau trong cùng câu. Phân biệt hoa/thường: "pháp luật về …" không phải tên văn bản.
_SENTENCE_NAME_RE = re.compile(r"[^.]*?(?P<name>(?:Bộ luật|Luật|Nghị định|Thông tư|Quyết định)\b[^,.;()\n]*)")
# Phần trích nội dung sửa đổi “…”: "Luật này", "Điều 2" trong đó là của văn bản BỊ sửa, không phải văn bản sửa.
_QUOTE_RE = re.compile(r"“[^”]*")
# Dẫn chiếu nằm trong phần mô tả một văn bản khác: "Nghị định số 41/2013/NĐ-CP ngày 08 tháng 5 năm 2013 của Chính
# phủ quy định chi tiết thi hành Điều 220 của Bộ luật Lao động" (NĐ 145/2020 Điều 114 bãi bỏ) — "Bộ luật Lao
# động" ở đây là bộ luật 2012, giải theo ngày của văn bản được mô tả chứ không theo ngày của đoạn dẫn.
_DESCRIBED_RE = re.compile(
    r"số\s+\d+/\d{4}/\S+\s+ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})\s+của\s[^.;\n()]*?"
    r"(?:quy định|hướng dẫn)[^.;\n()]*$", re.IGNORECASE)

# (doc_id, Điều) → chunk node của điều đó, mọi phiên bản; dựng một lần lúc khởi động.
_ARTICLES: dict[tuple[str, int], list] = {}
# số đoạn dẫn tới mỗi điều — "Điều 2. Đối tượng áp dụng" bị dẫn từ 37 đoạn, một phiếu cho nó gần như vô nghĩa
_INDEG: dict[tuple[str, int], int] = {}
# node_id → vector đã chuẩn hoá (float32: 1249 × 1024 ≈ 5 MB; list Python thì ~46 MB)
_VEC: dict = {}
# tên văn bản (thường) → [(ngày bắt đầu hiệu lực, doc_id)] tăng dần; so_hieu → doc_id
_BY_NAME: dict[str, list[tuple[str, str]]] = {}
_BY_SO_HIEU: dict[str, str] = {}


def _dieu(meta: dict) -> int | None:
    try:
        return int(meta.get("dieu"))
    except (TypeError, ValueError):
        return None


def build_index(nodes, vectors: dict | None = None) -> int:
    """Nạp chỉ mục từ các node đã có (BM25 corpus) và vector của chúng (theo node_id) nếu có.
    Trả số điều được lập chỉ mục."""
    articles: dict[tuple[str, int], list] = defaultdict(list)
    starts: dict[str, tuple[str, str]] = {}  # doc_id → (title, ngày bắt đầu sớm nhất)
    so_hieu: dict[str, str] = {}
    for n in nodes:
        meta = getattr(n, "metadata", None) or {}
        doc, dieu = meta.get("doc_id"), _dieu(meta)
        if not doc:
            continue
        if dieu is not None:
            articles[(doc, dieu)].append(n)
        title, start = (meta.get("title") or "").strip().lower(), meta.get("effective_from") or ""
        if title and (doc not in starts or start < starts[doc][1]):
            starts[doc] = (title, start)
        if meta.get("so_hieu"):
            so_hieu[meta["so_hieu"].lower()] = doc
    by_name: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for doc, (title, start) in starts.items():
        by_name[title].append((start, doc))
    _ARTICLES.clear(); _ARTICLES.update(articles)
    _BY_NAME.clear(); _BY_NAME.update({k: sorted(v) for k, v in by_name.items()})
    _BY_SO_HIEU.clear(); _BY_SO_HIEU.update(so_hieu)
    indeg: dict[tuple[str, int], int] = defaultdict(int)
    for n in nodes:
        for key in set(references(getattr(n, "text", "") or "", getattr(n, "metadata", None) or {})):
            indeg[key] += 1
    _INDEG.clear(); _INDEG.update(indeg)
    _INDEG[None] = len(nodes)  # N của IDF = số đoạn — khoá None không trùng điều nào
    _VEC.clear()
    if vectors:
        import numpy as np

        for nid, v in vectors.items():
            a = np.asarray(v, dtype=np.float32)
            norm = float(np.linalg.norm(a))
            if norm:
                _VEC[nid] = a / norm
    return len(_ARTICLES)


def _resolve(name: str, citing_doc: str, citing_from: str) -> str | None:
    if _THIS_RE.match(name):
        return citing_doc
    m = _SO_HIEU_RE.search(name)
    if m:
        return _BY_SO_HIEU.get(m.group(1).lower())
    low = name.strip().lower()
    for title in sorted(_BY_NAME, key=len, reverse=True):  # "luật bảo hiểm xã hội" trước "luật"
        if low.startswith(title):
            live = [doc for start, doc in _BY_NAME[title] if start <= (citing_from or "9999")]
            return live[-1] if live else None
    return None


def _article_numbers(lst: str) -> list[int]:
    """Số Điều trong danh sách đã khớp — bỏ số khoản ("khoản 3 Điều 7" → 7)."""
    nums = []
    for item in _ITEM_SPLIT.split(lst.strip()):
        m = re.search(r"(?:[Đđ]iều\s+)?(\d+)\s*$", item)
        if m and (item.strip().isdigit() or re.search(r"[Đđ]iều\s+\d+\s*$", item)):
            nums.append(int(m.group(1)))
    return nums


def references(text: str, meta: dict) -> list[tuple[str, int]]:
    """Các điều (doc_id, Điều) mà một đoạn dẫn chiếu tới, theo thứ tự xuất hiện, không lặp.

    "Điều N" trơn (không nêu văn bản) ở luật là của chính luật; ở nghị định/thông tư là của văn bản được nêu
    tên đầu tiên phía sau trong cùng câu (_SENTENCE_NAME_RE), không có thì bỏ. Trong phần trích “…”, sau câu
    dẫn _LIST_LEAD_RE và trong biểu mẫu phụ lục ("QUYẾT ĐỊNH: Điều 1…") chỉ giữ dẫn chiếu nêu đích danh văn
    bản khác: mất một cạnh thì chỉ thiếu ngữ cảnh, cạnh sai kéo điều không liên quan vào câu trả lời.
    Đo trên corpus (2026-10-19): "Điều N trơn = chính văn bản" đúng 6/8 ở luật nhưng 1/22 ở nghị định/thông tư
    (tự dẫn thì chúng ghi "Nghị định này"); bản trước còn gán NĐ 145/2020 Điều 1 → chính nó Điều 21/35/96/98,
    BLLĐ Điều 219 → chính nó Điều 54 (thật ra Luật BHXH 2014), biểu mẫu → Điều 1–11.
    """
    doc, own = meta.get("doc_id"), _dieu(meta)
    is_law = (meta.get("title") or "").lower().startswith(("luật", "bộ luật"))
    lead = _LIST_LEAD_RE.search(text)
    lead_at = lead.end() if lead else len(text)
    quotes = [q.span() for q in _QUOTE_RE.finditer(text)]
    out: list[tuple[str, int]] = []
    # Dòng tiêu đề "Điều N." của chính đoạn cũng khớp regex — bị loại bởi key != (doc, own) bên dưới.
    for m in _REF_RE.finditer(text):
        displaced = own is None or m.start() >= lead_at or any(a <= m.start() < b for a, b in quotes)
        of = _NAME_AFTER_RE.match(text, m.end())
        if not of and not is_law and not displaced:
            of = _SENTENCE_NAME_RE.match(text, m.end())
        name = of.group("name") if of else None
        bare, this = name is None, bool(name and _THIS_RE.match(name))
        # "Nghị định này" ngoài phần trích/câu dẫn/biểu mẫu vẫn là chính văn bản; "Điều N" trơn thì chỉ ở luật.
        if (bare or this) and (displaced or (bare and not is_law)):
            continue
        when = meta.get("effective_from") or ""
        if name and not this:
            d = _DESCRIBED_RE.search(text, 0, m.start())
            if d:
                when = f"{d.group(3)}-{int(d.group(2)):02d}-{int(d.group(1)):02d}"
        target = _resolve(name, doc, when) if name else doc
        if not target:
            continue
        for num in _article_numbers(m.group("list")):
            key = (target, num)
            if key in _ARTICLES and key != (doc, own) and key not in out:
                out.append(key)
    return out


def expand(chunks: list, allows, limit: int = 1, query_vec=None) -> list[tuple]:
    """(node, clause_uid các đoạn dẫn tới nó) của tối đa `limit` điều được `chunks` dẫn chiếu mà chưa có mặt.
    Danh sách thứ hai là provenance của cạnh — trace ghi lại để tra ngược đoạn nối thêm từ đâu ra.

    Đồ thị đề xuất, truy hồi quyết định: mỗi ứng viên xếp theo hai thứ hạng gộp RRF — số đoạn dẫn tới
    nó nhân IDF (log N / in-degree, để "Điều 2. Đối tượng áp dụng" không thắng chỉ vì ai cũng dẫn), và
    cosine với câu hỏi (`query_vec`; không có thì chỉ dùng thứ hạng đầu). Đo lại trên bản trích cạnh đã
    sửa precision (2026-10-19): legal_qa_200 gold trong ngữ cảnh LLM k=0/1/2 = 0.986/0.995/0.995; bộ
    multi-hop 23 câu 0.826/0.826/0.913 (k=1: đoạn kéo về bị ngân sách prompt cắt); đầu cuối multi-hop
    0.913/0.826/0.870 — không cải thiện, nên graph_expand mặc định 0.
    Chỉ lấy phiên bản `allows` cho qua (tenant/ACL + hiệu lực tại ngày tra cứu).
    """
    if not _ARTICLES or limit <= 0:
        return []
    import math

    present = {(c.metadata.get("doc_id"), _dieu(c.metadata)) for c in chunks}
    votes: dict[tuple[str, int], int] = defaultdict(int)
    via: dict[tuple[str, int], list] = defaultdict(list)
    for c in chunks:
        for key in references(c.text, c.metadata):
            if key not in present:
                votes[key] += 1
                via[key].append(c.metadata.get("clause_uid"))
    cands = []
    for key, v in votes.items():
        # .get: upload dựng lại chỉ mục (clear → update) trong lúc câu hỏi khác đang chạy
        live = [n for n in _ARTICLES.get(key, ()) if allows(n.metadata or {})]
        if live:
            # đoạn đầu của điều (điều dài bị tách thành nhiều đoạn — đoạn đầu mang tiêu đề + khoản 1)
            cands.append((key, min(live, key=lambda n: (n.metadata or {}).get("clause_uid", "")), v))
    if not cands:
        return []
    n_total = max(_INDEG.get(None, 1), 1)
    idf_rank = {c[0]: i for i, c in enumerate(sorted(cands, key=lambda c: -c[2] * math.log(n_total / max(_INDEG.get(c[0], 1), 1))))}
    score = {key: 1 / (60 + idf_rank[key]) for key, _, _ in cands}
    if query_vec is not None and _VEC:
        import numpy as np

        q = np.asarray(query_vec, dtype=np.float32)
        q = q / (float(np.linalg.norm(q)) or 1.0)
        cos = {key: float(q @ _VEC[getattr(n, "node_id", None)]) if getattr(n, "node_id", None) in _VEC else -1.0
               for key, n, _ in cands}
        for i, key in enumerate(sorted(cos, key=lambda k: -cos[k])):
            score[key] += 1 / (60 + i)
    order = sorted(cands, key=lambda c: -score[c[0]])
    return [(n, via[key]) for key, n, _ in order[:limit]]
