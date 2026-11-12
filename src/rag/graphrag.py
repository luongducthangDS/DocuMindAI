"""
GraphRAG trên đồ thị điều luật: hai lớp nằm trên đồ thị dẫn chiếu của src/rag/legal_graph.py.

1. Phả hệ phiên bản (lineage) — cạnh "điều X của văn bản cũ ⇢ điều Y của văn bản thay thế nó"
   (Luật BHXH 58/2014 Điều 73 ⇢ 41/2024 Điều 98). Quan hệ văn bản lấy từ manifest (`thay_the` /
   `bi_thay_the_boi`); quan hệ điều phải suy ra vì luật mới đánh số lại: hai điều chọn nhau là tốt nhất
   (mutual best) theo 0.5·cosine vector + 0.5·Jaccard tiêu đề, bỏ cặp điểm < `_LINEAGE_MIN`. Chấm tay mẫu
   ngẫu nhiên 40 cặp (2026-11-10, gemini-embedding-001): ~37/40 đúng; cặp sai là tiêu đề lệch nghĩa một
   từ ("… chậm" ⇢ "… tự nguyện chậm") hoặc điểm thấp. Vì vậy cạnh lineage chỉ là ỨNG VIÊN cho agent chọn
   (kèm tiêu đề hai phía), không bao giờ tự nối vào ngữ cảnh.
   Dùng cho câu hỏi "thay đổi gì so với luật cũ": lọc hiệu lực đã loại văn bản cũ khỏi truy hồi — không có
   cạnh này thì không so sánh được.

2. Cộng đồng chủ đề (community) — Louvain trên đồ thị gộp: cạnh dẫn chiếu (trọng số = số lần dẫn), cạnh
   lineage (2), hai điều liền kề trong cùng chương (0.5). Ra các cụm xuyên văn bản: "thai sản" gom Luật BHXH
   2014 + 2024 + NĐ 115/2015 + TT 59/2015 + TT 12/2025 — thứ mà cây Chương/Mục không có. Mỗi cụm có nhãn +
   tóm tắt do LLM viết TỪ TIÊU ĐỀ các điều (không từ trí nhớ), dựng offline, kiểm số liệu bịa.
   Tóm tắt chỉ để agent ĐỊNH HƯỚNG (bước map của global search) — không bao giờ vào prompt trả lời: cụm trộn
   luật cũ và mới, nên bằng chứng luôn là đoạn văn bản thật đã qua `ctx.allows` (tenant/ACL/hiệu lực).

Dựng: `python scripts/build_graphrag.py` → data/graph/{article_lineage,communities}.json (commit, đọc được,
chấm tay được). Runtime chỉ đọc hai file đó + chỉ mục node của legal_graph — không phụ thuộc model embedding
đang chạy (production FPT, local Gemini).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from loguru import logger

GRAPH_DIR = Path(__file__).resolve().parents[2] / "data" / "graph"
LINEAGE_PATH = GRAPH_DIR / "article_lineage.json"
COMMUNITIES_PATH = GRAPH_DIR / "communities.json"
_LINEAGE_MIN = 0.70

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_STOP = {"của", "và", "các", "cho", "theo", "đối", "với", "trong", "được", "là", "có", "không", "những", "về",
         "thì", "khi", "nào", "gì", "bao", "nhiêu", "người", "này", "đó", "từ", "đến", "một", "như", "thế"}

# (doc, dieu) → [(doc, dieu, score, title_khac)] — hai chiều: "truoc" (phiên bản cũ), "sau" (phiên bản mới)
_PREV: dict[tuple[str, int], list[dict]] = {}
_NEXT: dict[tuple[str, int], list[dict]] = {}
_COMMUNITIES: list[dict] = []


def _tokens(s: str) -> set[str]:
    return {w for w in _WORD_RE.findall((s or "").lower()) if w not in _STOP and not w.isdigit()}


# ── Dựng offline ───────────────────────────────────────────────────────────────

def doc_lineage(manifest: dict, in_corpus: set[str]) -> list[tuple[str, str]]:
    """[(doc cũ, doc thay thế)] giữa các văn bản cùng có trong corpus."""
    by_so = {str(d.get("so_hieu")).lower(): d["doc_id"] for d in manifest["documents"] if d["doc_id"] in in_corpus}
    pairs = set()
    for d in manifest["documents"]:
        if d["doc_id"] not in in_corpus:
            continue
        for s in d.get("thay_the") or []:
            if str(s).lower() in by_so:
                pairs.add((by_so[str(s).lower()], d["doc_id"]))
        for s in d.get("bi_thay_the_boi") or []:
            if str(s).lower() in by_so:
                pairs.add((d["doc_id"], by_so[str(s).lower()]))
    return sorted(pairs)


def first_chunks(rows) -> dict[str, dict[int, tuple[str, dict]]]:
    """doc_id → {Điều: (chunk_id, meta) của đoạn đầu điều}. rows = vector_backend.fetch_all_chunks."""
    arts: dict[str, dict[int, tuple[str, dict]]] = defaultdict(dict)
    for cid, _text, m in rows:
        try:
            d = int(m.get("dieu"))
        except (TypeError, ValueError):
            continue
        cur = arts[m["doc_id"]].get(d)
        if cur is None or m.get("clause_uid", "") < cur[1].get("clause_uid", ""):
            arts[m["doc_id"]][d] = (cid, m)
    return arts


def match_articles(arts: dict, vectors: dict, pairs: list[tuple[str, str]]) -> list[dict]:
    import numpy as np

    out = []
    for old, new in pairs:
        A, B = arts.get(old, {}), arts.get(new, {})
        ka = [k for k in sorted(A) if A[k][0] in vectors]
        kb = [k for k in sorted(B) if B[k][0] in vectors]
        if not ka or not kb:
            continue
        va = np.array([vectors[A[k][0]] for k in ka], dtype=np.float32)
        vb = np.array([vectors[B[k][0]] for k in kb], dtype=np.float32)
        va /= np.linalg.norm(va, axis=1, keepdims=True)
        vb /= np.linalg.norm(vb, axis=1, keepdims=True)
        cos = va @ vb.T
        ta = [_tokens(A[k][1].get("dieu_tieu_de")) for k in ka]
        tb = [_tokens(B[k][1].get("dieu_tieu_de")) for k in kb]
        jac = np.array([[len(x & y) / max(len(x | y), 1) for y in tb] for x in ta], dtype=np.float32)
        s = 0.5 * cos + 0.5 * jac
        for i, k in enumerate(ka):
            j = int(s[i].argmax())
            if int(s[:, j].argmax()) == i and s[i, j] >= _LINEAGE_MIN:
                out.append({"old": [old, k], "new": [new, kb[j]], "score": round(float(s[i, j]), 3),
                            "cos": round(float(cos[i, j]), 3), "jaccard": round(float(jac[i, j]), 3),
                            "old_title": A[k][1].get("dieu_tieu_de", ""), "new_title": B[kb[j]][1].get("dieu_tieu_de", "")})
    return out


def detect_communities(nodes, lineage: list[dict], resolution: float = 4.0, seed: int = 7) -> list[dict]:
    """nodes: có .text/.metadata (đã nạp vào legal_graph.build_index). Trả cụm ≥ 2 điều, lớn trước."""
    import networkx as nx

    from src.rag import legal_graph as lg

    G = nx.Graph()
    meta: dict[tuple[str, int], dict] = {}
    by_chapter: dict[tuple[str, str], set[int]] = defaultdict(set)

    def bump(a, b, w):
        G.add_edge(a, b, weight=G.get_edge_data(a, b, {}).get("weight", 0) + w)

    for n in nodes:
        m = n.metadata or {}
        d = lg._dieu(m)
        if d is None or not m.get("doc_id"):
            continue
        k = (m["doc_id"], d)
        G.add_node(k)
        if k not in meta or m.get("clause_uid", "") < meta[k].get("clause_uid", ""):
            meta[k] = m
        by_chapter[(m["doc_id"], m.get("chuong") or "")].add(d)
        for r in lg.references(n.text or "", m):
            bump(k, r, 1.0)
    for (doc, _), ds in by_chapter.items():
        ds = sorted(ds)
        for a, b in zip(ds, ds[1:]):
            bump((doc, a), (doc, b), 0.5)
    for e in lineage:
        bump(tuple(e["old"]), tuple(e["new"]), 2.0)

    comms = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    out = []
    for c in sorted(comms, key=lambda c: (-len(c), sorted(c))):
        if len(c) < 2:
            continue
        sub = G.subgraph(c)
        members = sorted(c, key=lambda k: (-sub.degree(k, weight="weight"), k))
        out.append({
            "id": f"c{len(out):03d}",
            "docs": sorted({d for d, _ in c}),
            # theo độ trung tâm trong cụm — điều "trục" trước
            "members": [{"doc_id": d, "dieu": n, "title": meta.get((d, n), {}).get("dieu_tieu_de", ""),
                         "so_hieu": meta.get((d, n), {}).get("so_hieu", "")} for d, n in members],
        })
    return out


_SUMMARY_PROMPT = """Dưới đây là danh sách các điều luật thuộc MỘT cụm chủ đề trong kho văn bản lao động, \
bảo hiểm xã hội Việt Nam (mỗi dòng: số hiệu văn bản — Điều N. tiêu đề). Viết:
- "label": tên chủ đề ngắn gọn, tối đa 10 từ;
- "summary": 2–3 câu nói cụm này quy định những vấn đề gì, thuộc những văn bản nào (nêu số hiệu).
CHỈ dựa vào các tiêu đề dưới đây. KHÔNG nêu con số, mức hưởng, thời hạn nào không có trong tiêu đề.
Trả về DUY NHẤT một dòng JSON: {{"label": "...", "summary": "..."}}

{lines}"""
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


def community_lines(c: dict, limit: int = 40) -> str:
    return "\n".join(f"{m['so_hieu']} — Điều {m['dieu']}. {m['title']}" for m in c["members"][:limit])


def summarize(c: dict, generate) -> dict:
    """generate(prompt) -> str. Trả {label, summary, issues}; issues = số liệu không có trong tiêu đề."""
    lines = community_lines(c)
    raw = generate(_SUMMARY_PROMPT.format(lines=lines))
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    label, summary = str(data.get("label", "")).strip(), str(data.get("summary", "")).strip()
    allowed = set(_NUM_RE.findall(lines))
    bad = sorted({n for n in _NUM_RE.findall(label + " " + summary) if n not in allowed})
    return {"label": label, "summary": summary, "issues": [f"số không có trong tiêu đề: {', '.join(bad)}"] if bad else []}


# ── Runtime ────────────────────────────────────────────────────────────────────

def load(lineage_path: Path = LINEAGE_PATH, communities_path: Path = COMMUNITIES_PATH) -> tuple[int, int]:
    """Nạp hai file đã dựng. Thiếu file = tính năng tắt (log WARNING), không làm hỏng khởi động."""
    prev, nxt = defaultdict(list), defaultdict(list)
    try:
        for e in json.loads(lineage_path.read_text(encoding="utf-8")):
            o, n = tuple(e["old"]), tuple(e["new"])
            nxt[o].append({"key": n, "score": e["score"], "title": e["new_title"]})
            prev[n].append({"key": o, "score": e["score"], "title": e["old_title"]})
    except FileNotFoundError:
        logger.warning("GraphRAG: {} chưa có — tool so sánh phiên bản tắt", lineage_path.name)
    try:
        comms = json.loads(communities_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("GraphRAG: {} chưa có — global search tắt", communities_path.name)
        comms = []
    _PREV.clear(); _PREV.update(prev)
    _NEXT.clear(); _NEXT.update(nxt)
    _COMMUNITIES.clear(); _COMMUNITIES.extend(comms)
    for c in _COMMUNITIES:
        # Cụm chưa tóm tắt được (LLM hết quota lúc dựng): nhãn tất định từ tiêu đề các điều trục — vẫn dẫn đường được.
        c["label"] = c.get("label") or "; ".join(m["title"] for m in c["members"][:3])
        c["_tok"] = _tokens(f"{c['label']} {c.get('summary', '')} " + " ".join(m["title"] for m in c["members"]))
    return sum(len(v) for v in _NEXT.values()), len(_COMMUNITIES)


def _article_node(key: tuple[str, int], allows):
    """Đoạn đầu của điều `key` mà `allows` cho qua (phiên bản nào cũng được, miễn đúng tenant/ACL)."""
    from src.rag import legal_graph as lg

    live = [n for n in lg._ARTICLES.get(key, ()) if allows(n.metadata or {})]
    return min(live, key=lambda n: (n.metadata or {}).get("clause_uid", "")) if live else None


def relevance(query: str):
    """node → điểm liên quan với câu hỏi: cosine với vector đoạn (legal_graph._VEC, đã nạp lúc khởi động) khi có,
    cộng trùng từ tiêu đề để phân xử. Embedding câu hỏi trúng cache (nhánh dense vừa embed đúng câu này).
    Đo 2026-11-10 (gr_g01): chỉ trùng từ tiêu đề kéo "Giải quyết hưởng chế độ thai sản" lên trước "Trợ cấp một lần
    khi sinh con" — tiêu đề luật quá ngắn để xếp hạng bằng từ."""
    from src.rag import legal_graph as lg

    q_tok = _tokens(query)
    qvec = None
    if lg._VEC:
        try:
            import numpy as np

            from src.rag.context import degraded_flags
            from src.rag.embedder import get_embedder

            if "dense_skipped" not in degraded_flags():
                v = np.asarray(get_embedder().get_query_embedding(query), dtype=np.float32)
                qvec = v / (float(np.linalg.norm(v)) or 1.0)
        except Exception:  # embedding hỏng: xếp theo từ
            qvec = None

    def score(node) -> float:
        m = node.metadata or {}
        lex = len(q_tok & _tokens(m.get("dieu_tieu_de") or m.get("dieu_header") or "")) / 100
        vec = lg._VEC.get(getattr(node, "node_id", None)) if qvec is not None else None
        return (float(qvec @ vec) if vec is not None else 0.0) + lex

    return score


def version_candidates(chunks: list, ctx, limit: int = 6, query: str | None = None) -> list[tuple[object, dict]]:
    """(node, info) — phiên bản trước/sau của các điều đang có, chưa có mặt. Chỉ kiểm tenant/ACL, KHÔNG kiểm
    hiệu lực: phiên bản cũ đã hết hiệu lực chính là thứ câu hỏi so sánh cần. info mang hướng + điều gốc.
    Có `query` thì xếp theo độ liên quan trước khi cắt `limit` — gr_c11: cặp cần (58/2014 Điều 54 ⇢ 41/2024 Điều 64)
    xuất phát từ đoạn hạng 9 nên bị cắt khi xếp theo thứ tự đoạn."""
    from src.rag.context import RetrievalContext

    acl = RetrievalContext(tenant_id=ctx.tenant_id, acl_labels=ctx.acl_labels)
    same_tenant = lambda m: (str(m.get("tenant_id", "public")) in acl.visible_tenants
                             and str(m.get("acl_label", "public")) in acl.acl_labels)
    present = {(c.metadata.get("doc_id"), _int(c.metadata.get("dieu"))) for c in chunks}
    out, seen = [], set()
    for c in chunks:
        src = (c.metadata.get("doc_id"), _int(c.metadata.get("dieu")))
        for direction, table in (("truoc", _PREV), ("sau", _NEXT)):
            for e in sorted(table.get(src, ()), key=lambda e: -e["score"]):
                if e["key"] in present or e["key"] in seen:
                    continue
                node = _article_node(e["key"], same_tenant)
                if node is not None:
                    seen.add(e["key"])
                    out.append((node, {"direction": direction, "of": c.metadata.get("clause_uid"),
                                       "of_label": f"Điều {src[1]} {c.metadata.get('so_hieu', '')}".strip(),
                                       "score": e["score"]}))
    if query and len(out) > limit:
        rel = relevance(query)
        out.sort(key=lambda t: -rel(t[0]))
    return out[:limit]


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def community_candidates(query: str, k: int = 6) -> list[dict]:
    """Bước map của global search: cụm khớp câu hỏi nhất theo từ (BM25-lite: trùng từ / log độ dài cụm).
    ponytail: chấm từ vựng, không embedding — tóm tắt dựng offline bằng model khác model đang chạy."""
    import math

    q = _tokens(query)
    if not q or not _COMMUNITIES:
        return []
    scored = [(len(q & c["_tok"]) / math.log(2 + len(c["_tok"])), c) for c in _COMMUNITIES]
    return [c for s, c in sorted(scored, key=lambda t: -t[0]) if s > 0][:k]


def community_articles(c: dict, query: str, allows, present: set, n: int = 3) -> list:
    """Bước reduce: các điều CÒN HIỆU LỰC (allows) của cụm, liên quan câu hỏi nhất (relevance)."""
    nodes = []
    for m in c["members"]:
        key = (m["doc_id"], m["dieu"])
        if key not in present and (node := _article_node(key, allows)) is not None:
            nodes.append(node)
    rel = relevance(query)
    return sorted(nodes, key=lambda nd: -rel(nd))[:n]
