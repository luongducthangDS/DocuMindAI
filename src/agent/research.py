"""
Vòng tìm kiếm agentic: sau lượt truy hồi đầu, LLM đọc những gì đã có rồi QUYẾT ĐỊNH đủ để trả lời hay phải tìm thêm.

Mỗi vòng (tối đa `agent_research_rounds`), một lời gọi LLM trả JSON:
  • enough  — đủ thì dừng, không tốn gì thêm;
  • fetch   — lấy đích danh điều luật mà các đoạn đã có dẫn chiếu tới. LLM chỉ CHỌN mã R1, R2… trong danh sách
              ứng viên do đồ thị dẫn chiếu (src/rag/legal_graph.py) đã giải sẵn — văn bản đúng thời kỳ, đúng
              phiên bản hiệu lực, đúng quyền tenant. LLM không tự viết số hiệu nên không bịa được văn bản;
  • search  — một câu tìm kiếm con cho phần câu hỏi chưa có đoạn nào nói tới (retrieve_with_context, cùng ctx).
  • versions — (chỉ câu hỏi so sánh/thay đổi, _CHANGE_RE) lấy PHIÊN BẢN TRƯỚC/SAU của điều đang có qua cạnh phả hệ
              của src/rag/graphrag.py (58/2014 Điều 73 ⇢ 41/2024 Điều 98). Lọc hiệu lực đã loại văn bản cũ khỏi truy
              hồi, nên không có tool này thì câu "luật mới khác gì luật cũ" chỉ thấy một nửa. Đoạn lấy về mang nhãn
              hiệu lực trong prompt (generator._build_context) để LLM không trả lời bằng luật cũ.
  • overview — (chỉ câu hỏi tổng quát, _BROAD_RE) global search kiểu GraphRAG: LLM chọn cụm chủ đề C* theo nhãn +
              tóm tắt (bước map), hệ thống lấy các điều CÒN HIỆU LỰC của cụm khớp câu hỏi nhất (bước reduce). Tóm tắt
              chỉ để định hướng, không vào prompt trả lời.
Mỗi vòng tính lại ứng viên trên tập đoạn mới (điều vừa lấy có thể dẫn tiếp) — multi-hop nhiều bước.
Ca thúc đẩy: mh_01 — Điều 23 Luật BHXH 2024 dẫn 5 điều, đồ thị xếp Điều 67 ("Điều chỉnh lương hưu") hạng 3 nên
graph_expand=1 không kéo về; LLM đọc câu hỏi "điều chỉnh … trên cơ sở nào" thì chọn được nó.

Lỗi LLM/JSON ⇒ giữ nguyên các đoạn (fail open), giống grader.
"""
from __future__ import annotations

import json
import re
import time

from loguru import logger

_CANDIDATES = 8          # số điều dẫn chiếu đưa cho LLM chọn
_MAX_FETCH = 2
_SEARCH_TOP = 3          # số đoạn mới tối đa lấy từ một câu tìm kiếm con
_EXCERPT_CHARS = 500     # mỗi đoạn đã có: LLM chỉ cần biết đoạn nói về gì, không cần đủ văn bản
_MAX_VERSIONS = 2
_MAX_OVERVIEW = 2         # số cụm chọn tối đa; mỗi cụm góp _OVERVIEW_ARTICLES điều
_OVERVIEW_ARTICLES = 3
# Câu hỏi so sánh theo thời gian / tổng quát — chỉ khi đó mới đưa tool versions / overview cho LLM. Câu thường không
# khớp ⇒ không thêm ứng viên ⇒ ngân sách lời gọi LLM giữ nguyên (tests/test_llm_budget.py).
# Đo 2026-11-10: 16/16 câu data/eval/graphrag_questions.json khớp; legal_qa_200 + multi-hop (231 câu): CHANGE 0,
# BROAD 9 (câu "gồm những chế độ nào" — đúng là câu tổng quát). Bản đầu bắt cả "thay đổi cơ cấu", "khác nhau".
_CHANGE_RE = re.compile(
    r"thay đổi (?:gì|thế nào|ra sao|như thế nào|những gì|so|điều kiện)|có gì khác|khác (?:gì|thế nào) so|"
    r"so với (?:luật|quy định|nghị định|bộ luật|trước|mức|bản|văn bản)[^?]*?(?:cũ|trước|20\d\d)|trước đây|"
    r"(?:luật|quy định|bản|văn bản|mức) cũ|điểm mới|mới so với|sắp có hiệu lực|sửa đổi (?:gì|những|ra sao|thế nào)",
    re.IGNORECASE)
_BROAD_RE = re.compile(
    r"(?:những|các) (?:\w+\s+){1,4}nào\b|tổng quan|liệt kê|gồm (?:những )?(?:gì|nào)|bao gồm (?:những )?(?:gì|nào)|"
    r"những nội dung gì", re.IGNORECASE)

_PROMPT = """Bạn là trợ lý tra cứu pháp luật lao động, bảo hiểm xã hội Việt Nam. Nhiệm vụ: quyết định các đoạn văn bản \
đã tìm được có ĐỦ để trả lời đầy đủ câu hỏi chưa. Chưa trả lời câu hỏi.

Câu hỏi: {query}

Các đoạn đã có — chỉ hiện PHẦN ĐẦU mỗi đoạn; TOÀN VĂN các đoạn này đã nằm trong tài liệu dùng để trả lời, \
nên đừng tìm lại nội dung của chính các điều này chỉ vì phần trích bị cắt:
{excerpts}

Các điều được những đoạn trên dẫn chiếu tới nhưng CHƯA có nội dung:
{candidates}
{extra}
Quy tắc:
- Nếu câu trả lời phụ thuộc nội dung một điều trong danh sách dẫn chiếu (ví dụ đoạn chỉ ghi "theo Điều N" mà \
câu hỏi hỏi đúng nội dung đó), chọn mã của điều đó vào "fetch" (tối đa {max_fetch}).
- Nếu câu hỏi có một phần mà không đoạn nào và không điều dẫn chiếu nào nói tới, viết MỘT câu tìm kiếm ngắn \
bằng thuật ngữ pháp lý vào "search".
{extra_rules}- Đã đủ thì "enough": true và để trống mọi danh sách.

Trả về DUY NHẤT một dòng JSON:
{{"enough": true hoặc false, "fetch": ["R1"], "search": ["..."],{extra_json} "reason": "lý do 1 câu"}}"""

_VERSIONS_BLOCK = """
Phiên bản TRƯỚC/SAU của các điều đã có (văn bản cũ đã bị thay thế, hoặc văn bản mới thay nó) — dùng khi câu hỏi \
so sánh, hỏi thay đổi so với quy định cũ:
{lines}
"""
_VERSIONS_RULE = ('- Nếu câu hỏi so sánh / hỏi thay đổi giữa quy định cũ và mới, chọn mã V* của phiên bản cần đối chiếu '
                  'vào "versions" (tối đa {n}).\n')
_OVERVIEW_BLOCK = """
Các cụm chủ đề trong kho văn bản (mỗi cụm gom các điều liên quan, có thể thuộc nhiều văn bản) — dùng cho câu hỏi \
tổng quát cần nhiều điều:
{lines}
"""
_OVERVIEW_RULE = ('- Nếu câu hỏi tổng quát cần liệt kê/khái quát nhiều quy định mà các đoạn đã có chưa đủ, chọn mã C* '
                  'của cụm phù hợp vào "overview" (tối đa {n}).\n')


def _excerpts(chunks: list) -> str:
    return "\n".join(f"[{i}] {' '.join((c.text or '')[:_EXCERPT_CHARS].split())}" for i, c in enumerate(chunks, 1))


def _mention(text: str, dieu: int) -> str:
    """Câu chữ quanh chỗ nhắc tới Điều `dieu` — cho LLM thấy đoạn dẫn chiếu nói gì về nó.
    ponytail: tìm "Điều N", không có thì số N trơn (danh sách "Điều 35, 36 và 37"); trượt thì không có trích."""
    m = re.search(rf"[Đđ]iều\s+{dieu}\b", text) or re.search(rf"(?<=[\s,]){dieu}\b", text)
    if not m:
        return ""
    return " ".join(text[max(0, m.start() - 160): m.end() + 80].split())


def _candidates(chunks: list, ctx) -> list[tuple[str, object, list[str], str]]:
    """(mã R*, node, clause_uid các đoạn dẫn tới, dòng mô tả cho prompt)."""
    from src.rag import legal_graph

    index = {c.metadata.get("clause_uid"): i for i, c in enumerate(chunks, 1)}
    by_uid = {c.metadata.get("clause_uid"): c for c in chunks}
    out = []
    for k, (node, via) in enumerate(legal_graph.expand(chunks, ctx.allows, limit=_CANDIDATES), 1):
        m = node.metadata or {}
        head = m.get("dieu_header") or f"Điều {m.get('dieu')}"
        src = next((u for u in via if u in index), None)
        quote = _mention(by_uid[src].text, int(m.get("dieu"))) if src else ""
        line = f"R{k}. {head} — {m.get('so_hieu') or m.get('title', '')}"
        if src:
            line += f" (dẫn từ [{index[src]}]: \"…{quote}…\")" if quote else f" (dẫn từ [{index[src]}])"
        out.append((f"R{k}", node, via, line))
    return out


def _parse(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    as_list = lambda v: [str(x).strip() for x in (v if isinstance(v, list) else [v] if v else []) if str(x).strip()]
    return {"enough": bool(data.get("enough")), "fetch": as_list(data.get("fetch")),
            "search": as_list(data.get("search"))[:1], "versions": as_list(data.get("versions")),
            "overview": as_list(data.get("overview")), "reason": str(data.get("reason", ""))[:200]}


def _version_candidates(query: str, chunks: list, ctx) -> list[tuple[str, object, dict, str]]:
    """(mã V*, node, info, dòng prompt) — chỉ cho câu hỏi so sánh/thay đổi."""
    if not _CHANGE_RE.search(query):
        return []
    from src.rag import graphrag

    out = []
    for k, (node, info) in enumerate(graphrag.version_candidates(chunks, ctx, query=query), 1):
        m = node.metadata or {}
        when = f"hiệu lực {m.get('effective_from', '?')} → {m.get('effective_to', '?')}"
        side = "phiên bản trước" if info["direction"] == "truoc" else "phiên bản sau"
        head = m.get("dieu_header") or f"Điều {m.get('dieu')}"
        out.append((f"V{k}", node, info, f"V{k}. {head} — {m.get('so_hieu', '')} ({side} của {info['of_label']}; {when})"))
    return out


def _overview_candidates(query: str) -> list[tuple[str, dict, str]]:
    """(mã C*, cụm, dòng prompt) — chỉ cho câu hỏi tổng quát."""
    from src.config import get_settings

    if not get_settings().agent_overview or not _BROAD_RE.search(query):
        return []
    from src.rag import graphrag

    return [(f"C{k}", c, f"C{k}. {c['label']} — {c.get('summary') or ''} (văn bản: {', '.join(c['docs'])})")
            for k, c in enumerate(graphrag.community_candidates(query), 1)]


def _decide(query: str, chunks: list, cands: list, vcands: list = (), ccands: list = ()) -> dict | None:
    from src.config import get_settings
    from src.rag.generator import gemini_generate

    extra = extra_rules = extra_json = ""
    if vcands:
        extra += _VERSIONS_BLOCK.format(lines="\n".join(c[-1] for c in vcands))
        extra_rules += _VERSIONS_RULE.format(n=_MAX_VERSIONS)
        extra_json += ' "versions": ["V1"],'
    if ccands:
        extra += _OVERVIEW_BLOCK.format(lines="\n".join(c[-1] for c in ccands))
        extra_rules += _OVERVIEW_RULE.format(n=_MAX_OVERVIEW)
        extra_json += ' "overview": ["C1"],'
    prompt = _PROMPT.format(query=query, excerpts=_excerpts(chunks) or "(không có)",
                            candidates="\n".join(line for *_, line in cands) or "(không có)", max_fetch=_MAX_FETCH,
                            extra=extra, extra_rules=extra_rules, extra_json=extra_json)
    models = [m.strip() for m in get_settings().gemini_judge_models.split(",") if m.strip()]
    from src.rag.context import mark_degraded

    try:
        decision = _parse(gemini_generate(prompt, models=models or None, log_input=query, name="agent-research"))
        if decision is None:
            mark_degraded("agent_research_failed")
        return decision
    except Exception as exc:  # LLM hỏng: trả lời bằng những gì đã có
        # Cờ degraded: fail open mà không để dấu thì eval không tách được "agent quyết định không làm gì" với "agent
        # không chạy" — 2026-11-10 hai lượt đo GraphRAG có câu agent im lặng vì Gemini 429, report vẫn sạch.
        mark_degraded("agent_research_failed")
        logger.warning("agent-research LLM failed, answering with what was retrieved: {}", exc)
        return None


def _fetch(codes: list[str], cands: list) -> list:
    from llama_index.core.schema import NodeWithScore

    from src.rag.retriever import nodes_to_chunks

    picked = [(node, via) for code, node, via, _ in cands if code in codes][:_MAX_FETCH]
    out = nodes_to_chunks([NodeWithScore(node=n, score=0.0) for n, _ in picked])
    for c, (_, via) in zip(out, picked):
        c.metadata["graph_via"] = via         # _budget xếp chỗ trước + nhãn "dẫn chiếu từ đoạn [j]"
        c.metadata["agent_step"] = "fetch"
    return out


def _versions(codes: list[str], vcands: list, ctx) -> list:
    from llama_index.core.schema import NodeWithScore

    from src.rag.retriever import nodes_to_chunks

    picked = [(node, info) for code, node, info, _ in vcands if code in codes][:_MAX_VERSIONS]
    out = nodes_to_chunks([NodeWithScore(node=n, score=0.0) for n, _ in picked])
    for c, (_, info) in zip(out, picked):
        c.metadata.update(agent_step="versions", lineage_of=info["of"], lineage_dir=info["direction"],
                          lineage_of_label=info["of_label"], in_force_at_query=bool(ctx.allows(c.metadata)))
    return out


def _overview(codes: list[str], ccands: list, query: str, ctx, seen_keys: set) -> list:
    from llama_index.core.schema import NodeWithScore

    from src.rag import graphrag
    from src.rag.retriever import nodes_to_chunks

    out = []
    for code, c, _ in ccands:
        if code not in codes or len(out) >= _MAX_OVERVIEW * _OVERVIEW_ARTICLES:
            continue
        nodes = graphrag.community_articles(c, query, ctx.allows, seen_keys, _OVERVIEW_ARTICLES)
        for ch in nodes_to_chunks([NodeWithScore(node=n, score=0.0) for n in nodes]):
            ch.metadata.update(agent_step="overview", community=c["label"])
            seen_keys.add((ch.metadata.get("doc_id"), graphrag._int(ch.metadata.get("dieu"))))
            out.append(ch)
    return out


def _search(query: str, ctx, seen: set) -> list:
    from src.rag.retriever import retrieve_with_context

    out = []
    for c in retrieve_with_context(query, ctx):
        if c.metadata.get("clause_uid") not in seen and len(out) < _SEARCH_TOP:
            c.metadata["agent_query"] = query  # _budget xếp chỗ trước, trace biết đoạn từ câu tìm nào
            c.metadata.pop("graph_via", None)   # đồ thị của lượt con, không trỏ tới số đoạn của prompt này
            out.append(c)
    return out


def research(query: str, chunks: list, ctx, rounds: int | None = None) -> tuple[list, list[dict]]:
    """Trả (các đoạn sau khi tìm thêm, các bước cho UI [{label, detail, ms}]).

    Bước này chỉ BỔ SUNG: lỗi bất kỳ (không chỉ LLM) ⇒ trả lời bằng những gì đã có, không làm hỏng lượt hỏi."""
    from src.config import get_settings

    rounds = get_settings().agent_research_rounds if rounds is None else rounds
    steps: list[dict] = []
    try:
        return _loop(query, chunks, ctx, rounds, steps)
    except Exception as exc:
        logger.warning("agent-research failed, answering with what was retrieved: {}", exc)
        return chunks, steps


def _loop(query: str, chunks: list, ctx, rounds: int, steps: list[dict]) -> tuple[list, list[dict]]:
    from src.rag import graphrag

    for _ in range(max(rounds, 0)):
        t0 = time.time()
        cands = _candidates(chunks, ctx)
        vcands = _version_candidates(query, chunks, ctx)
        ccands = _overview_candidates(query)
        # Không có ứng viên nào ⇒ không hỏi LLM: câu rõ ràng giữ ngân sách 1 lời gọi
        # (tests/test_llm_budget.py); truy hồi lệch hẳn đã có vòng grade → reformulate lo.
        if not (cands or vcands or ccands):
            break
        decision = _decide(query, chunks, cands, vcands, ccands)
        ms = lambda: int((time.time() - t0) * 1000)
        if decision is None:
            break
        acts = decision["fetch"] or decision["search"] or decision["versions"] or decision["overview"]
        # Có mã hành động thì làm, kể cả khi kèm "enough": true — Gemini hay trả {"enough": true, "versions": ["V1"]}
        # nghĩa là "đủ khi có thêm V1" (gr_c04, gr_c05 bị bỏ phiên bản cũ vì vậy). enough chỉ dừng vòng sau.
        if not acts:
            steps.append({"label": "Đánh giá đủ thông tin", "detail": decision["reason"] or "Đủ để trả lời", "ms": ms()})
            break
        seen = {c.metadata.get("clause_uid") for c in chunks}
        added = [c for c in _fetch(decision["fetch"], cands) if c.metadata.get("clause_uid") not in seen]
        added += [c for c in _versions(decision["versions"], vcands, ctx) if c.metadata.get("clause_uid") not in seen]
        seen |= {c.metadata.get("clause_uid") for c in added}
        keys = {(c.metadata.get("doc_id"), graphrag._int(c.metadata.get("dieu"))) for c in chunks + added}
        added += [c for c in _overview(decision["overview"], ccands, query, ctx, keys)
                  if c.metadata.get("clause_uid") not in seen]
        seen |= {c.metadata.get("clause_uid") for c in added}
        for s in decision["search"]:
            added += _search(s, ctx, seen)
        label = lambda c: f"Điều {c.metadata.get('dieu')} {c.metadata.get('so_hieu', '')}".strip()
        step = lambda c: c.metadata.get("agent_step")
        what = [label(c) for c in added if step(c) == "fetch"]
        what += [f"{label(c)} (phiên bản {'trước' if c.metadata['lineage_dir'] == 'truoc' else 'sau'})"
                 for c in added if step(c) == "versions"]
        what += sorted({f'cụm "{c.metadata["community"]}"' for c in added if step(c) == "overview"})
        what += [f'tìm "{q}"' for q in decision["search"]]
        steps.append({"label": "Tìm thêm", "detail": f"{'; '.join(what)} — {decision['reason']}", "ms": ms()})
        logger.info("agent-research: +{} đoạn ({}) | '{}'", len(added), "; ".join(what), query[:60])
        if added:
            chunks = chunks + added
        if not added or decision["enough"]:
            break
    return chunks, steps
