"""Từ danh sách file đổi → test nào phải chạy, eval nào phải chạy, câu gold nào bị ảnh hưởng.

Ba nguồn quan hệ, đều tất định (không LLM):
1. Đồ thị import Python (ast) — test import trực tiếp/bắc cầu module đổi.
2. Tham chiếu chuỗi — test/module nhắc tên file dữ liệu đổi (vd. `criteria.json`).
3. Corpus → gold — văn bản `data/raw/lao_dong/<doc_id>.md` đổi ⇒ câu gold trỏ tới doc_id đó.

  python -m src.qa.regression_scope --base main
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_PY_DIRS = ("src", "eval", "scripts", "tests")
GOLD_SETS = ("data/eval/legal_qa_200.json", "data/eval/multihop_questions.json")

# Thay đổi chạm vào chất lượng truy hồi / câu trả lời ⇒ unit test không đủ, phải đo lại.
_RETRIEVAL_PREFIXES = ("src/rag/", "src/ingestion/", "data/raw/", "docs/corpus/", "data/eval/")
_ANSWER_PREFIXES = ("src/rag/generator.py", "src/rag/grader.py", "src/agent/", "src/config.py",
                    "src/guardrails.py")
_REINGEST_PREFIXES = ("data/raw/", "docs/corpus/", "src/ingestion/")
# Đổi những file này = mọi test có thể đổi kết quả.
_RUN_ALL = ("tests/conftest.py", "pytest.ini", "pyproject.toml", "requirements.txt", "requirements-dev.txt")


def changed_files(base: str = "main") -> list[str]:
    """File đổi so với merge-base với `base`, cộng thay đổi chưa commit/chưa track."""
    def git(*args: str) -> list[str]:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return [l.strip() for l in out.splitlines() if l.strip()]
    files = set(git("diff", "--name-only", f"{base}...HEAD"))
    files |= set(git("diff", "--name-only", "HEAD"))
    files |= set(git("ls-files", "--others", "--exclude-standard"))
    return sorted(files)


def _module_name(rel: str) -> str:
    return rel[:-3].replace("/", ".").removesuffix(".__init__")


@lru_cache(maxsize=1)
def _python_files() -> dict[str, str]:
    """module name → repo-relative path, cho mọi file .py dự án (không .venv)."""
    out = {}
    for d in _PY_DIRS:
        for p in (ROOT / d).rglob("*.py"):
            if "__pycache__" not in p.parts:
                rel = p.relative_to(ROOT).as_posix()
                out[_module_name(rel)] = rel
    return out


@lru_cache(maxsize=1)
def _imports() -> dict[str, set[str]]:
    """path → các path dự án mà nó import (cạnh thẳng)."""
    mods = _python_files()
    graph: dict[str, set[str]] = {}
    for mod, rel in mods.items():
        try:
            tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            graph[rel] = set()
            continue
        deps = set()
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for n in names:
                # `from src.rag import retriever` → src.rag.retriever; `import src.rag.x.y` → src.rag.x
                while n and n not in mods:
                    n = n.rpartition(".")[0]
                if n:
                    deps.add(mods[n])
        deps.discard(rel)
        graph[rel] = deps
    return graph


def _dependents(targets: set[str]) -> set[str]:
    """Mọi file .py import (bắc cầu) một trong `targets`, gồm cả targets."""
    rev: dict[str, set[str]] = {}
    for src, deps in _imports().items():
        for d in deps:
            rev.setdefault(d, set()).add(src)
    seen, stack = set(targets), list(targets)
    while stack:
        for parent in rev.get(stack.pop(), ()):
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return seen


def _mentions(name: str) -> set[str]:
    """File .py dự án nhắc tới tên file dữ liệu (theo tên file, không theo đường dẫn đầy đủ)."""
    return {rel for rel in _python_files().values()
            if name in (ROOT / rel).read_text(encoding="utf-8", errors="ignore")}


def _gold_refs(q: dict) -> set[str]:
    clauses = [q.get("source_clause") or "", *q.get("source_clauses", []), *q.get("acceptable_clauses", [])]
    docs = {c.split("__")[0] for c in clauses if c}
    return docs | {q.get("source_doc") or ""} | set(q.get("distractor_docs", []))


def affected_gold(doc_ids: set[str]) -> dict[str, list[str]]:
    out = {}
    for gs in GOLD_SETS:
        p = ROOT / gs
        if p.exists() and doc_ids:
            ids = [q["id"] for q in json.loads(p.read_text(encoding="utf-8")) if _gold_refs(q) & doc_ids]
            if ids:
                out[gs] = ids
    return out


def regression_scope(files: list[str]) -> dict:
    files = [f.replace("\\", "/") for f in files]
    reasons: list[str] = []
    py = {f for f in files if f.endswith(".py") and f in _python_files().values()}
    hit = _dependents(py)
    for f in files:
        if not f.endswith(".py") and "/" in f and not f.startswith(("tests/", "frontend/")):
            refs = _mentions(Path(f).name)
            if refs:
                reasons.append(f"{f}: được nhắc trong {len(refs)} file .py")
                hit |= _dependents(refs)
    tests = sorted(t for t in hit if t.startswith("tests/") and Path(t).name.startswith("test_"))

    run_all = [f for f in files if f in _RUN_ALL]
    if run_all:
        reasons.append(f"{', '.join(run_all)} đổi ⇒ chạy toàn bộ test")

    doc_ids = {Path(f).stem for f in files if f.startswith("data/raw/") and f.endswith(".md")}
    gold = affected_gold(doc_ids)
    evals = []
    if any(f.startswith(_RETRIEVAL_PREFIXES) for f in files):
        evals.append({"name": "retrieval",
                      "cmd": "python eval/temporal_eval.py --gold data/eval/legal_qa_200.json "
                             "--retrieval-only --arms pre_filter --output reports/ci_eval.json "
                             "&& python eval/ci_gate.py reports/ci_eval.json",
                      "reason": "đổi code/dữ liệu truy hồi"})
    if any(f.startswith(_ANSWER_PREFIXES) for f in files):
        evals.append({"name": "e2e",
                      "cmd": "python eval/temporal_eval.py --gold data/eval/legal_qa_200.json "
                             "--arms pre_filter --output reports/eval_e2e_pr.json",
                      "reason": "đổi prompt/agent/cấu hình sinh câu trả lời — unit test không bắt được"})
    for gs, ids in gold.items():
        evals.append({"name": f"gold-subset:{Path(gs).stem}",
                      "cmd": f"python eval/temporal_eval.py --gold {gs} --ids {' '.join(ids)}",
                      "reason": f"{len(ids)} câu gold trỏ tới văn bản đổi ({', '.join(sorted(doc_ids))})"})

    return {
        "changed": files,
        "run_all_tests": bool(run_all),
        "tests": tests,
        "pytest_cmd": "pytest -q" if run_all or not tests else "pytest -q " + " ".join(tests),
        "evals": evals,
        "gold_ids": gold,
        "reingest": any(f.startswith(_REINGEST_PREFIXES) for f in files),
        "reasons": reasons,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("files", nargs="*", help="file đổi; bỏ trống = lấy từ git")
    p.add_argument("--base", default="main")
    args = p.parse_args()
    print(json.dumps(regression_scope(args.files or changed_files(args.base)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
