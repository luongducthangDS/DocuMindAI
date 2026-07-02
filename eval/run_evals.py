"""
eval/run_evals.py — canonical entry point for the DocuMind AI retrieval benchmark.

Runs a four-step ablation: BM25-only → dense-only → hybrid → hybrid+rerank.
Delegates to rag_comparison.main() so all logic stays in one place.

Recommended two-step workflow (avoids burning LLM generation quota on RAGAS re-runs):

  # Step 1 — generate answers, cache them, get local metrics (no RAGAS judge needed)
  python eval/run_evals.py --skip-ragas

  # Step 2 — load cached answers, run RAGAS judge only (no generation LLM needed)
  python eval/run_evals.py

Other useful flags:
  --limit 5                   smoke test (first 5 questions)
  --strategies dense rerank   subset of strategies
  --no-cache                  force regeneration even if cache exists
  --cache-dir path/to/dir     custom cache location (default: reports/answers_cache)

See eval/rag_comparison.py for implementation details.
"""

import os, sys, pathlib

# Force HuggingFace to use local cache (not G:\My Drive) — must be set before any HF imports
_repo_root = pathlib.Path(__file__).parent.parent
_hf_local = _repo_root / "data" / "hf_cache"
if _hf_local.exists():
    os.environ.setdefault("HF_HOME", str(_hf_local))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(_hf_local / "hub"))

sys.path.insert(0, str(_repo_root))
from eval.rag_comparison import main

if __name__ == "__main__":
    main()
