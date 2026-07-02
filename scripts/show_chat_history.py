"""
Pretty-print chat_history.jsonl as readable bullet points.
Usage:
    python scripts/show_chat_history.py           # last 10 entries
    python scripts/show_chat_history.py --n 20    # last 20
    python scripts/show_chat_history.py --all     # all entries
"""

import argparse
import json
import sys
from pathlib import Path

LOG_PATH = Path(__file__).parent.parent / "logs" / "chat_history.jsonl"

# Fix Windows console encoding for Vietnamese text
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def fmt_ts(ts: str) -> str:
    # "2026-06-19T13:26:06.113149+00:00" -> "19/06 13:26:06"
    try:
        date, time_ = ts[:19].split("T")
        y, m, d = date.split("-")
        return f"{d}/{m} {time_}"
    except Exception:
        return ts[:19]


def fmt_answer(text: str, width: int = 90) -> list[str]:
    """Wrap answer into lines, indent continuation."""
    import textwrap
    lines = text.replace("\n\n", "\n").splitlines()
    out = []
    for line in lines:
        if not line.strip():
            continue
        wrapped = textwrap.wrap(line, width)
        out.extend(wrapped if wrapped else [line])
    return out


def print_entry(i: int, entry: dict) -> None:
    error = entry.get("error")
    ts = fmt_ts(entry.get("ts", ""))
    session = entry.get("session_id", "")[-8:]
    query = entry.get("query", "")
    llm = entry.get("used_llm", "?")
    latency = entry.get("latency_ms", 0)
    chunks = entry.get("chunk_count", 0)
    sources = entry.get("source_titles", [])
    answer = entry.get("answer", "") or ""

    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  #{i}  {ts}  │  session: …{session}  │  {llm}  │  {latency}ms  │  {chunks} chunks")
    print(f"{sep}")
    print(f"  Q: {query}")

    if error:
        print(f"  • ERROR: {error}")
    else:
        lines = fmt_answer(answer)
        if lines:
            print(f"  A: {lines[0]}")
            for l in lines[1:]:
                print(f"     {l}")

        if sources:
            unique = list(dict.fromkeys(sources))
            print(f"  • Sources: {', '.join(unique)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of last entries to show")
    parser.add_argument("--all", action="store_true", help="Show all entries")
    args = parser.parse_args()

    if not LOG_PATH.exists():
        print(f"No chat history found at {LOG_PATH}")
        sys.exit(0)

    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    entries = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    total = len(entries)
    if args.all:
        subset = entries
        offset = 1
    else:
        subset = entries[-args.n:]
        offset = max(1, total - len(subset) + 1)

    print(f"\nChat history: {total} total entries, showing {len(subset)}")

    for i, entry in enumerate(subset, start=offset):
        print_entry(i, entry)

    print(f"\n{'─' * 70}\n")


if __name__ == "__main__":
    main()
