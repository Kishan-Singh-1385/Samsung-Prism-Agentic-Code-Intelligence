"""Run a terminal demo of the official MiniLM AppsRetrieval pipeline."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prism.retrieval.apps_index import AppsRetrievalIndex

WIDTH = 72
PREVIEW_LINES = 5
PREVIEW_COLUMNS = 76


def wrapped(label: str, value: str, width: int = WIDTH) -> str:
    prefix = f"{label}: "
    subsequent = " " * len(prefix)
    return textwrap.fill(
        value,
        width=width,
        initial_indent=prefix,
        subsequent_indent=subsequent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def preview_lines(text: str) -> tuple[list[str], bool]:
    lines = text.splitlines() or [text]
    output = []
    truncated = len(lines) > PREVIEW_LINES
    for line in lines[:PREVIEW_LINES]:
        line = line.expandtabs(4).rstrip()
        if len(line) > PREVIEW_COLUMNS:
            truncated = True
            line = line[: PREVIEW_COLUMNS - 3] + "..."
        output.append(line)
    return output, truncated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--query", help="Retrieve code for an arbitrary natural-language query.")
    selection.add_argument("--demo-query", type=int, help="Predefined demo query number (1–5; 0 remains a legacy alias for 1).")
    parser.add_argument("--top-k", type=int, default=5, help="Number of snippets to display (default: 5).")
    parser.add_argument("--device", default=None, help="Inference device (defaults to config.yaml).")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config.yaml")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    intent = None
    if args.demo_query is not None:
        examples = json.loads((ROOT / "artifacts/demo_queries.json").read_text(encoding="utf-8"))
        if args.demo_query < 0 or args.demo_query > len(examples):
            parser.error(f"--demo-query must be from 1 to {len(examples)} (or 0 for the legacy first-query alias)")
        example_index = 0 if args.demo_query in (0, 1) else args.demo_query - 1
        intent = str(examples[example_index].get("intent") or "")
        query = str(examples[example_index]["query"])
    else:
        query = args.query or ""
        if not query.strip():
            parser.error("--query must contain text")

    rule = "=" * WIDTH
    print(rule)
    print("SAMSUNG PRISM | AGENTIC CODE INTELLIGENCE")
    print(rule)
    if intent:
        print(f"Demo query {1 if args.demo_query == 0 else args.demo_query} of 5 | {intent}")
    print(wrapped("Query", f'"{query}"'))
    print()

    print("Preparing retrieval index and model...")
    # Keep library weight-loading progress and hub diagnostics out of the demo UI.
    previous_log_level = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            index = AppsRetrievalIndex.load(args.config, device=args.device)
    finally:
        logging.disable(previous_log_level)
    corpus_size = len(index.corpus_ids)
    print(f"Searching {corpus_size:,} code snippets...")
    results, latency = index.retrieve(query, top_k=args.top_k)
    print(f"Retrieval latency: {latency:.3f} seconds")
    print()
    print(f"TOP {len(results)} RETRIEVED CODE SNIPPETS")
    print("-" * WIDTH)

    text_by_id = dict(zip(index.corpus_ids, index.corpus_texts, strict=True))
    for rank, result in enumerate(results, start=1):
        document_id = str(result["document_id"])
        score = float(result["similarity_score"])
        code_lines, truncated = preview_lines(text_by_id.get(document_id, ""))
        print(f"[{rank}]  Score: {score:.4f}")
        print(f"     ID: {document_id}")
        print("     Code:")
        if code_lines:
            for line in code_lines:
                print(f"       {line}")
            if truncated:
                print("       ...")
        else:
            print("       (No code preview available)")
        if rank < len(results):
            print()

    top_score = float(results[0]["similarity_score"]) if results else 0.0
    print()
    print(rule)
    print("Query processed successfully")
    print(f"Top result score: {top_score:.4f}" if results else "Top result score: unavailable")
    print(f"Retrieval latency: {latency * 1000:.2f} ms")
    print(f"Corpus searched: {corpus_size:,} snippets")
    print(rule)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
