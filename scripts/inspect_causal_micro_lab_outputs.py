#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def record_category(record: dict[str, Any]) -> str:
    verification = record.get("verification") or {}
    output = str(record.get("output") or "")
    if record.get("request_error"):
        return "request_error"
    if not output.strip():
        return "empty"
    if not verification.get("parse_valid"):
        return "parse_invalid"
    if not verification.get("syntax_valid"):
        return "syntax_invalid"
    if not verification.get("evidence_consistent"):
        return "evidence_inconsistent"
    if verification.get("is_currently_valid_mode"):
        return "valid_mode"
    return "not_currently_valid"


def compact_text(text: str, *, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n... [truncated]"


def print_record(record: dict[str, Any], *, max_chars: int, show_thinking: bool) -> None:
    verification = record.get("verification") or {}
    metadata = record.get("state_metadata") or {}
    print("=" * 88)
    print(
        "sample_id={sample_id} category={category} M={m} sep={sep} family={family}".format(
            sample_id=record.get("sample_id"),
            category=record_category(record),
            m=metadata.get("valid_mode_count"),
            sep=metadata.get("separation_bucket"),
            family=metadata.get("family_bucket"),
        )
    )
    print(
        "parse={parse} syntax={syntax} evidence={evidence} valid_mode={valid} "
        "mode={mode} seconds={seconds:.2f}".format(
            parse=verification.get("parse_valid"),
            syntax=verification.get("syntax_valid"),
            evidence=verification.get("evidence_consistent"),
            valid=verification.get("is_currently_valid_mode"),
            mode=verification.get("semantic_mode_id"),
            seconds=float(record.get("model_seconds") or 0.0),
        )
    )
    if record.get("request_error"):
        print("\nREQUEST ERROR:")
        print(record["request_error"])
    if verification.get("error"):
        print("\nVERIFIER ERROR:")
        print(verification["error"])
    if show_thinking and record.get("thinking"):
        print("\nTHINKING:")
        print(compact_text(str(record["thinking"]), max_chars=max_chars))
    print("\nOUTPUT:")
    print(compact_text(str(record.get("output") or ""), max_chars=max_chars))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect causal micro-lab eval completions from episodes.jsonl."
    )
    parser.add_argument(
        "episodes",
        type=Path,
        help="Path to episodes.jsonl or episodes.partial.jsonl.",
    )
    parser.add_argument(
        "--category",
        choices=[
            "all",
            "empty",
            "nonempty",
            "parse_invalid",
            "syntax_invalid",
            "evidence_inconsistent",
            "not_currently_valid",
            "valid_mode",
            "request_error",
        ],
        default="all",
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=1600)
    parser.add_argument("--show-thinking", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.episodes)
    categories = Counter(record_category(record) for record in rows)
    print(f"episodes: {len(rows)}")
    for category, count in sorted(categories.items()):
        print(f"{category}: {count}")

    if args.category == "all":
        selected = rows
    elif args.category == "nonempty":
        selected = [record for record in rows if str(record.get("output") or "").strip()]
    else:
        selected = [
            record for record in rows if record_category(record) == args.category
        ]

    print(f"\nselected[{args.category}]: {len(selected)}")
    for record in selected[args.offset : args.offset + args.limit]:
        print_record(
            record,
            max_chars=args.max_chars,
            show_thinking=args.show_thinking,
        )


if __name__ == "__main__":
    main()
