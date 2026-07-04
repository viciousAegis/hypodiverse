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


def category(record: dict[str, Any], *, generated_tokens: int, cap_tokens: int) -> str:
    verification = record.get("verification") or {}
    output = str(record.get("output") or "")
    near_cap = generated_tokens >= cap_tokens
    if record.get("request_error"):
        return "request_error"
    if verification.get("is_currently_valid_mode"):
        return "valid_mode"
    if not output.strip():
        return "empty_likely_max_tokens" if near_cap else "empty_not_near_max_tokens"
    if not verification.get("parse_valid"):
        return "parse_invalid_likely_max_tokens" if near_cap else "parse_invalid_format"
    if not verification.get("syntax_valid"):
        return "syntax_invalid_likely_max_tokens" if near_cap else "syntax_invalid_format"
    if not verification.get("evidence_consistent"):
        return "evidence_inconsistent"
    return "not_currently_valid"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc failure audit for causal micro-lab episodes. This estimates "
            "max-token failures from saved text because old runs did not log "
            "server finish_reason."
        )
    )
    parser.add_argument("episodes", type=Path)
    parser.add_argument("--tokenizer", help="HF tokenizer/model path for exact token counts.")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--cap-ratio", type=float, default=0.95)
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    rows = read_jsonl(args.episodes)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    cap_tokens = int(args.max_new_tokens * args.cap_ratio)
    counts: Counter[str] = Counter()
    by_m: dict[str, Counter[str]] = {}
    token_values: list[int] = []
    examples: dict[str, list[dict[str, Any]]] = {}

    for record in rows:
        text = "\n".join(
            part
            for part in (
                str(record.get("thinking") or ""),
                str(record.get("output") or ""),
            )
            if part
        )
        if tokenizer is not None:
            generated_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        else:
            generated_tokens = max(0, round(len(text) / 4))
        token_values.append(generated_tokens)
        label = category(record, generated_tokens=generated_tokens, cap_tokens=cap_tokens)
        counts[label] += 1
        metadata = record.get("state_metadata") or {}
        m_label = str(metadata.get("valid_mode_count", "missing"))
        by_m.setdefault(m_label, Counter())[label] += 1
        if len(examples.setdefault(label, [])) < args.examples:
            examples[label].append(
                {
                    "sample_id": record.get("sample_id"),
                    "M": metadata.get("valid_mode_count"),
                    "separation_bucket": metadata.get("separation_bucket"),
                    "generated_tokens": generated_tokens,
                    "output_chars": len(str(record.get("output") or "")),
                    "thinking_chars": len(str(record.get("thinking") or "")),
                    "verifier_error": (record.get("verification") or {}).get("error"),
                    "output": str(record.get("output") or "")[:500],
                }
            )

    total = len(rows)
    likely_max = sum(
        count for label, count in counts.items() if label.endswith("_likely_max_tokens")
    )
    format_fail = counts["parse_invalid_format"] + counts["syntax_invalid_format"]
    empty_short = counts["empty_not_near_max_tokens"]
    payload = {
        "episodes": total,
        "max_new_tokens": args.max_new_tokens,
        "cap_ratio": args.cap_ratio,
        "cap_tokens": cap_tokens,
        "tokenizer": args.tokenizer or "chars/4 heuristic",
        "counts": dict(counts.most_common()),
        "rates": {
            "likely_max_token_failure": likely_max / total if total else 0.0,
            "format_failure_not_near_cap": format_fail / total if total else 0.0,
            "empty_not_near_cap": empty_short / total if total else 0.0,
            "valid_mode": counts["valid_mode"] / total if total else 0.0,
            "evidence_inconsistent": counts["evidence_inconsistent"] / total if total else 0.0,
        },
        "token_stats": {
            "min": min(token_values) if token_values else 0,
            "max": max(token_values) if token_values else 0,
            "mean": sum(token_values) / len(token_values) if token_values else 0,
            "near_cap_count": sum(value >= cap_tokens for value in token_values),
        },
        "by_M": {
            label: dict(counter.most_common())
            for label, counter in sorted(by_m.items(), key=lambda item: item[0])
        },
        "examples": examples,
        "caveat": (
            "This is post-hoc. Old episodes did not log finish_reason, so "
            "max-token labels are inferred from saved thinking/output length."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
