from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scattered_discovery.backends import (
    ChatBackend,
    OllamaBackend,
    OpenAICompatibleBackend,
)
from scattered_discovery.rollout.local_loop import run_local_episode


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Parquet input requires pandas and pyarrow. Install with "
            "`uv sync --extra verl`."
        ) from exc
    return pd.read_parquet(path).to_dict(orient="records")


def load_env_specs(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".parquet":
        rows = _load_parquet(source)
    elif source.suffix == ".jsonl":
        rows = _load_jsonl(source)
    else:
        raw = json.loads(source.read_text(encoding="utf-8"))
        rows = raw if isinstance(raw, list) else [raw]

    specs = []
    for row in rows:
        if "env_spec_json" in row:
            value = row["env_spec_json"]
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            specs.append(json.loads(str(value)))
        elif "env_type" in row and "task" in row:
            specs.append(dict(row))
        else:
            raise ValueError(
                "Each eval row must contain env_spec_json or an EnvSpec-shaped mapping."
            )
    return specs


def build_backend(args: argparse.Namespace) -> ChatBackend:
    if args.provider == "ollama":
        return OllamaBackend(
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            top_p=args.top_p,
            num_predict=args.num_predict,
            request_timeout_s=args.request_timeout_s,
            think=args.think,
        )
    if args.provider == "openai-compatible":
        return OpenAICompatibleBackend(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.num_predict,
            request_timeout_s=args.request_timeout_s,
        )
    raise ValueError(f"Unsupported provider: {args.provider}")


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(record["score"]["reward"]) for record in records]
    valid_unique_counts = [
        float(record["score"].get("valid_unique_count", 0)) for record in records
    ]
    validities = [float(record["score"].get("validity", 0.0)) for record in records]
    uniquenesses = [float(record["score"].get("uniqueness", 0.0)) for record in records]
    false_counts = [float(record["score"].get("false_count", 0)) for record in records]
    non_final_counts = [
        float(record["score"].get("non_final_count", 0)) for record in records
    ]
    unsupported = [
        float(record["score"].get("unsupported_count", 0)) for record in records
    ]
    parse_failures = [
        float(record["score"].get("parse_failures", 0)) for record in records
    ]
    invalid_actions = [
        float(record["score"].get("invalid_actions", 0)) for record in records
    ]
    recoveries = [
        float(record["score"].get("metrics", {}).get("recovery", 0.0))
        for record in records
    ]
    return {
        "episodes": len(records),
        "reward_mean": _mean(rewards),
        "reward_stdev": _stdev(rewards),
        "valid_unique_count_mean": _mean(valid_unique_counts),
        "validity_mean": _mean(validities),
        "uniqueness_mean": _mean(uniquenesses),
        "false_count_mean": _mean(false_counts),
        "non_final_count_mean": _mean(non_final_counts),
        "unsupported_count_mean": _mean(unsupported),
        "parse_failures_mean": _mean(parse_failures),
        "invalid_actions_mean": _mean(invalid_actions),
        "recovery_mean": _mean(recoveries),
        "model_seconds_total": sum(
            float(record.get("model_seconds", 0.0)) for record in records
        ),
    }


def _maybe_start_wandb(args: argparse.Namespace):
    if not args.wandb_project:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging requires wandb. Install with `uv sync --extra verl`."
        ) from exc
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "provider": args.provider,
            "model": args.model,
            "input": args.input,
            "rollouts_per_spec": args.rollouts_per_spec,
            "max_steps": args.max_steps,
            "num_predict": args.num_predict,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    )


def run_eval(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    specs = load_env_specs(args.input)
    if args.max_examples is not None:
        specs = specs[: args.max_examples]

    output_dir = Path(args.output_dir)
    if args.run_name:
        output_dir = output_dir / args.run_name
    else:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_dir = (
            output_dir / f"{args.provider}_{args.model.replace('/', '_')}_{timestamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = build_backend(args)
    wandb_run = _maybe_start_wandb(args)
    records = []
    started = time.monotonic()

    episodes_path = output_dir / "episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as handle:
        for spec_index, spec in enumerate(specs):
            for rollout_index in range(args.rollouts_per_spec):
                record = run_local_episode(
                    spec=spec,
                    backend=backend,
                    max_steps=args.max_steps,
                    output_transcript=args.transcripts,
                )
                record["spec_index"] = spec_index
                record["rollout_index"] = rollout_index
                record["env_type"] = spec["env_type"]
                record["model"] = args.model
                records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "eval/reward": record["score"]["reward"],
                            "eval/valid_unique_count": record["score"].get(
                                "valid_unique_count", 0
                            ),
                            "eval/validity": record["score"].get("validity", 0.0),
                            "eval/uniqueness": record["score"].get("uniqueness", 0.0),
                            "eval/non_final_count": record["score"].get(
                                "non_final_count", 0
                            ),
                            "eval/recovery": record["score"]
                            .get("metrics", {})
                            .get("recovery", 0.0),
                            "eval/spec_index": spec_index,
                        }
                    )

    summary = summarize_records(records)
    summary.update(
        {
            "input": args.input,
            "provider": args.provider,
            "model": args.model,
            "specs": len(specs),
            "rollouts_per_spec": args.rollouts_per_spec,
            "wall_seconds": time.monotonic() - started,
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8"
    )
    if wandb_run is not None:
        wandb_run.log({f"eval_summary/{key}": value for key, value in summary.items()})
        wandb_run.finish()
    return output_dir, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="EnvSpec JSON/JSONL/Parquet.")
    parser.add_argument("--output-dir", default="results/envspec_eval")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai-compatible"],
        default="ollama",
    )
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--rollouts-per-spec", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--request-timeout-s", type=float, default=180.0)
    parser.add_argument("--think", default=True)
    parser.add_argument("--transcripts", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    output_dir, summary = run_eval(args)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
