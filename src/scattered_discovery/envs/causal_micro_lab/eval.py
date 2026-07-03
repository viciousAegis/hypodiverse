from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from scattered_discovery.backends import (
    ChatBackend,
    ChatMessage,
    OllamaBackend,
    OpenAICompatibleBackend,
)
from scattered_discovery.envs.causal_micro_lab.parser import parse_record_state
from scattered_discovery.envs.causal_micro_lab.prompt_builder import build_prompt
from scattered_discovery.envs.causal_micro_lab.rewards import group_metrics
from scattered_discovery.envs.causal_micro_lab.state_generator import EvidenceState
from scattered_discovery.envs.causal_micro_lab.verifier import (
    VerificationResult,
    verify_output,
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _flatten_numeric(
    data: dict[str, Any],
    *,
    prefix: str = "",
) -> dict[str, float | int]:
    flat: dict[str, float | int] = {}
    for key, value in data.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, bool):
            flat[name] = float(value)
        elif isinstance(value, int | float):
            flat[name] = value
        elif isinstance(value, dict):
            flat.update(_flatten_numeric(value, prefix=name))
    return flat


def _maybe_start_wandb(args: argparse.Namespace):
    if not args.wandb_project:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging requires wandb. Install it in the eval environment."
        ) from exc
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or args.run_name,
        config={
            "provider": args.provider,
            "model": args.model,
            "input": args.input,
            "rollouts_per_state": args.rollouts_per_state,
            "workers": args.workers,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "max_response_length": args.num_predict,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "prefix_ks": args.prefix_ks,
        },
    )


def load_states(path: str | Path) -> list[EvidenceState]:
    states = []
    for row in _read_jsonl(path):
        if "env_spec_json" in row:
            spec = json.loads(str(row["env_spec_json"]))
            states.append(parse_record_state(spec["task"]["state"]))
        elif "state_json" in row:
            states.append(parse_record_state(json.loads(str(row["state_json"]))))
        elif "visible_experiments" in row:
            states.append(parse_record_state(row))
        else:
            raise ValueError(
                "Input rows must be states, veRL rows with state_json, "
                "or rows with env_spec_json."
            )
    return states


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


def _summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(records),
        "parse_valid": _mean([float(r["verification"]["parse_valid"]) for r in records]),
        "syntax_valid": _mean([float(r["verification"]["syntax_valid"]) for r in records]),
        "evidence_consistent": _mean(
            [float(r["verification"]["evidence_consistent"]) for r in records]
        ),
        "currently_valid_mode": _mean(
            [float(r["verification"]["is_currently_valid_mode"]) for r in records]
        ),
        "nonempty_output": _mean([float(bool(r["output"].strip())) for r in records]),
        "request_error": _mean([float(bool(r.get("request_error"))) for r in records]),
        "model_seconds_total": sum(float(r["model_seconds"]) for r in records),
    }


def _group_by(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        metadata = record["state_metadata"]
        groups.setdefault(str(metadata.get(key, "missing")), []).append(record)
    return {
        label: _summarize_group(group)
        for label, group in sorted(groups.items(), key=lambda item: item[0])
    }


def _verification_from_dict(data: dict[str, Any]) -> VerificationResult:
    family = data.get("mechanism_family")
    return VerificationResult(
        parse_valid=bool(data.get("parse_valid")),
        syntax_valid=bool(data.get("syntax_valid")),
        evidence_consistent=bool(data.get("evidence_consistent")),
        semantic_mode_id=data.get("semantic_mode_id"),
        is_currently_valid_mode=bool(data.get("is_currently_valid_mode")),
        prediction_signature=data.get("prediction_signature"),
        mechanism_family=tuple(family) if family is not None else None,
        error=data.get("error"),
    )


def _mean_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    return {key: _mean([float(item.get(key, 0.0)) for item in items]) for key in keys}


def _grouped_set_records(
    records: list[dict[str, Any]],
    states: list[EvidenceState],
) -> list[dict[str, Any]]:
    states_by_id = {state.state_id: state for state in states}
    by_state: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_state.setdefault(str(record["state_id"]), []).append(record)

    grouped = []
    for state_id, state_records in sorted(by_state.items()):
        state_records = sorted(
            state_records, key=lambda record: int(record["rollout_index"])
        )
        state = states_by_id[state_id]
        results = [
            _verification_from_dict(record["verification"]) for record in state_records
        ]
        metrics = group_metrics(results, state)
        valid_count = int(metrics["num_unique_valid_modes"] + metrics["duplicate_valid_modes"])
        metrics.update(
            {
                "k": float(len(state_records)),
                "valid_mode_rate": valid_count / max(1, len(state_records)),
                "duplicate_rate": metrics["duplicate_valid_modes"]
                / max(1, len(state_records)),
                "extra_duplicate_rate": metrics["extra_duplicate_valid_modes"]
                / max(1, len(state_records)),
            }
        )
        grouped.append(
            {
                "state_id": state_id,
                "metrics": metrics,
                "state_metadata": {
                    "valid_mode_count": state.valid_mode_count,
                    "evidence_size": state.evidence_size,
                    "separation_bucket": state.separation_bucket,
                    "family_bucket": state.family_bucket,
                    "mean_separation": state.mean_separation,
                    "minimum_separation": state.minimum_separation,
                    "maximum_separation": state.maximum_separation,
                },
            }
        )
    return grouped


def _grouped_summary_by(
    grouped: list[dict[str, Any]], key: str
) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, float]]] = {}
    for item in grouped:
        label = str(item["state_metadata"].get(key, "missing"))
        buckets.setdefault(label, []).append(item["metrics"])
    return {
        label: _mean_dicts(items)
        for label, items in sorted(buckets.items(), key=lambda item: item[0])
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _summarize_group(records)
    valid_modes = [
        r["verification"]["semantic_mode_id"]
        for r in records
        if r["verification"]["is_currently_valid_mode"]
        and r["verification"]["semantic_mode_id"]
    ]
    parse_errors = Counter(
        str(r["verification"].get("error"))
        for r in records
        if not r["verification"].get("parse_valid")
    )
    summary.update(
        {
            "unique_valid_modes": len(set(valid_modes)),
            "duplicate_valid_modes": sum(count - 1 for count in Counter(valid_modes).values()),
            "invalid_format_count": sum(parse_errors.values()),
            "invalid_format_errors": dict(parse_errors.most_common(20)),
            "by_M": _group_by(records, "valid_mode_count"),
            "by_separation_bucket": _group_by(records, "separation_bucket"),
            "by_family_bucket": _group_by(records, "family_bucket"),
        }
    )
    return summary


def summarize_grouped_records(
    records: list[dict[str, Any]],
    states: list[EvidenceState],
    max_rollout_index: int | None = None,
) -> dict[str, Any]:
    if max_rollout_index is not None:
        records = [
            record
            for record in records
            if int(record["rollout_index"]) < max_rollout_index
        ]
    grouped = _grouped_set_records(records, states)
    metric_rows = [item["metrics"] for item in grouped]
    summary = _mean_dicts(metric_rows)
    summary.update(
        {
            "states": len(grouped),
            "by_M": _grouped_summary_by(grouped, "valid_mode_count"),
            "by_separation_bucket": _grouped_summary_by(
                grouped, "separation_bucket"
            ),
            "by_family_bucket": _grouped_summary_by(grouped, "family_bucket"),
        }
    )
    return summary


def evaluate_states(
    *,
    states: list[EvidenceState],
    backend: ChatBackend,
    model: str,
    rollouts_per_state: int = 1,
    workers: int = 1,
    output_transcripts: bool = False,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    jobs = []
    for state_index, state in enumerate(states):
        prompt = build_prompt(state)
        messages = [
            ChatMessage(
                role="system",
                content="You are solving a single-shot scientific hypothesis task.",
            ),
            ChatMessage(role="user", content=prompt),
        ]
        for rollout_index in range(rollouts_per_state):
            jobs.append((state_index, state, rollout_index, prompt, messages))

    def run_job(job: tuple[int, EvidenceState, int, str, list[ChatMessage]]) -> dict[str, Any]:
        state_index, state, rollout_index, prompt, messages = job
        started = time.monotonic()
        elapsed = time.monotonic() - started
        sample_id = f"{state.state_id}:sample{rollout_index:04d}"
        try:
            response = backend.chat(messages)
            elapsed = time.monotonic() - started
            output = response.content
            thinking = response.thinking
            verification = verify_output(output, state).as_dict()
            request_error = None
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - started
            output = ""
            thinking = ""
            request_error = f"{type(exc).__name__}: {exc}"
            verification = {
                "parse_valid": False,
                "syntax_valid": False,
                "evidence_consistent": False,
                "semantic_mode_id": None,
                "is_currently_valid_mode": False,
                "prediction_signature": None,
                "mechanism_family": None,
                "error": request_error,
            }
        record = {
            "sample_id": sample_id,
            "state_index": state_index,
            "state_id": state.state_id,
            "rollout_index": rollout_index,
            "model": model,
            "model_seconds": elapsed,
            "output": output,
            "thinking": thinking,
            "request_error": request_error,
            "verification": verification,
            "state_metadata": {
                "valid_mode_count": state.valid_mode_count,
                "evidence_size": state.evidence_size,
                "separation_bucket": state.separation_bucket,
                "family_bucket": state.family_bucket,
                "mean_separation": state.mean_separation,
                "minimum_separation": state.minimum_separation,
                "maximum_separation": state.maximum_separation,
            },
        }
        if output_transcripts:
            record["prompt"] = prompt
        return record

    if workers <= 1:
        records = []
        for job in jobs:
            record = run_job(job)
            records.append(record)
            if on_record is not None:
                on_record(record)
    else:
        records = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_job, job) for job in jobs]
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                if on_record is not None:
                    on_record(record)
    return sorted(
        records,
        key=lambda record: (int(record["state_index"]), int(record["rollout_index"])),
    )


def run_eval(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    states = load_states(args.input)
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, num-shards)")
    if args.num_shards > 1:
        states = [
            state
            for index, state in enumerate(states)
            if index % args.num_shards == args.shard_index
        ]
    if args.max_examples is not None:
        states = states[: args.max_examples]
    backend = build_backend(args)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.provider}_{args.model.replace('/', '_')}"
    output_dir = Path(args.output_dir) / run_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = output_dir.parent / "latest"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(output_dir.name, target_is_directory=True)
    except OSError:
        pass

    wandb_run = _maybe_start_wandb(args)
    episodes_path = output_dir / "episodes.jsonl"
    partial_episodes_path = output_dir / "episodes.partial.jsonl"

    with partial_episodes_path.open("w", encoding="utf-8") as partial_handle:

        def on_record(record: dict[str, Any]) -> None:
            partial_handle.write(json.dumps(record, sort_keys=True) + "\n")
            partial_handle.flush()
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/parse_valid": float(
                            record["verification"]["parse_valid"]
                        ),
                        "eval/syntax_valid": float(
                            record["verification"]["syntax_valid"]
                        ),
                        "eval/evidence_consistent": float(
                            record["verification"]["evidence_consistent"]
                        ),
                        "eval/currently_valid_mode": float(
                            record["verification"]["is_currently_valid_mode"]
                        ),
                        "eval/nonempty_output": float(bool(record["output"].strip())),
                        "eval/request_error": float(bool(record["request_error"])),
                        "eval/model_seconds": float(record["model_seconds"]),
                        "eval/state_index": int(record["state_index"]),
                        "eval/rollout_index": int(record["rollout_index"]),
                        "eval/M": int(
                            record["state_metadata"]["valid_mode_count"]
                        ),
                    }
                )

        records = evaluate_states(
            states=states,
            backend=backend,
            model=args.model,
            rollouts_per_state=args.rollouts_per_state,
            workers=args.workers,
            output_transcripts=args.transcripts,
            on_record=on_record,
        )

    with episodes_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary = summarize_records(records)
    set_summary = summarize_grouped_records(records, states)
    summary.update(
        {
            "input": args.input,
            "model": args.model,
            "provider": args.provider,
            "states": len(states),
            "rollouts_per_state": args.rollouts_per_state,
            "workers": args.workers,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "episodes_path": str(episodes_path),
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "set_summary.json").write_text(
        json.dumps(set_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    prefix_ks = [
        int(item)
        for item in str(args.prefix_ks or "").split(",")
        if item.strip()
    ]
    for prefix_k in prefix_ks:
        if prefix_k < 1 or prefix_k > args.rollouts_per_state:
            raise SystemExit(
                "--prefix-ks values must be in [1, rollouts-per-state]"
            )
        prefix_summary = summarize_grouped_records(
            records,
            states,
            max_rollout_index=prefix_k,
        )
        (output_dir / f"set_summary_k{prefix_k}.json").write_text(
            json.dumps(prefix_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if wandb_run is not None:
            wandb_run.log(
                _flatten_numeric(prefix_summary, prefix=f"set_summary_k{prefix_k}")
            )
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if wandb_run is not None:
        wandb_run.log(_flatten_numeric(summary, prefix="eval_summary"))
        wandb_run.log(_flatten_numeric(set_summary, prefix="set_summary"))
        wandb_run.finish()
    return output_dir, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="states JSONL or veRL JSONL.")
    parser.add_argument("--output-dir", default="results/causal_micro_lab_eval")
    parser.add_argument("--run-name")
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai-compatible"],
        default="ollama",
    )
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--api-key")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--rollouts-per-state", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--prefix-ks",
        default="",
        help="Optional comma-separated k prefixes to summarize from one run.",
    )
    parser.add_argument("--num-predict", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--request-timeout-s", type=float, default=180.0)
    parser.add_argument("--think", default=True)
    parser.add_argument("--transcripts", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    args = parser.parse_args()
    if args.rollouts_per_state < 1:
        raise SystemExit("--rollouts-per-state must be >= 1")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    output_dir, summary = run_eval(args)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
