from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from scattered_discovery.backends import (
    ChatBackend,
    ChatMessage,
    ChatOptions,
    HuggingFaceBackend,
    OllamaBackend,
    OpenAICompatibleBackend,
)
from scattered_discovery.envs.causal_micro_lab.parser import parse_record_state
from scattered_discovery.envs.causal_micro_lab.prompt_builder import (
    build_latent_prompt,
    build_prompt,
)
from scattered_discovery.envs.causal_micro_lab.rewards import group_metrics
from scattered_discovery.envs.causal_micro_lab.state_generator import EvidenceState
from scattered_discovery.envs.causal_micro_lab.verifier import (
    SetVerificationResult,
    VerificationResult,
    verify_output,
    verify_output_set,
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
            "output_mode": args.output_mode,
            "answer_count": args.answer_count,
            "thinking_fallback": args.thinking_fallback,
            "fallback_num_predict": args.fallback_num_predict,
            "fallback_temperature": args.fallback_temperature,
            "latent_count": args.latent_count,
        },
    )


def _log_wandb_set_plots(
    wandb_run: Any,
    summaries_by_k: dict[int, dict[str, Any]],
) -> None:
    import wandb

    ks = sorted(summaries_by_k)
    mode_counts = sorted(
        {
            int(mode_count)
            for summary in summaries_by_k.values()
            for mode_count in summary.get("by_M", {})
        }
    )
    metrics = (
        "exact_coverage",
        "budget_normalized_coverage",
        "valid_mode_rate",
        "duplicity",
        "dominant_mode_mass",
        "effective_mode_count",
        "family_coverage",
        "generated_mode_separation",
    )
    table_columns = ["K", "M", *metrics]
    table_data = []
    for k in ks:
        by_m = summaries_by_k[k].get("by_M", {})
        for mode_count in mode_counts:
            values = by_m.get(str(mode_count), {})
            table_data.append(
                [
                    k,
                    mode_count,
                    *[float(values.get(metric, 0.0)) for metric in metrics],
                ]
            )
    wandb_run.log(
        {
            "eval_tables/set_metrics_by_k_m": wandb.Table(
                columns=table_columns,
                data=table_data,
            )
        }
    )
    for metric in metrics:
        wandb_run.log(
            {
                f"eval_plots/{metric}_by_M": wandb.plot.line_series(
                    xs=ks,
                    ys=[
                        [
                            float(
                                summaries_by_k[k]
                                .get("by_M", {})
                                .get(str(mode_count), {})
                                .get(metric, 0.0)
                            )
                            for k in ks
                        ]
                        for mode_count in mode_counts
                    ],
                    keys=[f"M={mode_count}" for mode_count in mode_counts],
                    title=f"{metric.replace('_', ' ').title()} by M",
                    xname="K",
                )
            }
        )

    separation_buckets = ("low", "medium", "high")
    for metric in (
        "exact_coverage",
        "duplicity",
        "effective_mode_count",
        "generated_mode_separation",
    ):
        wandb_run.log(
            {
                f"eval_plots/{metric}_by_separation": wandb.plot.line_series(
                    xs=ks,
                    ys=[
                        [
                            float(
                                summaries_by_k[k]
                                .get("by_separation_bucket", {})
                                .get(bucket, {})
                                .get(metric, 0.0)
                            )
                            for k in ks
                        ]
                        for bucket in separation_buckets
                    ],
                    keys=list(separation_buckets),
                    title=f"{metric.replace('_', ' ').title()} by Separation",
                    xname="K",
                )
            }
        )


def _coerce_csv_value(value: str) -> str | int | float | None:
    if value == "":
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def _read_csv_table(path: Path) -> tuple[list[str], list[list[Any]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = [
            [_coerce_csv_value(row.get(column, "")) for column in columns]
            for row in reader
        ]
    return columns, rows


def _bootstrap_metric_name(row: dict[str, str]) -> str:
    labels = []
    for key in ("K", "M", "separation_bucket", "family_bucket"):
        value = row.get(key)
        if value:
            safe_value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
            labels.append(f"{key}_{safe_value}")
    label_path = "/".join(labels)
    metric = re.sub(r"[^A-Za-z0-9_.-]+", "_", row["metric"])
    return "/".join(
        part
        for part in ("bootstrap_ci95", row["slice"], label_path, metric)
        if part
    )


def _log_wandb_bootstrap_report(
    wandb_run: Any,
    *,
    report_dir: Path,
    bootstrap_samples: int,
) -> None:
    import wandb

    table_files = (
        "bootstrap_ci95.csv",
        "primary_bootstrap_ci95_by_k_m.csv",
        "primary_bootstrap_ci95_by_k_separation.csv",
        "primary_bootstrap_ci95_by_k_family.csv",
    )
    tables: dict[str, Any] = {}
    for filename in table_files:
        path = report_dir / filename
        columns, rows = _read_csv_table(path)
        tables[f"eval_tables/{path.stem}"] = wandb.Table(
            columns=columns,
            data=rows,
        )
    wandb_run.log(tables)

    primary_metrics = {
        "pass_at_k",
        "modes_recovered_given_success",
        "fraction_modes_recovered_given_success",
    }
    scalar_metrics: dict[str, float | int] = {}
    bootstrap_path = report_dir / "bootstrap_ci95.csv"
    with bootstrap_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] not in primary_metrics:
                continue
            prefix = _bootstrap_metric_name(row)
            for key in (
                "mean",
                "ci95_low",
                "ci95_high",
                "support_states",
                "successful_states",
                "bootstrap_samples",
            ):
                scalar_metrics[f"{prefix}/{key}"] = float(row[key])
    if scalar_metrics:
        wandb_run.log(scalar_metrics)

    artifact = wandb.Artifact(
        name=f"causal-micro-lab-eval-report-{wandb_run.id}",
        type="evaluation-report",
        metadata={
            "bootstrap_samples": bootstrap_samples,
            "contains_per_state_metrics": True,
        },
    )
    artifact.add_dir(str(report_dir))
    wandb_run.log_artifact(artifact)


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


def _parse_think(value: Any) -> bool | str | None:
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return str(value)


def build_backend(args: argparse.Namespace) -> ChatBackend:
    think = _parse_think(args.think)
    if args.provider == "ollama":
        return OllamaBackend(
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            top_p=args.top_p,
            num_predict=args.num_predict,
            request_timeout_s=args.request_timeout_s,
            think=think,
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
            think=think,
        )
    if args.provider == "transformers":
        return HuggingFaceBackend(
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.num_predict,
            think=think,
        )
    raise ValueError(f"Unsupported provider: {args.provider}")


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    initial_tokens = [
        float(r["initial_completion_tokens"])
        for r in records
        if r.get("initial_completion_tokens") is not None
    ]
    total_tokens = [
        float(r.get("initial_completion_tokens") or 0)
        + float(r.get("fallback_completion_tokens") or 0)
        for r in records
        if r.get("initial_completion_tokens") is not None
        or r.get("fallback_completion_tokens") is not None
    ]
    return {
        "episodes": len(records),
        "parse_valid": _mean(
            [float(r["verification"]["parse_valid"]) for r in records]
        ),
        "syntax_valid": _mean(
            [float(r["verification"]["syntax_valid"]) for r in records]
        ),
        "evidence_consistent": _mean(
            [float(r["verification"]["evidence_consistent"]) for r in records]
        ),
        "currently_valid_mode": _mean(
            [float(r["verification"]["is_currently_valid_mode"]) for r in records]
        ),
        "nonempty_output": _mean([float(bool(r["output"].strip())) for r in records]),
        "request_error": _mean([float(bool(r.get("request_error"))) for r in records]),
        "length_cap_hit": _mean(
            [
                float(r.get("initial_finish_reason") in {"length", "max_tokens"})
                for r in records
            ]
        ),
        "fallback_used": _mean([float(bool(r.get("fallback_used"))) for r in records]),
        "fallback_produced_output": _mean(
            [float(bool(r.get("fallback_produced_output"))) for r in records]
        ),
        "fallback_request_error": _mean(
            [float(bool(r.get("fallback_request_error"))) for r in records]
        ),
        "initial_completion_tokens_mean": _mean(initial_tokens),
        "total_completion_tokens_mean": _mean(total_tokens),
        "thinking_chars_mean": _mean(
            [float(len(str(r.get("thinking") or ""))) for r in records]
        ),
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


def _set_verification_to_eval_dict(
    result: SetVerificationResult,
    state: EvidenceState,
) -> dict[str, Any]:
    data = result.as_dict()
    data.update(
        {
            "output_mode": "multi_answer_rlvr",
            "parse_valid": result.parse_valid_count == result.expected_count,
            "syntax_valid": result.syntax_valid_count == result.expected_count,
            "evidence_consistent": result.evidence_consistent_count > 0,
            "semantic_mode_id": result.unique_valid_mode_ids[0]
            if result.unique_valid_mode_ids
            else None,
            "is_currently_valid_mode": result.any_valid,
            "prediction_signature": None,
            "mechanism_family": None,
            "coverage_per_available": result.coverage_per_available(state),
            "any_valid": result.any_valid,
            "error": None if result.format_valid else "missing_or_empty_answer_tags",
        }
    )
    return data


def _verification_results_for_record(
    record: dict[str, Any],
) -> list[VerificationResult]:
    verification = record["verification"]
    candidates = verification.get("candidates")
    if candidates:
        return [
            _verification_from_dict(candidate["verification"])
            for candidate in candidates
        ]
    return [_verification_from_dict(verification)]


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
            result
            for record in state_records
            for result in _verification_results_for_record(record)
        ]
        metrics = group_metrics(results, state)
        valid_count = int(
            metrics["num_unique_valid_modes"] + metrics["duplicate_valid_modes"]
        )
        metrics.update(
            {
                "k": float(len(results)),
                "valid_mode_rate": valid_count / max(1, len(results)),
                "duplicate_rate": metrics["duplicate_valid_modes"]
                / max(1, len(results)),
                "extra_duplicate_rate": metrics["extra_duplicate_valid_modes"]
                / max(1, len(results)),
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
    result = {}
    for label, items in sorted(buckets.items(), key=lambda item: item[0]):
        result[label] = _mean_dicts(items)
        result[label]["states"] = float(len(items))
    return result


def _grouped_summary_by_pair(
    grouped: list[dict[str, Any]],
    outer_key: str,
    inner_key: str,
) -> dict[str, dict[str, dict[str, float]]]:
    outer_labels = sorted(
        {str(item["state_metadata"].get(outer_key, "missing")) for item in grouped}
    )
    return {
        outer_label: _grouped_summary_by(
            [
                item
                for item in grouped
                if str(item["state_metadata"].get(outer_key, "missing")) == outer_label
            ],
            inner_key,
        )
        for outer_label in outer_labels
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _summarize_group(records)
    flat_results = [
        result
        for record in records
        for result in _verification_results_for_record(record)
    ]
    valid_modes = [
        result.semantic_mode_id
        for result in flat_results
        if result.is_currently_valid_mode and result.semantic_mode_id
    ]
    parse_errors = Counter(
        str(r["verification"].get("error"))
        for r in records
        if not r["verification"].get("parse_valid")
    )
    summary.update(
        {
            "unique_valid_modes": len(set(valid_modes)),
            "duplicate_valid_modes": sum(
                count - 1 for count in Counter(valid_modes).values()
            ),
            "invalid_format_count": sum(parse_errors.values()),
            "invalid_format_errors": dict(parse_errors.most_common(20)),
            "candidate_outputs": len(flat_results),
            "candidate_parse_valid": _mean(
                [float(r.parse_valid) for r in flat_results]
            ),
            "candidate_syntax_valid": _mean(
                [float(r.syntax_valid) for r in flat_results]
            ),
            "candidate_evidence_consistent": _mean(
                [float(r.evidence_consistent) for r in flat_results]
            ),
            "candidate_currently_valid_mode": _mean(
                [float(r.is_currently_valid_mode) for r in flat_results]
            ),
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
            "by_separation_bucket": _grouped_summary_by(grouped, "separation_bucket"),
            "by_family_bucket": _grouped_summary_by(grouped, "family_bucket"),
            "by_M_and_separation_bucket": _grouped_summary_by_pair(
                grouped,
                "valid_mode_count",
                "separation_bucket",
            ),
            "by_M_and_family_bucket": _grouped_summary_by_pair(
                grouped,
                "valid_mode_count",
                "family_bucket",
            ),
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
    on_progress: Callable[[int, int], None] | None = None,
    progress_interval_s: float = 60.0,
    output_mode: str = "single",
    answer_count: int = 1,
    thinking_fallback: bool = False,
    fallback_num_predict: int = 256,
    fallback_temperature: float = 0.0,
    latent_count: int = 0,
) -> list[dict[str, Any]]:
    jobs = []
    for state_index, state in enumerate(states):
        for rollout_index in range(rollouts_per_state):
            prompt = build_prompt(
                state,
                output_mode=output_mode,
                answer_count=answer_count,
            )
            latent_id = rollout_index % latent_count + 1 if latent_count else 0
            if latent_id:
                prompt = build_latent_prompt(prompt, latent_id)
            messages = [
                ChatMessage(
                    role="system",
                    content="You are solving a single-shot scientific hypothesis task.",
                ),
                ChatMessage(role="user", content=prompt),
            ]
            jobs.append(
                (
                    state_index,
                    state,
                    rollout_index,
                    prompt,
                    messages,
                    latent_id,
                )
            )

    def run_job(
        job: tuple[
            int,
            EvidenceState,
            int,
            str,
            list[ChatMessage],
            int,
        ],
    ) -> dict[str, Any]:
        (
            state_index,
            state,
            rollout_index,
            prompt,
            messages,
            latent_id,
        ) = job
        started = time.monotonic()
        elapsed = time.monotonic() - started
        sample_id = f"{state.state_id}:sample{rollout_index:04d}"
        initial_finish_reason = None
        initial_completion_tokens = None
        fallback_used = False
        fallback_finish_reason = None
        fallback_completion_tokens = None
        fallback_seconds = 0.0
        fallback_request_error = None
        fallback_produced_output = False
        try:
            response = backend.chat(messages)
            initial_elapsed = time.monotonic() - started
            output = response.content
            thinking = response.thinking
            initial_finish_reason = response.finish_reason
            initial_completion_tokens = response.completion_tokens
            if thinking_fallback and response.finish_reason in {"length", "max_tokens"}:
                fallback_used = True
                if output_mode == "multi_answer_rlvr":
                    final_instruction = (
                        f"Using the reasoning above, return exactly {answer_count} "
                        "answers in the required answer-tag format. Do not think further "
                        "and output nothing except the answers."
                    )
                else:
                    final_instruction = (
                        "Using the reasoning above, return the final hypothesis now. "
                        "Do not think further. Output exactly these three rule lines and "
                        "nothing else:\nZ1: ...\nZ2: ...\nY: ..."
                    )
                fallback_messages = [
                    *messages,
                    ChatMessage(
                        role="assistant",
                        content=f"<think>\n{thinking}\n</think>",
                    ),
                    ChatMessage(role="user", content=final_instruction),
                ]
                fallback_started = time.monotonic()
                try:
                    finalized = backend.chat(
                        fallback_messages,
                        options=ChatOptions(
                            think=False,
                            num_predict=fallback_num_predict,
                            temperature=fallback_temperature,
                            top_p=1.0,
                        ),
                    )
                    fallback_seconds = time.monotonic() - fallback_started
                    fallback_finish_reason = finalized.finish_reason
                    fallback_completion_tokens = finalized.completion_tokens
                    if finalized.content.strip():
                        output = finalized.content
                        fallback_produced_output = True
                except Exception as exc:  # noqa: BLE001
                    fallback_seconds = time.monotonic() - fallback_started
                    fallback_request_error = f"{type(exc).__name__}: {exc}"
                elapsed = initial_elapsed + fallback_seconds
            else:
                elapsed = initial_elapsed
            canonical_output = output
            if output_mode == "multi_answer_rlvr":
                verification = _set_verification_to_eval_dict(
                    verify_output_set(
                        output,
                        state,
                        expected_count=answer_count,
                    ),
                    state,
                )
            else:
                verification = verify_output(canonical_output, state).as_dict()
            request_error = None
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - started
            output = ""
            canonical_output = ""
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
            if output_mode == "multi_answer_rlvr":
                verification.update(
                    {
                        "output_mode": "multi_answer_rlvr",
                        "expected_count": answer_count,
                        "candidate_count": 0,
                        "format_valid": False,
                        "parse_valid_count": 0,
                        "syntax_valid_count": 0,
                        "evidence_consistent_count": 0,
                        "valid_count": 0,
                        "valid_mode_ids": [],
                        "unique_valid_mode_ids": [],
                        "duplicate_valid_modes": 0,
                        "coverage_per_k": 0.0,
                        "coverage_per_available": 0.0,
                        "any_valid": False,
                        "candidates": [],
                    }
                )
        record = {
            "sample_id": sample_id,
            "state_index": state_index,
            "state_id": state.state_id,
            "rollout_index": rollout_index,
            "model": model,
            "model_seconds": elapsed,
            "output": output,
            "canonical_output": canonical_output,
            "latent_count": latent_count,
            "latent_id": latent_id,
            "thinking": thinking,
            "request_error": request_error,
            "initial_finish_reason": initial_finish_reason,
            "initial_completion_tokens": initial_completion_tokens,
            "fallback_used": fallback_used,
            "fallback_finish_reason": fallback_finish_reason,
            "fallback_completion_tokens": fallback_completion_tokens,
            "fallback_seconds": fallback_seconds,
            "fallback_request_error": fallback_request_error,
            "fallback_produced_output": fallback_produced_output,
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
            if on_progress is not None:
                on_progress(len(records), len(jobs))
    else:
        records = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = {executor.submit(run_job, job) for job in jobs}
            last_progress = time.monotonic()
            while pending:
                done, pending = wait(
                    pending,
                    timeout=progress_interval_s,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    record = future.result()
                    records.append(record)
                    if on_record is not None:
                        on_record(record)
                now = time.monotonic()
                if on_progress is not None and (
                    done or now - last_progress >= progress_interval_s
                ):
                    on_progress(len(records), len(jobs))
                    last_progress = now
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
    total_jobs = len(states) * args.rollouts_per_state
    if wandb_run is not None:
        wandb_run.log(
            {
                "eval/jobs_total": total_jobs,
                "eval/jobs_completed": 0,
                "eval/jobs_pending": total_jobs,
                "eval/progress_fraction": 0.0,
            }
        )
    print(
        "Starting causal micro-lab eval: "
        f"states={len(states)} rollouts_per_state={args.rollouts_per_state} "
        f"jobs={total_jobs} workers={args.workers}",
        flush=True,
    )

    with partial_episodes_path.open("w", encoding="utf-8") as partial_handle:

        def on_record(record: dict[str, Any]) -> None:
            partial_handle.write(json.dumps(record, sort_keys=True) + "\n")
            partial_handle.flush()
            if wandb_run is not None:
                metrics = {
                    "eval/parse_valid": float(record["verification"]["parse_valid"]),
                    "eval/syntax_valid": float(record["verification"]["syntax_valid"]),
                    "eval/evidence_consistent": float(
                        record["verification"]["evidence_consistent"]
                    ),
                    "eval/currently_valid_mode": float(
                        record["verification"]["is_currently_valid_mode"]
                    ),
                    "eval/nonempty_output": float(bool(record["output"].strip())),
                    "eval/request_error": float(bool(record["request_error"])),
                    "eval/length_cap_hit": float(
                        record.get("initial_finish_reason") in {"length", "max_tokens"}
                    ),
                    "eval/fallback_used": float(bool(record.get("fallback_used"))),
                    "eval/fallback_produced_output": float(
                        bool(record.get("fallback_produced_output"))
                    ),
                    "eval/fallback_request_error": float(
                        bool(record.get("fallback_request_error"))
                    ),
                    "eval/initial_completion_tokens": float(
                        record.get("initial_completion_tokens") or 0
                    ),
                    "eval/fallback_completion_tokens": float(
                        record.get("fallback_completion_tokens") or 0
                    ),
                    "eval/model_seconds": float(record["model_seconds"]),
                    "eval/state_index": int(record["state_index"]),
                    "eval/rollout_index": int(record["rollout_index"]),
                    "eval/M": int(record["state_metadata"]["valid_mode_count"]),
                }
                for key in (
                    "format_valid",
                    "candidate_count",
                    "parse_valid_count",
                    "syntax_valid_count",
                    "evidence_consistent_count",
                    "valid_count",
                    "duplicate_valid_modes",
                    "coverage_per_k",
                    "coverage_per_available",
                    "any_valid",
                ):
                    value = record["verification"].get(key)
                    if isinstance(value, bool):
                        metrics[f"eval/{key}"] = float(value)
                    elif isinstance(value, int | float):
                        metrics[f"eval/{key}"] = value
                wandb_run.log(metrics)

        def on_progress(completed: int, total: int) -> None:
            pending = total - completed
            print(
                f"causal micro-lab eval progress: {completed}/{total} "
                f"completed ({pending} pending)",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/jobs_total": total,
                        "eval/jobs_completed": completed,
                        "eval/jobs_pending": pending,
                        "eval/progress_fraction": completed / total if total else 0.0,
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
            on_progress=on_progress,
            progress_interval_s=30.0,
            output_mode=args.output_mode,
            answer_count=args.answer_count,
            thinking_fallback=args.thinking_fallback,
            fallback_num_predict=args.fallback_num_predict,
            fallback_temperature=args.fallback_temperature,
            latent_count=args.latent_count,
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
            "latent_count": args.latent_count,
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
        int(item) for item in str(args.prefix_ks or "").split(",") if item.strip()
    ]
    prefix_summaries: dict[int, dict[str, Any]] = {}
    for prefix_k in prefix_ks:
        if prefix_k < 1 or prefix_k > args.rollouts_per_state:
            raise SystemExit("--prefix-ks values must be in [1, rollouts-per-state]")
        prefix_summary = summarize_grouped_records(
            records,
            states,
            max_rollout_index=prefix_k,
        )
        prefix_summaries[prefix_k] = prefix_summary
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
        if prefix_summaries:
            _log_wandb_set_plots(wandb_run, prefix_summaries)
    if args.build_report:
        from scattered_discovery.envs.causal_micro_lab.report import build_report

        report_dir = output_dir / "report"
        report_ks = tuple(prefix_ks or [args.rollouts_per_state])
        build_report(
            episodes_path=episodes_path,
            states_path=Path(args.input),
            output_dir=report_dir,
            ks=report_ks,
            bootstrap_samples=args.bootstrap_samples,
        )
        if wandb_run is not None:
            _log_wandb_bootstrap_report(
                wandb_run,
                report_dir=report_dir,
                bootstrap_samples=args.bootstrap_samples,
            )
    if wandb_run is not None:
        wandb_run.finish()
    return output_dir, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="states JSONL or veRL JSONL.")
    parser.add_argument("--output-dir", default="results/causal_micro_lab_eval")
    parser.add_argument("--run-name")
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai-compatible", "transformers"],
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
    parser.add_argument(
        "--output-mode",
        choices=["single", "multi_answer_rlvr"],
        default="single",
    )
    parser.add_argument("--answer-count", type=int, default=1)
    parser.add_argument("--transcripts", action="store_true")
    parser.add_argument(
        "--thinking-fallback",
        action="store_true",
        help=(
            "When the thinking request ends with finish_reason=length, append its "
            "reasoning to the conversation and make a short non-thinking finalizer call."
        ),
    )
    parser.add_argument("--fallback-num-predict", type=int, default=256)
    parser.add_argument("--fallback-temperature", type=float, default=0.0)
    parser.add_argument(
        "--latent-count",
        type=int,
        default=0,
        help=(
            "Cycle rollouts through Strategy 1..N prompt labels. "
            "Zero disables latent conditioning."
        ),
    )
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--build-report",
        action="store_true",
        help=(
            "Build per-state CSVs and bootstrap intervals before closing W&B."
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    if args.rollouts_per_state < 1:
        raise SystemExit("--rollouts-per-state must be >= 1")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.fallback_num_predict < 1:
        raise SystemExit("--fallback-num-predict must be >= 1")
    if args.answer_count < 1:
        raise SystemExit("--answer-count must be >= 1")
    if args.latent_count < 0:
        raise SystemExit("--latent-count must be >= 0")
    if args.bootstrap_samples < 1:
        raise SystemExit("--bootstrap-samples must be >= 1")
    output_dir, summary = run_eval(args)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
