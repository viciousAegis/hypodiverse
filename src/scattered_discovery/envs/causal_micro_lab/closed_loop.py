from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scattered_discovery.envs.causal_micro_lab.eval import (
    build_backend,
    evaluate_states,
    load_states,
)
from scattered_discovery.envs.causal_micro_lab.planner import (
    available_experiment_ids,
    experiment_entropy,
    oracle_disagreement_experiment,
    prediction_distributions,
    run_oracle_closed_loop,
    select_disagreement_experiment,
    select_seeded_random_experiment,
)
from scattered_discovery.envs.causal_micro_lab.rewards import group_metrics
from scattered_discovery.envs.causal_micro_lab.signatures import (
    ModeTable,
    build_mode_table,
)
from scattered_discovery.envs.causal_micro_lab.state_generator import (
    EvidenceState,
    make_state,
)
from scattered_discovery.envs.causal_micro_lab.verifier import VerificationResult


@dataclass
class TrajectoryRuntime:
    index: int
    initial: EvidenceState
    current: EvidenceState
    records: list[dict[str, Any]] = field(default_factory=list)

    def is_active(self, max_steps: int, *, mode_table: ModeTable) -> bool:
        return (
            len(self.records) < max_steps
            and self.current.valid_mode_count > 1
            and bool(available_experiment_ids(self.current, mode_table=mode_table))
        )


def _stable_seed(seed: int, state: EvidenceState, step: int) -> int:
    payload = (
        f"{seed}:{state.hidden_mode_id}:{step}:"
        + ",".join(str(item) for item in state.observed_experiment_ids())
    )
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big")


def _verification_from_record(record: dict[str, Any]) -> VerificationResult:
    value = record["verification"]
    family = value.get("mechanism_family")
    return VerificationResult(
        parse_valid=bool(value.get("parse_valid")),
        syntax_valid=bool(value.get("syntax_valid")),
        evidence_consistent=bool(value.get("evidence_consistent")),
        semantic_mode_id=value.get("semantic_mode_id"),
        is_currently_valid_mode=bool(value.get("is_currently_valid_mode")),
        prediction_signature=value.get("prediction_signature"),
        mechanism_family=tuple(family) if family is not None else None,
        error=value.get("error"),
    )


def _completion_trace(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "rollout_index": record["rollout_index"],
        "latent_id": record.get("latent_id", 0),
        "output": record["output"],
        "thinking": record["thinking"],
        "verification": record["verification"],
        "request_error": record.get("request_error"),
        "model_seconds": record["model_seconds"],
        "finish_reason": record.get("initial_finish_reason"),
        "completion_tokens": record.get("initial_completion_tokens"),
        "fallback_used": record.get("fallback_used", False),
        "fallback_finish_reason": record.get("fallback_finish_reason"),
        "fallback_completion_tokens": record.get("fallback_completion_tokens"),
    }


def _outcome_json(outcome: tuple[int, int, int]) -> dict[str, int]:
    return {"Z1": outcome[0], "Z2": outcome[1], "Y": outcome[2]}


def _bank_metrics(
    records: list[dict[str, Any]],
    state: EvidenceState,
) -> tuple[dict[str, float], list[str], list[str]]:
    results = [_verification_from_record(record) for record in records]
    metrics = group_metrics(results, state)
    sample_count = max(1, len(results))
    metrics.update(
        {
            "parse_valid_rate": metrics["num_parse_valid"] / sample_count,
            "syntax_valid_rate": metrics["num_syntax_valid"] / sample_count,
            "evidence_consistent_rate": metrics["num_evidence_consistent"]
            / sample_count,
            "duplicate_rate": metrics["duplicate_valid_modes"] / sample_count,
        }
    )
    valid_mode_ids = [
        result.semantic_mode_id
        for result in results
        if result.is_currently_valid_mode and result.semantic_mode_id is not None
    ]
    return metrics, valid_mode_ids, list(dict.fromkeys(valid_mode_ids))


def _visible_evidence(state: EvidenceState, table: ModeTable) -> list[dict[str, object]]:
    return list(
        state.to_record(mode_table=table, include_private=False)["visible_experiments"]
    )


def _step_record(
    runtime: TrajectoryRuntime,
    generation_records: list[dict[str, Any]],
    *,
    run_name: str,
    model: str,
    seed: int,
    deduplicate_planner_modes: bool,
    mode_table: ModeTable,
) -> tuple[dict[str, Any], EvidenceState]:
    state = runtime.current
    step = len(runtime.records) + 1
    metrics, valid_mode_ids, unique_mode_ids = _bank_metrics(
        generation_records,
        state,
    )
    metrics["hidden_mode_in_bank"] = float(
        state.hidden_mode_id in set(valid_mode_ids)
    )
    planner_mode_ids = unique_mode_ids if deduplicate_planner_modes else valid_mode_ids
    distributions = prediction_distributions(
        planner_mode_ids,
        state,
        mode_table=mode_table,
    )
    if planner_mode_ids:
        selected_experiment = select_disagreement_experiment(
            planner_mode_ids,
            state,
            mode_table=mode_table,
        )
        selection_reason = "generated_entropy"
    else:
        selected_experiment = select_seeded_random_experiment(
            state,
            seed=_stable_seed(seed, runtime.initial, step),
            mode_table=mode_table,
        )
        selection_reason = "seeded_random_fallback"
    if selected_experiment is None:
        raise RuntimeError("active trajectory has no selectable experiment")

    deduplicated_experiment = select_disagreement_experiment(
        unique_mode_ids,
        state,
        mode_table=mode_table,
    )
    oracle_experiment = oracle_disagreement_experiment(state, mode_table=mode_table)
    oracle_entropy = (
        experiment_entropy(
            state.valid_mode_ids,
            oracle_experiment,
            mode_table=mode_table,
        )
        if oracle_experiment is not None
        else 0.0
    )
    selected_true_entropy = experiment_entropy(
        state.valid_mode_ids,
        selected_experiment,
        mode_table=mode_table,
    )
    empirical_entropy = float(
        distributions.get(selected_experiment, {}).get("entropy", 0.0)
    )

    hidden = mode_table.modes_by_id[state.hidden_mode_id]
    observed_outcome = hidden.signature[selected_experiment]
    before_size = state.valid_mode_count
    next_evidence_ids = tuple(
        sorted((*state.observed_experiment_ids(), selected_experiment))
    )
    updated = make_state(
        hidden_mode=hidden,
        evidence_ids=next_evidence_ids,
        mode_table=mode_table,
        compute_separation=True,
    )
    after_size = updated.valid_mode_count
    information_gain_bits = math.log2(before_size) - math.log2(after_size)

    record: dict[str, Any] = {
        "schema_version": 1,
        "run_name": run_name,
        "model": model,
        "trajectory_index": runtime.index,
        "initial_state_id": runtime.initial.state_id,
        "hidden_mode_id": runtime.initial.hidden_mode_id,
        "experiment_step": step,
        "current_state_id": state.state_id,
        "visible_evidence_before": _visible_evidence(state, mode_table),
        "prompt": generation_records[0].get("prompt") if generation_records else None,
        "completions": [_completion_trace(record) for record in generation_records],
        "bank_metrics": metrics,
        "valid_mode_ids": valid_mode_ids,
        "unique_valid_mode_ids": unique_mode_ids,
        "hidden_mode_in_bank": bool(metrics["hidden_mode_in_bank"]),
        "planner_mode_ids": planner_mode_ids,
        "planner_deduplicated": deduplicate_planner_modes,
        "predicted_outcome_distributions": {
            str(experiment_id): value
            for experiment_id, value in distributions.items()
        },
        "selected_experiment_id": selected_experiment,
        "selection_reason": selection_reason,
        "selected_empirical_entropy": empirical_entropy,
        "deduplicated_selected_experiment_id": deduplicated_experiment,
        "observed_outcome": _outcome_json(observed_outcome),
        "version_space_size_before": before_size,
        "version_space_size_after": after_size,
        "information_gain_bits": information_gain_bits,
        "information_gain_modes": before_size - after_size,
        "identified": after_size == 1,
        "oracle_selected_experiment_id": oracle_experiment,
        "oracle_entropy": oracle_entropy,
        "selected_experiment_true_entropy": selected_true_entropy,
        "entropy_regret": max(0.0, oracle_entropy - selected_true_entropy),
        "request_seconds_total": sum(
            float(item["model_seconds"]) for item in generation_records
        ),
        "completion_tokens_total": sum(
            int(item.get("initial_completion_tokens") or 0)
            + int(item.get("fallback_completion_tokens") or 0)
            for item in generation_records
        ),
        "request_error_count": sum(
            bool(item.get("request_error")) for item in generation_records
        ),
        "initial_metadata": {
            "valid_mode_count": runtime.initial.valid_mode_count,
            "separation_bucket": runtime.initial.separation_bucket,
            "family_bucket": runtime.initial.family_bucket,
            "mean_separation": runtime.initial.mean_separation,
        },
        "private_valid_mode_ids_before": list(state.valid_mode_ids),
        "private_valid_mode_ids_after": list(updated.valid_mode_ids),
    }
    return record, updated


def select_initial_states(
    states: list[EvidenceState],
    *,
    initial_mode_counts: tuple[int, ...],
    trajectories_per_count: int,
    seed: int,
) -> list[EvidenceState]:
    if not initial_mode_counts:
        raise ValueError("at least one initial mode count is required")
    if trajectories_per_count < 1:
        raise ValueError("trajectories_per_count must be positive")
    unique = {state.state_id: state for state in states}.values()
    rng = random.Random(seed)
    result: list[EvidenceState] = []
    used_hidden_modes: set[str] = set()
    for mode_count in initial_mode_counts:
        by_separation: dict[str, list[EvidenceState]] = {}
        for state in unique:
            if state.valid_mode_count != mode_count:
                continue
            by_separation.setdefault(state.separation_bucket, []).append(state)
        for candidates in by_separation.values():
            candidates.sort(key=lambda state: state.state_id)
            rng.shuffle(candidates)
        selected_for_count: list[EvidenceState] = []
        labels = sorted(by_separation)
        while labels and len(selected_for_count) < trajectories_per_count:
            remaining = []
            for label in labels:
                candidates = by_separation[label]
                while (
                    candidates
                    and candidates[-1].hidden_mode_id in used_hidden_modes
                ):
                    candidates.pop()
                if candidates:
                    state = candidates.pop()
                    selected_for_count.append(state)
                    used_hidden_modes.add(state.hidden_mode_id)
                if candidates:
                    remaining.append(label)
                if len(selected_for_count) == trajectories_per_count:
                    break
            labels = remaining
        if len(selected_for_count) != trajectories_per_count:
            raise ValueError(
                f"M={mode_count}: found {len(selected_for_count)} distinct held-out "
                f"worlds, need {trajectories_per_count}"
            )
        result.extend(selected_for_count)
    return sorted(result, key=lambda state: (state.valid_mode_count, state.state_id))


def _read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _restore_runtimes(
    states: list[EvidenceState],
    trace_records: list[dict[str, Any]],
    *,
    mode_table: ModeTable,
) -> list[TrajectoryRuntime]:
    by_state: dict[str, list[dict[str, Any]]] = {}
    for record in trace_records:
        by_state.setdefault(str(record["initial_state_id"]), []).append(record)
    runtimes = []
    for index, initial in enumerate(states):
        records = sorted(
            by_state.get(initial.state_id, []),
            key=lambda record: int(record["experiment_step"]),
        )
        current = initial
        seen: set[int] = set(current.observed_experiment_ids())
        for expected_step, record in enumerate(records, start=1):
            if int(record["experiment_step"]) != expected_step:
                raise ValueError(
                    f"non-contiguous trace for state {initial.state_id}"
                )
            if str(record["hidden_mode_id"]) != initial.hidden_mode_id:
                raise ValueError(f"hidden mode mismatch for state {initial.state_id}")
            experiment_id = int(record["selected_experiment_id"])
            if experiment_id in seen:
                raise ValueError(
                    f"trace repeats experiment {experiment_id} for {initial.state_id}"
                )
            seen.add(experiment_id)
            current = make_state(
                hidden_mode=mode_table.modes_by_id[initial.hidden_mode_id],
                evidence_ids=tuple(sorted(seen)),
                mode_table=mode_table,
                compute_separation=True,
            )
            if current.valid_mode_count != int(record["version_space_size_after"]):
                raise ValueError(
                    f"version-space mismatch while replaying {initial.state_id}"
                )
        runtimes.append(
            TrajectoryRuntime(
                index=index,
                initial=initial,
                current=current,
                records=records,
            )
        )
    unknown = set(by_state) - {state.state_id for state in states}
    if unknown:
        raise ValueError(f"trace contains {len(unknown)} states absent from input")
    return runtimes


def _carry_curve(values: list[float], max_steps: int) -> list[float]:
    if not values:
        return [0.0] * (max_steps + 1)
    return values + [values[-1]] * (max_steps + 1 - len(values))


def _bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = statistics.fmean(values)
    if len(values) == 1 or samples <= 0:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    rng = random.Random(seed)
    boot = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(samples)
    )
    return {
        "mean": mean,
        "ci95_low": boot[int(0.025 * (samples - 1))],
        "ci95_high": boot[int(0.975 * (samples - 1))],
    }


def _reference_oracle_curve(
    state: EvidenceState,
    *,
    max_steps: int,
    mode_table: ModeTable,
) -> list[float]:
    trace = run_oracle_closed_loop(
        state,
        max_steps=max_steps,
        mode_table=mode_table,
    )
    sizes = [float(state.valid_mode_count)]
    sizes.extend(float(step["remaining_version_space_size"]) for step in trace.steps)
    return _carry_curve(sizes, max_steps)


def _reference_seed(
    seed: int,
    state: EvidenceState,
    step: int,
    policy: str,
    replicate: int,
) -> int:
    payload = (
        f"{seed}:{policy}:{replicate}:{state.hidden_mode_id}:{step}:"
        + ",".join(str(item) for item in state.observed_experiment_ids())
    )
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big")


def _reference_curve(
    state: EvidenceState,
    *,
    policy: str,
    k: int,
    max_steps: int,
    seed: int,
    replicate: int,
    mode_table: ModeTable,
) -> tuple[list[float], int]:
    current = state
    sizes = [float(current.valid_mode_count)]
    steps = 0
    while (
        steps < max_steps
        and current.valid_mode_count > 1
        and available_experiment_ids(current, mode_table=mode_table)
    ):
        step = steps + 1
        if policy == "oracle":
            experiment_id = oracle_disagreement_experiment(
                current,
                mode_table=mode_table,
            )
        elif policy == "random_experiment":
            experiment_id = select_seeded_random_experiment(
                current,
                seed=_reference_seed(seed, current, step, policy, replicate),
                mode_table=mode_table,
            )
        else:
            rng = random.Random(
                _reference_seed(seed, current, step, policy, replicate)
            )
            modes = list(current.valid_mode_ids)
            if policy == "uniform_without_replacement":
                bank = rng.sample(modes, min(k, len(modes)))
            elif policy == "uniform_with_replacement":
                bank = [rng.choice(modes) for _ in range(k)]
            elif policy == "collapsed":
                bank = [rng.choice(modes)] * k
            else:
                raise ValueError(f"unsupported reference policy: {policy}")
            experiment_id = select_disagreement_experiment(
                bank,
                current,
                mode_table=mode_table,
            )
        if experiment_id is None:
            break
        hidden = mode_table.modes_by_id[current.hidden_mode_id]
        current = make_state(
            hidden_mode=hidden,
            evidence_ids=tuple(
                sorted((*current.observed_experiment_ids(), experiment_id))
            ),
            mode_table=mode_table,
            compute_separation=False,
        )
        sizes.append(float(current.valid_mode_count))
        steps += 1
    return _carry_curve(sizes, max_steps), steps


def summarize_reference_policies(
    states: list[EvidenceState],
    *,
    k: int,
    max_steps: int,
    seed: int,
    sampled_replicates: int,
    mode_table: ModeTable,
) -> dict[str, Any]:
    policies = (
        "oracle",
        "random_experiment",
        "uniform_without_replacement",
        "uniform_with_replacement",
        "collapsed",
    )
    result = {}
    for policy in policies:
        replicates = (
            1
            if policy in {"oracle", "random_experiment"}
            else max(1, sampled_replicates)
        )
        curves: list[list[float]] = []
        experiments = []
        initial_ms = []
        for state in states:
            for replicate in range(replicates):
                curve, steps = _reference_curve(
                    state,
                    policy=policy,
                    k=k,
                    max_steps=max_steps,
                    seed=seed,
                    replicate=replicate,
                    mode_table=mode_table,
                )
                curves.append(curve)
                experiments.append(float(steps))
                initial_ms.append(state.valid_mode_count)
        final_successes = [float(curve[-1] == 1) for curve in curves]
        budgeted_experiments = [
            steps if success else float(max_steps)
            for steps, success in zip(experiments, final_successes, strict=True)
        ]
        total_experiments = sum(budgeted_experiments)
        result[policy] = {
            "replicates_per_state": replicates,
            "identification_at_budget": statistics.fmean(final_successes),
            "mean_experiments_used": statistics.fmean(budgeted_experiments),
            "mean_experiments_executed": statistics.fmean(experiments),
            "identifications_per_100_experiments": (
                100.0 * sum(final_successes) / total_experiments
                if total_experiments
                else 0.0
            ),
            "curves": [
                {
                    "step": step,
                    "identification_success": statistics.fmean(
                        float(curve[step] == 1) for curve in curves
                    ),
                    "mean_log2_version_space_size": statistics.fmean(
                        math.log2(max(1.0, curve[step])) for curve in curves
                    ),
                }
                for step in range(max_steps + 1)
            ],
            "by_initial_M": {
                str(initial_m): {
                    "support": sum(value == initial_m for value in initial_ms),
                    "identification_at_budget": statistics.fmean(
                        float(curves[index][-1] == 1)
                        for index, value in enumerate(initial_ms)
                        if value == initial_m
                    ),
                    "mean_experiments_used": statistics.fmean(
                        budgeted_experiments[index]
                        for index, value in enumerate(initial_ms)
                        if value == initial_m
                    ),
                }
                for initial_m in sorted(set(initial_ms))
            },
        }
    return result


def summarize_trajectories(
    runtimes: list[TrajectoryRuntime],
    *,
    max_steps: int,
    bootstrap_samples: int,
    seed: int,
    mode_table: ModeTable,
) -> dict[str, Any]:
    size_curves: list[list[float]] = []
    regret_curves: list[list[float]] = []
    oracle_curves: list[list[float]] = []
    endpoint_rows: list[dict[str, Any]] = []
    for runtime in runtimes:
        sizes = [float(runtime.initial.valid_mode_count)]
        sizes.extend(
            float(record["version_space_size_after"]) for record in runtime.records
        )
        size_curve = _carry_curve(sizes, max_steps)
        size_curves.append(size_curve)
        cumulative_regret = [0.0]
        for record in runtime.records:
            cumulative_regret.append(
                cumulative_regret[-1] + float(record["entropy_regret"])
            )
        regret_curves.append(_carry_curve(cumulative_regret, max_steps))
        oracle_curves.append(
            _reference_oracle_curve(
                runtime.initial,
                max_steps=max_steps,
                mode_table=mode_table,
            )
        )
        identified_step = next(
            (
                int(record["experiment_step"])
                for record in runtime.records
                if bool(record["identified"])
            ),
            None,
        )
        log_curve = [math.log2(max(1.0, value)) for value in size_curve]
        endpoint_rows.append(
            {
                "initial_state_id": runtime.initial.state_id,
                "hidden_mode_id": runtime.initial.hidden_mode_id,
                "initial_M": runtime.initial.valid_mode_count,
                "separation_bucket": runtime.initial.separation_bucket,
                "family_bucket": runtime.initial.family_bucket,
                "steps_executed": len(runtime.records),
                "experiments_executed": len(runtime.records),
                "experiments_used": (
                    identified_step if identified_step is not None else max_steps
                ),
                "identified": size_curve[-1] == 1,
                "experiments_to_singleton": identified_step,
                "final_version_space_size": int(size_curve[-1]),
                "final_log2_version_space_size": log_curve[-1],
                "area_under_log2_version_space_curve": sum(log_curve),
                "cumulative_information_gain": log_curve[0] - log_curve[-1],
                "cumulative_entropy_regret": regret_curves[-1][-1],
                "oracle_final_version_space_size": int(oracle_curves[-1][-1]),
            }
        )

    def curve_mean(curves: list[list[float]], index: int) -> float:
        return statistics.fmean(curve[index] for curve in curves) if curves else 0.0

    curves = []
    for step in range(max_steps + 1):
        version_sizes = [curve[step] for curve in size_curves]
        identified = [float(value == 1) for value in version_sizes]
        log_sizes = [math.log2(max(1.0, value)) for value in version_sizes]
        oracle_sizes = [curve[step] for curve in oracle_curves]
        identification_ci = _bootstrap_mean_ci(
            identified,
            samples=bootstrap_samples,
            seed=seed + step,
        )
        log_size_ci = _bootstrap_mean_ci(
            log_sizes,
            samples=bootstrap_samples,
            seed=seed + 1000 + step,
        )
        curves.append(
            {
                "step": step,
                "identification_success": identification_ci["mean"],
                "identification_ci95_low": identification_ci["ci95_low"],
                "identification_ci95_high": identification_ci["ci95_high"],
                "mean_version_space_size": statistics.fmean(version_sizes)
                if version_sizes
                else 0.0,
                "median_version_space_size": statistics.median(version_sizes)
                if version_sizes
                else 0.0,
                "mean_log2_version_space_size": log_size_ci["mean"],
                "log2_version_space_ci95_low": log_size_ci["ci95_low"],
                "log2_version_space_ci95_high": log_size_ci["ci95_high"],
                "mean_cumulative_entropy_regret": curve_mean(regret_curves, step),
                "oracle_identification_success": statistics.fmean(
                    float(value == 1) for value in oracle_sizes
                )
                if oracle_sizes
                else 0.0,
                "oracle_mean_version_space_size": statistics.fmean(oracle_sizes)
                if oracle_sizes
                else 0.0,
            }
        )

    curves_by_initial_m: dict[str, list[dict[str, float | int]]] = {}
    for initial_m in sorted({runtime.initial.valid_mode_count for runtime in runtimes}):
        indices = [
            index
            for index, runtime in enumerate(runtimes)
            if runtime.initial.valid_mode_count == initial_m
        ]
        curves_by_initial_m[str(initial_m)] = [
            {
                "step": step,
                "support": len(indices),
                "identification_success": statistics.fmean(
                    float(size_curves[index][step] == 1) for index in indices
                ),
                "mean_log2_version_space_size": statistics.fmean(
                    math.log2(max(1.0, size_curves[index][step]))
                    for index in indices
                ),
                "oracle_identification_success": statistics.fmean(
                    float(oracle_curves[index][step] == 1) for index in indices
                ),
            }
            for step in range(max_steps + 1)
        ]

    final_success = [float(row["identified"]) for row in endpoint_rows]
    final_log = [
        float(row["final_log2_version_space_size"]) for row in endpoint_rows
    ]
    total_experiments = sum(int(row["experiments_used"]) for row in endpoint_rows)
    identified_count = sum(bool(row["identified"]) for row in endpoint_rows)
    summary: dict[str, Any] = {
        "trajectories": len(runtimes),
        "max_steps": max_steps,
        "curves": curves,
        "curves_by_initial_M": curves_by_initial_m,
        "identification_at_budget": statistics.fmean(final_success)
        if final_success
        else 0.0,
        "mean_experiments_used": statistics.fmean(
            float(row["experiments_used"]) for row in endpoint_rows
        )
        if endpoint_rows
        else 0.0,
        "identifications_per_100_experiments": (
            100.0 * identified_count / total_experiments
            if total_experiments
            else 0.0
        ),
        "final_identification": _bootstrap_mean_ci(
            final_success,
            samples=bootstrap_samples,
            seed=seed + 2000,
        ),
        "final_log2_version_space_size": _bootstrap_mean_ci(
            final_log,
            samples=bootstrap_samples,
            seed=seed + 3000,
        ),
        "mean_experiments_to_singleton_given_success": statistics.fmean(
            float(row["experiments_to_singleton"])
            for row in endpoint_rows
            if row["experiments_to_singleton"] is not None
        )
        if any(row["experiments_to_singleton"] is not None for row in endpoint_rows)
        else None,
        "mean_area_under_log2_version_space_curve": statistics.fmean(
            float(row["area_under_log2_version_space_curve"])
            for row in endpoint_rows
        )
        if endpoint_rows
        else 0.0,
        "mean_cumulative_information_gain": statistics.fmean(
            float(row["cumulative_information_gain"]) for row in endpoint_rows
        )
        if endpoint_rows
        else 0.0,
        "mean_cumulative_entropy_regret": statistics.fmean(
            float(row["cumulative_entropy_regret"]) for row in endpoint_rows
        )
        if endpoint_rows
        else 0.0,
        "endpoint_rows": endpoint_rows,
    }
    summary["slices"] = {}
    for key in ("initial_M", "separation_bucket"):
        values: dict[str, dict[str, float]] = {}
        for label in sorted({str(row[key]) for row in endpoint_rows}):
            rows = [row for row in endpoint_rows if str(row[key]) == label]
            experiments = sum(int(row["experiments_used"]) for row in rows)
            successes = sum(bool(row["identified"]) for row in rows)
            values[label] = {
                "support": float(len(rows)),
                "identification_success": statistics.fmean(
                    float(row["identified"]) for row in rows
                ),
                "mean_experiments_used": statistics.fmean(
                    float(row["experiments_used"]) for row in rows
                ),
                "identifications_per_100_experiments": (
                    100.0 * successes / experiments if experiments else 0.0
                ),
                "mean_final_log2_version_space_size": statistics.fmean(
                    float(row["final_log2_version_space_size"]) for row in rows
                ),
                "mean_cumulative_information_gain": statistics.fmean(
                    float(row["cumulative_information_gain"]) for row in rows
                ),
            }
        summary["slices"][key] = values
    return summary


def _write_summary_files(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    curves = summary["curves"]
    if curves:
        with (output_dir / "curves.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
            writer.writeheader()
            writer.writerows(curves)
    curves_by_m = [
        {"initial_M": int(initial_m), **row}
        for initial_m, rows in summary["curves_by_initial_M"].items()
        for row in rows
    ]
    if curves_by_m:
        with (output_dir / "curves_by_initial_M.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curves_by_m[0]))
            writer.writeheader()
            writer.writerows(curves_by_m)
    endpoints = summary["endpoint_rows"]
    if endpoints:
        with (output_dir / "trajectory_summary.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(endpoints[0]))
            writer.writeheader()
            writer.writerows(endpoints)


def _write_plots(output_dir: Path, summary: dict[str, Any]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    curves = summary["curves"]
    if not curves:
        return []
    steps = [row["step"] for row in curves]
    written = []

    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    for initial_m, rows in summary["curves_by_initial_M"].items():
        axis.plot(
            steps,
            [row["identification_success"] for row in rows],
            marker="o",
            label=f"Generated, M={initial_m}",
        )
        axis.plot(
            steps,
            [row["oracle_identification_success"] for row in rows],
            linestyle="--",
            label=f"Oracle, M={initial_m}",
        )
    axis.set(xlabel="Experiments", ylabel="Identification success", ylim=(0, 1.02))
    axis.legend()
    fig.tight_layout()
    path = output_dir / "identification_curve.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path))

    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    for initial_m, rows in summary["curves_by_initial_M"].items():
        axis.plot(
            steps,
            [row["mean_log2_version_space_size"] for row in rows],
            marker="o",
            label=f"M={initial_m}",
        )
    axis.set(xlabel="Experiments", ylabel="Mean log2 version-space size")
    axis.legend()
    fig.tight_layout()
    path = output_dir / "version_space_curve.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path))

    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(
        steps,
        [row["mean_cumulative_entropy_regret"] for row in curves],
        marker="o",
    )
    axis.set(xlabel="Experiments", ylabel="Mean cumulative entropy regret (bits)")
    fig.tight_layout()
    path = output_dir / "oracle_regret_curve.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path))
    return written


def _start_wandb(args: argparse.Namespace, output_dir: Path):
    if not args.wandb_project:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or args.run_name,
        dir=str(output_dir),
        config={
            "model": args.model,
            "input": args.input,
            "K": args.k,
            "T": args.max_steps,
            "initial_mode_counts": list(args.initial_mode_counts),
            "trajectories_per_count": args.trajectories_per_count,
            "workers": args.workers,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_response_length": args.num_predict,
            "seed": args.seed,
            "deduplicate_planner_modes": args.deduplicate_planner_modes,
            "latent_count": args.latent_count,
            "reference_samples": args.reference_samples,
        },
    )


def run_closed_loop(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    table = build_mode_table()
    states = load_states(args.input)
    states = select_initial_states(
        states,
        initial_mode_counts=args.initial_mode_counts,
        trajectories_per_count=args.trajectories_per_count,
        seed=args.seed,
    )
    if not states:
        raise ValueError("closed-loop evaluation requires at least one state")
    missing_hidden = [state.state_id for state in states if not state.hidden_mode_id]
    if missing_hidden:
        raise ValueError("all closed-loop states require private hidden_mode_id")

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.jsonl"
    existing = _read_trace(trace_path)
    if existing and not args.resume:
        raise FileExistsError(
            f"{trace_path} already exists; pass --resume or choose another run name"
        )
    runtimes = _restore_runtimes(states, existing, mode_table=table)
    backend = build_backend(args)
    wandb_run = _start_wandb(args, output_dir)
    started = time.monotonic()
    new_records = 0

    with trace_path.open("a", encoding="utf-8") as trace_handle:
        while True:
            active = [
                runtime
                for runtime in runtimes
                if runtime.is_active(args.max_steps, mode_table=table)
            ]
            if not active:
                break
            generation_records = evaluate_states(
                states=[runtime.current for runtime in active],
                backend=backend,
                model=args.model,
                rollouts_per_state=args.k,
                workers=args.workers,
                output_transcripts=True,
                progress_interval_s=args.progress_interval_s,
                thinking_fallback=args.thinking_fallback,
                fallback_num_predict=args.fallback_num_predict,
                fallback_temperature=args.fallback_temperature,
                latent_count=args.latent_count,
            )
            by_state_index: dict[int, list[dict[str, Any]]] = {}
            for record in generation_records:
                by_state_index.setdefault(int(record["state_index"]), []).append(record)
            round_records = []
            for active_index, runtime in enumerate(active):
                bank = sorted(
                    by_state_index.get(active_index, []),
                    key=lambda record: int(record["rollout_index"]),
                )
                if len(bank) != args.k:
                    raise RuntimeError(
                        f"trajectory {runtime.initial.state_id} returned "
                        f"{len(bank)} of {args.k} completions"
                    )
                record, updated = _step_record(
                    runtime,
                    bank,
                    run_name=args.run_name,
                    model=args.model,
                    seed=args.seed,
                    deduplicate_planner_modes=args.deduplicate_planner_modes,
                    mode_table=table,
                )
                trace_handle.write(json.dumps(record, sort_keys=True) + "\n")
                trace_handle.flush()
                runtime.records.append(record)
                runtime.current = updated
                round_records.append(record)
                new_records += 1
            summary = summarize_trajectories(
                runtimes,
                max_steps=args.max_steps,
                bootstrap_samples=0,
                seed=args.seed,
                mode_table=table,
            )
            curve = summary["curves"][
                min(max(len(runtime.records) for runtime in runtimes), args.max_steps)
            ]
            elapsed = time.monotonic() - started
            completed_steps = sum(len(runtime.records) for runtime in runtimes)
            print(
                "closed-loop progress:"
                f" trajectory_steps={completed_steps}"
                f" active={sum(runtime.is_active(args.max_steps, mode_table=table) for runtime in runtimes)}"
                f" identified={sum(runtime.current.valid_mode_count == 1 for runtime in runtimes)}"
                f" elapsed_s={elapsed:.1f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_metrics = {
                    "closed_loop/trajectory_steps_completed": completed_steps,
                    "closed_loop/new_trace_records": new_records,
                    "closed_loop/active_trajectories": sum(
                        runtime.is_active(args.max_steps, mode_table=table)
                        for runtime in runtimes
                    ),
                    "closed_loop/identification_success": curve[
                        "identification_success"
                    ],
                    "closed_loop/mean_log2_version_space_size": curve[
                        "mean_log2_version_space_size"
                    ],
                    "closed_loop/mean_cumulative_entropy_regret": curve[
                        "mean_cumulative_entropy_regret"
                    ],
                }
                for initial_m, rows in summary["curves_by_initial_M"].items():
                    row = rows[int(curve["step"])]
                    wandb_metrics.update(
                        {
                            f"closed_loop/M_{initial_m}/identification_success": row[
                                "identification_success"
                            ],
                            f"closed_loop/M_{initial_m}/mean_log2_version_space_size": row[
                                "mean_log2_version_space_size"
                            ],
                        }
                    )
                bank_metric_names = (
                    "parse_valid_rate",
                    "syntax_valid_rate",
                    "evidence_consistent_rate",
                    "num_unique_valid_modes",
                    "budget_normalized_coverage",
                    "duplicate_rate",
                    "effective_mode_count",
                    "generated_mode_separation",
                    "hidden_mode_in_bank",
                )
                for name in bank_metric_names:
                    wandb_metrics[f"closed_loop/bank/{name}"] = statistics.fmean(
                        float(record["bank_metrics"].get(name, 0.0))
                        for record in round_records
                    )
                wandb_metrics.update(
                    {
                        "closed_loop/step/information_gain_bits": statistics.fmean(
                            float(record["information_gain_bits"])
                            for record in round_records
                        ),
                        "closed_loop/step/selected_empirical_entropy": statistics.fmean(
                            float(record["selected_empirical_entropy"])
                            for record in round_records
                        ),
                        "closed_loop/step/request_error_rate": statistics.fmean(
                            float(record["request_error_count"] > 0)
                            for record in round_records
                        ),
                    }
                )
                wandb_run.log(wandb_metrics)

    summary = summarize_trajectories(
        runtimes,
        max_steps=args.max_steps,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        mode_table=table,
    )
    summary.update(
        {
            "run_name": args.run_name,
            "model": args.model,
            "input": args.input,
            "K": args.k,
            "T": args.max_steps,
            "initial_mode_counts": list(args.initial_mode_counts),
            "trajectories_per_count": args.trajectories_per_count,
            "seed": args.seed,
            "deduplicate_planner_modes": args.deduplicate_planner_modes,
            "trace_path": str(trace_path),
            "wall_seconds": time.monotonic() - started,
        }
    )
    summary["reference_policies"] = summarize_reference_policies(
        [runtime.initial for runtime in runtimes],
        k=args.k,
        max_steps=args.max_steps,
        seed=args.seed,
        sampled_replicates=args.reference_samples,
        mode_table=table,
    )
    _write_summary_files(output_dir, summary)
    plot_paths = _write_plots(output_dir, summary)
    summary["plot_paths"] = plot_paths
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "final_identification_success": summary["final_identification"]["mean"],
                "mean_experiments_used": summary["mean_experiments_used"],
                "identifications_per_100_experiments": summary[
                    "identifications_per_100_experiments"
                ],
                "final_mean_log2_version_space_size": summary[
                    "final_log2_version_space_size"
                ]["mean"],
                "mean_cumulative_information_gain": summary[
                    "mean_cumulative_information_gain"
                ],
                "mean_cumulative_entropy_regret": summary[
                    "mean_cumulative_entropy_regret"
                ],
            }
        )
        for initial_m, values in summary["slices"]["initial_M"].items():
            wandb_run.summary.update(
                {
                    f"M_{initial_m}/identification_at_8": values[
                        "identification_success"
                    ],
                    f"M_{initial_m}/mean_experiments_used": values[
                        "mean_experiments_used"
                    ],
                    f"M_{initial_m}/identifications_per_100_experiments": values[
                        "identifications_per_100_experiments"
                    ],
                }
            )
        import wandb

        artifact = wandb.Artifact(
            f"causal-micro-lab-closed-loop-{wandb_run.id}",
            type="closed-loop-evaluation",
        )
        artifact.add_file(str(trace_path))
        artifact.add_file(str(output_dir / "summary.json"))
        artifact.add_file(str(output_dir / "curves.csv"))
        artifact.add_file(str(output_dir / "curves_by_initial_M.csv"))
        artifact.add_file(str(output_dir / "trajectory_summary.csv"))
        for path in plot_paths:
            artifact.add_file(path)
        wandb_run.log_artifact(artifact)
        wandb_run.finish()
    return output_dir, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Model-backed Boolean Causal Micro-Lab closed-loop evaluation."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="artifacts/causal_micro_lab_closed_loop")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--provider",
        choices=("openai-compatible", "ollama", "transformers"),
        default="openai-compatible",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--initial-mode-counts", default="16,32")
    parser.add_argument("--trajectories-per-count", type=int, default=64)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-predict", type=int, default=6000)
    parser.add_argument("--request-timeout-s", type=float, default=1800)
    parser.add_argument("--think", default="true")
    parser.add_argument("--thinking-fallback", action="store_true")
    parser.add_argument("--fallback-num-predict", type=int, default=256)
    parser.add_argument("--fallback-temperature", type=float, default=0.0)
    parser.add_argument("--latent-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--deduplicate-planner-modes", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-interval-s", type=float, default=60.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--reference-samples", type=int, default=32)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.initial_mode_counts = tuple(
        int(item)
        for item in str(args.initial_mode_counts).split(",")
        if item.strip()
    )
    if args.k < 1:
        raise SystemExit("--k must be positive")
    if args.max_steps < 0:
        raise SystemExit("--max-steps must be non-negative")
    output_dir, summary = run_closed_loop(args)
    print(f"closed-loop output: {output_dir}")
    print(json.dumps({key: summary[key] for key in (
        "trajectories",
        "final_identification",
        "final_log2_version_space_size",
        "mean_cumulative_information_gain",
        "mean_cumulative_entropy_regret",
        "mean_experiments_used",
        "identifications_per_100_experiments",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
