from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from scattered_discovery.envs.causal_micro_lab.eval import (
    _read_jsonl,
    _verification_results_for_record,
    load_states,
    summarize_grouped_records,
)
from scattered_discovery.envs.causal_micro_lab.predictive_diversity import (
    RepresentativeCoverageMatrix,
)
from scattered_discovery.envs.causal_micro_lab.rewards import group_metrics
from scattered_discovery.envs.causal_micro_lab.signatures import build_mode_table
from scattered_discovery.envs.causal_micro_lab.state_generator import EvidenceState


METRICS = (
    "pass_at_k",
    "valid_mode_rate",
)
CONDITIONAL_METRICS = (
    ("modes_recovered_given_success", "num_unique_valid_modes"),
    ("fraction_modes_recovered_given_success", "exact_coverage"),
    (
        "budget_normalized_coverage_given_success",
        "budget_normalized_coverage",
    ),
    ("family_coverage_given_success", "family_coverage"),
    ("effective_mode_count_given_success", "effective_mode_count"),
    ("mode_entropy_given_success", "mode_entropy"),
    ("dominant_mode_mass_given_success", "dominant_mode_mass"),
    ("duplicity_given_success", "duplicity"),
    ("generated_mode_separation_given_success", "generated_mode_separation"),
    (
        "generated_to_available_separation_given_success",
        "generated_to_available_separation",
    ),
    (
        "predictive_diversity_recovery_given_success",
        "predictive_diversity_recovery",
    ),
    ("predictive_coverage_auc_given_success", "predictive_coverage_auc"),
    (
        "predictive_placement_regret_given_success",
        "predictive_placement_regret",
    ),
    (
        "full_outcome_generated_separation_given_success",
        "full_outcome_generated_separation",
    ),
)
PRIMARY_METRICS = (
    "pass_at_k",
    "predictive_diversity_recovery_given_success",
    "modes_recovered_given_success",
    "fraction_modes_recovered_given_success",
    "predictive_coverage_auc_given_success",
    "predictive_placement_regret_given_success",
)

COVERAGE_CURVE_RADII = tuple(index / 20 for index in range(21))


def _mean(items: list[float]) -> float:
    return fmean(items) if items else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _state_metric_rows(
    records: list[dict[str, Any]],
    states: list[EvidenceState],
    ks: tuple[int, ...],
    *,
    set_answer_count: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_state[str(record["state_id"])].append(record)
    table = build_mode_table()
    metric_rows: list[dict[str, Any]] = []
    reachability_rows: list[dict[str, Any]] = []
    for state in states:
        representative_matrix = RepresentativeCoverageMatrix(
            state.valid_mode_ids,
            state.observed_experiment_ids(),
            mode_table=table,
        )
        state_records = sorted(
            by_state[state.state_id],
            key=lambda record: int(record["rollout_index"]),
        )
        for k in ks:
            reported_k = set_answer_count if set_answer_count is not None else k
            prefix = [
                record for record in state_records if int(record["rollout_index"]) < k
            ]
            results = [
                result
                for record in prefix
                for result in _verification_results_for_record(record)
            ]
            metrics = group_metrics(
                results,
                state,
                representative_matrix=representative_matrix,
            )
            valid_count = int(
                metrics["num_unique_valid_modes"] + metrics["duplicate_valid_modes"]
            )
            metrics["valid_mode_rate"] = valid_count / max(1, len(results))
            metrics["valid_output_rate"] = metrics["valid_mode_rate"]
            metrics["valid_outputs_per_state"] = metrics["num_evidence_consistent"]
            metrics["distinct_valid_hypotheses"] = metrics["num_unique_valid_modes"]
            metrics["duplicate_valid_outputs"] = metrics["duplicate_valid_modes"]
            row: dict[str, Any] = {
                "state_id": state.state_id,
                "K": reported_k,
                "M": state.valid_mode_count,
                "evidence_size": state.evidence_size,
                "separation_bucket": state.separation_bucket,
                "family_bucket": state.family_bucket,
                "mean_separation": state.mean_separation,
                "minimum_separation": state.minimum_separation,
                "maximum_separation": state.maximum_separation,
                "separation_definition": state.separation_definition,
                "representative_coverage_opportunity": (
                    state.representative_coverage_opportunity
                ),
                **metrics,
            }
            metric_rows.append(row)
            generated = Counter(
                result.semantic_mode_id
                for result in results
                if result.is_currently_valid_mode
                and result.semantic_mode_id is not None
            )
            for mode_id in state.valid_mode_ids:
                family = "/".join(table.modes_by_id[mode_id].family)
                reachability_rows.append(
                    {
                        "state_id": state.state_id,
                        "K": reported_k,
                        "M": state.valid_mode_count,
                        "separation_bucket": state.separation_bucket,
                        "state_family_bucket": state.family_bucket,
                        "mode_id": mode_id,
                        "mode_family": family,
                        "generated_count": generated.get(mode_id, 0),
                        "discovered": int(mode_id in generated),
                    }
                )
    return metric_rows, reachability_rows


def _aggregate_rows(
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for labels, items in sorted(grouped.items(), key=lambda item: item[0]):
        result = {key: value for key, value in zip(group_keys, labels, strict=True)}
        result["support_states"] = len(items)
        successes = [item for item in items if float(item["pass_at_k"]) > 0]
        result["successful_states"] = len(successes)
        for metric in METRICS:
            result[metric] = _mean([float(item[metric]) for item in items])
        for output_metric, source_metric in CONDITIONAL_METRICS:
            result[output_metric] = _mean(
                [float(item[source_metric]) for item in successes]
            )
        output.append(result)
    return output


def _aggregate_output_counts(
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Summarize validity, distinctness, and repetition on all states."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for labels, items in sorted(grouped.items(), key=lambda item: item[0]):
        result = {key: value for key, value in zip(group_keys, labels, strict=True)}
        result.update(
            {
                "support_states": len(items),
                "pass_at_k": _mean([float(item["pass_at_k"]) for item in items]),
                "valid_output_rate": _mean(
                    [float(item["valid_mode_rate"]) for item in items]
                ),
                "valid_outputs_per_state": _mean(
                    [float(item["num_evidence_consistent"]) for item in items]
                ),
                "distinct_valid_hypotheses": _mean(
                    [float(item["num_unique_valid_modes"]) for item in items]
                ),
                "duplicate_valid_outputs": _mean(
                    [float(item["duplicate_valid_modes"]) for item in items]
                ),
            }
        )
        output.append(result)
    return output


def _aggregate_diversity_at_4(
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Aggregate Diversity@4 only where at least four valid outputs exist."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for labels, items in sorted(grouped.items(), key=lambda item: item[0]):
        defined = [item for item in items if float(item["diversity_at_4_defined"]) > 0]
        result = {key: value for key, value in zip(group_keys, labels, strict=True)}
        result.update(
            {
                "support_states": len(items),
                "defined_states": len(defined),
                "defined_rate": len(defined) / max(1, len(items)),
                "diversity_at_4": _mean(
                    [float(item["diversity_at_4"]) for item in defined]
                ),
                "oracle_diversity_at_4": _mean(
                    [float(item["oracle_diversity_at_4"]) for item in defined]
                ),
                "normalized_diversity_at_4": _mean(
                    [float(item["normalized_diversity_at_4"]) for item in defined]
                ),
            }
        )
        output.append(result)
    return output


def _aggregate_primary_rows(
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for labels, items in sorted(grouped.items(), key=lambda item: item[0]):
        successes = [item for item in items if float(item["pass_at_k"]) > 0]
        result = {key: value for key, value in zip(group_keys, labels, strict=True)}
        result.update(
            {
                "support_states": len(items),
                "successful_states": len(successes),
                "pass_at_k": _mean([float(item["pass_at_k"]) for item in items]),
                "predictive_diversity_recovery_given_success": _mean(
                    [float(item["predictive_diversity_recovery"]) for item in successes]
                ),
                "modes_recovered_given_success": _mean(
                    [float(item["num_unique_valid_modes"]) for item in successes]
                ),
                "fraction_modes_recovered_given_success": _mean(
                    [float(item["exact_coverage"]) for item in successes]
                ),
                "predictive_coverage_auc_given_success": _mean(
                    [float(item["predictive_coverage_auc"]) for item in successes]
                ),
                "predictive_placement_regret_given_success": _mean(
                    [float(item["predictive_placement_regret"]) for item in successes]
                ),
            }
        )
        output.append(result)
    return output


def _bootstrap_grouped_rows(
    rows: list[dict[str, Any]],
    *,
    slice_name: str,
    group_keys: tuple[str, ...],
    samples: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for labels, items in sorted(grouped.items(), key=lambda item: item[0]):
        successes = [item for item in items if float(item["pass_at_k"]) > 0]
        metric_specs = [(metric, metric, "all_states", items) for metric in METRICS]
        metric_specs.extend(
            (
                output_metric,
                source_metric,
                "successful_states",
                successes,
            )
            for output_metric, source_metric in CONDITIONAL_METRICS
        )
        for output_metric, source_metric, conditioning, source_items in metric_specs:
            if not source_items:
                continue
            values = [float(item[source_metric]) for item in source_items]
            estimates = [
                _mean([values[rng.randrange(len(values))] for _ in range(len(values))])
                for _ in range(samples)
            ]
            estimates.sort()
            result = {
                "slice": slice_name,
                **{key: value for key, value in zip(group_keys, labels, strict=True)},
                "metric": output_metric,
                "conditioning": conditioning,
                "support_states": len(items),
                "successful_states": len(successes),
                "mean": _mean(values),
                "ci95_low": estimates[int(0.025 * (samples - 1))],
                "ci95_high": estimates[int(0.975 * (samples - 1))],
                "bootstrap_samples": samples,
            }
            output.append(result)
    return output


def _bootstrap_rows(
    rows: list[dict[str, Any]],
    *,
    samples: int = 1000,
    seed: int = 20260726,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    slices = (
        ("by_k", ("K",)),
        ("by_k_m", ("K", "M")),
        ("by_k_separation", ("K", "separation_bucket")),
        (
            "by_k_m_separation",
            ("K", "M", "separation_bucket"),
        ),
        ("by_k_family", ("K", "family_bucket")),
    )
    return [
        result
        for slice_name, group_keys in slices
        for result in _bootstrap_grouped_rows(
            rows,
            slice_name=slice_name,
            group_keys=group_keys,
            samples=samples,
            rng=rng,
        )
    ]


def _primary_bootstrap_rows(
    bootstrap_rows: list[dict[str, Any]],
    *,
    slice_name: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in bootstrap_rows
        if row["slice"] == slice_name and row["metric"] in PRIMARY_METRICS
    ]


def _mode_family_rows(reachability: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in reachability:
        grouped[
            (
                int(row["K"]),
                int(row["M"]),
                str(row["separation_bucket"]),
                str(row["mode_family"]),
            )
        ].append(row)
    output = []
    for labels, items in sorted(grouped.items()):
        k, m, separation, family = labels
        output.append(
            {
                "K": k,
                "M": m,
                "separation_bucket": separation,
                "mode_family": family,
                "mode_opportunities": len(items),
                "discovered_opportunities": sum(
                    int(item["discovered"]) for item in items
                ),
                "discovery_rate": _mean([float(item["discovered"]) for item in items]),
                "mean_generated_count": _mean(
                    [float(item["generated_count"]) for item in items]
                ),
            }
        )
    return output


def _coverage_curve_rows(
    reachability: list[dict[str, Any]],
    states: list[EvidenceState],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    states_by_id = {state.state_id: state for state in states}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in reachability:
        grouped[(str(row["state_id"]), int(row["K"]))].append(row)

    state_rows = []
    for (state_id, k), items in sorted(grouped.items()):
        state = states_by_id[state_id]
        generated = tuple(
            str(item["mode_id"]) for item in items if int(item["generated_count"]) > 0
        )
        matrix = RepresentativeCoverageMatrix(
            state.valid_mode_ids,
            state.observed_experiment_ids(),
        )
        coverage = matrix.coverage_curve(generated, COVERAGE_CURVE_RADII)
        for radius, value in zip(COVERAGE_CURVE_RADII, coverage, strict=True):
            state_rows.append(
                {
                    "state_id": state_id,
                    "K": k,
                    "M": state.valid_mode_count,
                    "radius": radius,
                    "predictive_coverage": value,
                    "pass_at_k": float(bool(generated)),
                    "representative_coverage_opportunity": (
                        state.representative_coverage_opportunity
                    ),
                }
            )

    aggregated = []
    grouped_curves: dict[tuple[int, int, float], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in state_rows:
        grouped_curves[(int(row["K"]), int(row["M"]), float(row["radius"]))].append(row)
    for (k, mode_count, radius), items in sorted(grouped_curves.items()):
        successes = [item for item in items if float(item["pass_at_k"]) > 0]
        aggregated.append(
            {
                "K": k,
                "M": mode_count,
                "radius": radius,
                "support_states": len(items),
                "successful_states": len(successes),
                "predictive_coverage_given_success": _mean(
                    [float(item["predictive_coverage"]) for item in successes]
                ),
            }
        )
    return state_rows, aggregated


def _sample_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    initial_tokens = [
        int(record["initial_completion_tokens"])
        for record in records
        if record.get("initial_completion_tokens") is not None
    ]
    fallback_tokens = [
        int(record["fallback_completion_tokens"])
        for record in records
        if record.get("fallback_completion_tokens") is not None
    ]
    return {
        "episodes": len(records),
        "initial_length_cap_rate": _mean(
            [
                float(record.get("initial_finish_reason") in {"length", "max_tokens"})
                for record in records
            ]
        ),
        "fallback_rate": _mean(
            [float(bool(record.get("fallback_used"))) for record in records]
        ),
        "fallback_output_rate": _mean(
            [float(bool(record.get("fallback_produced_output"))) for record in records]
        ),
        "fallback_error_rate": _mean(
            [float(bool(record.get("fallback_request_error"))) for record in records]
        ),
        "initial_tokens_mean": _mean([float(value) for value in initial_tokens]),
        "fallback_tokens_mean_when_used": _mean(
            [float(value) for value in fallback_tokens]
        ),
        "thinking_chars_mean": _mean(
            [float(len(str(record.get("thinking") or ""))) for record in records]
        ),
        "model_seconds_total": sum(
            float(record.get("model_seconds") or 0.0) for record in records
        ),
    }


def build_report(
    *,
    episodes_path: Path,
    states_path: Path,
    output_dir: Path,
    ks: tuple[int, ...],
    bootstrap_samples: int = 1000,
    set_answer_count: int | None = None,
) -> dict[str, Any]:
    records = _read_jsonl(episodes_path)
    states = load_states(states_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows, reachability_rows = _state_metric_rows(
        records,
        states,
        ks,
        set_answer_count=set_answer_count,
    )
    by_k = _aggregate_rows(metric_rows, ("K",))
    output_counts_by_k = _aggregate_output_counts(metric_rows, ("K",))
    diversity_at_4_by_k = _aggregate_diversity_at_4(metric_rows, ("K",))
    by_k_m = _aggregate_rows(metric_rows, ("K", "M"))
    by_k_separation = _aggregate_rows(metric_rows, ("K", "separation_bucket"))
    by_k_m_separation = _aggregate_rows(
        metric_rows,
        ("K", "M", "separation_bucket"),
    )
    by_k_family = _aggregate_rows(metric_rows, ("K", "family_bucket"))
    primary_by_k_m = _aggregate_primary_rows(metric_rows, ("K", "M"))
    primary_by_k_separation = _aggregate_primary_rows(
        metric_rows,
        ("K", "separation_bucket"),
    )
    primary_by_k_family = _aggregate_primary_rows(
        metric_rows,
        ("K", "family_bucket"),
    )
    bootstrap_rows = _bootstrap_rows(
        metric_rows,
        samples=bootstrap_samples,
    )
    mode_family_rows = _mode_family_rows(reachability_rows)
    coverage_curve_rows, coverage_curve_summary = _coverage_curve_rows(
        reachability_rows,
        states,
    )
    sample_summary = _sample_summary(records)
    set_summaries = {
        str(set_answer_count if set_answer_count is not None else k): (
            summarize_grouped_records(
                records,
                states,
                max_rollout_index=k,
            )
        )
        for k in ks
    }
    summary = {
        "episodes_path": str(episodes_path),
        "states_path": str(states_path),
        "states": len(states),
        "episodes": len(records),
        "Ks": [set_answer_count if set_answer_count is not None else k for k in ks],
        "Ms": sorted({state.valid_mode_count for state in states}),
        "predictive_coverage_definition": "full_outcome_facility_location_v3",
        "coverage_curve_radii": list(COVERAGE_CURVE_RADII),
        "sample_summary": sample_summary,
        "set_summaries": set_summaries,
    }
    _write_csv(output_dir / "state_metrics.csv", metric_rows)
    _write_csv(output_dir / "metrics_by_k.csv", by_k)
    _write_csv(output_dir / "output_counts_by_k.csv", output_counts_by_k)
    _write_csv(output_dir / "diversity_at_4_by_k.csv", diversity_at_4_by_k)
    _write_csv(output_dir / "metrics_by_k_m.csv", by_k_m)
    _write_csv(output_dir / "metrics_by_k_separation.csv", by_k_separation)
    _write_csv(
        output_dir / "metrics_by_k_m_separation.csv",
        by_k_m_separation,
    )
    _write_csv(output_dir / "metrics_by_k_family.csv", by_k_family)
    _write_csv(
        output_dir / "primary_metrics_by_k_m.csv",
        primary_by_k_m,
    )
    _write_csv(
        output_dir / "primary_metrics_by_k_separation.csv",
        primary_by_k_separation,
    )
    _write_csv(
        output_dir / "primary_metrics_by_k_family.csv",
        primary_by_k_family,
    )
    _write_csv(output_dir / "bootstrap_ci95.csv", bootstrap_rows)
    _write_csv(
        output_dir / "primary_bootstrap_ci95_by_k_m.csv",
        _primary_bootstrap_rows(bootstrap_rows, slice_name="by_k_m"),
    )
    _write_csv(
        output_dir / "primary_bootstrap_ci95_by_k_separation.csv",
        _primary_bootstrap_rows(
            bootstrap_rows,
            slice_name="by_k_separation",
        ),
    )
    _write_csv(
        output_dir / "primary_bootstrap_ci95_by_k_family.csv",
        _primary_bootstrap_rows(
            bootstrap_rows,
            slice_name="by_k_family",
        ),
    )
    _write_csv(output_dir / "mode_reachability.csv", reachability_rows)
    _write_csv(output_dir / "mode_discovery_by_family.csv", mode_family_rows)
    _write_csv(
        output_dir / "predictive_coverage_curves_by_state.csv", coverage_curve_rows
    )
    _write_csv(
        output_dir / "predictive_coverage_curves_by_k_m.csv", coverage_curve_summary
    )
    _write_csv(
        output_dir / "sample_summary.csv",
        [{"metric": key, "value": value} for key, value in sample_summary.items()],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--ks", default="4,8,12,16")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    episodes = Path(args.episodes)
    output_dir = (
        Path(args.output_dir) if args.output_dir else episodes.parent / "report"
    )
    ks = tuple(int(item) for item in args.ks.split(",") if item.strip())
    summary = build_report(
        episodes_path=episodes,
        states_path=Path(args.states),
        output_dir=output_dir,
        ks=ks,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(
        json.dumps(
            {
                "states": summary["states"],
                "episodes": summary["episodes"],
                "Ks": summary["Ks"],
                "Ms": summary["Ms"],
                "sample_summary": summary["sample_summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"report_csv_dir={output_dir}")


if __name__ == "__main__":
    main()
