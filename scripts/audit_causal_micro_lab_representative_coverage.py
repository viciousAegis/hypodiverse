#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
import math
from pathlib import Path
import random
from statistics import fmean
from typing import Any

from scattered_discovery.envs.causal_micro_lab.eval import load_states
from scattered_discovery.envs.causal_micro_lab.predictive_diversity import (
    RepresentativeCoverageMatrix,
)


POLICIES = (
    "collapsed",
    "concentrated_sampling",
    "uniform_sampling",
    "uniform_distinct",
    "pairwise_oracle",
    "representative_oracle",
)
POLICY_LABELS = {
    "collapsed": "Collapsed",
    "concentrated_sampling": "Concentrated sampling",
    "uniform_sampling": "Uniform sampling",
    "uniform_distinct": "Uniform distinct",
    "pairwise_oracle": "Pairwise-distance oracle",
    "representative_oracle": "Representative oracle",
}


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _bootstrap(
    values: list[float],
    *,
    rng: random.Random,
    samples: int,
) -> tuple[float, float, float]:
    estimates = sorted(
        _mean([rng.choice(values) for _ in values]) for _ in range(samples)
    )
    return (
        _mean(values),
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    )


def _policy_scores(
    state: Any,
    *,
    budget: int,
    draws: int,
    seed: int,
    concentration_temperature: float,
) -> dict[str, float]:
    matrix = RepresentativeCoverageMatrix(
        state.valid_mode_ids,
        state.observed_experiment_ids(),
    )
    modes = tuple(sorted(state.valid_mode_ids))
    target_size = min(budget, len(modes))
    representative_error, representative_modes = matrix.optimal_subset(target_size)
    medoid_error, medoid_modes = matrix.optimal_subset(1)
    medoid = medoid_modes[0]
    subsets = tuple(combinations(modes, target_size))
    pairwise_modes = max(
        subsets,
        key=lambda subset: (
            sum(
                matrix.distance(left, right) for left, right in combinations(subset, 2)
            ),
            tuple(reversed(subset)),
        ),
    )
    concentrated_weights = [
        math.exp(
            -matrix.distance(medoid, mode_id) / max(1e-9, concentration_temperature)
        )
        for mode_id in modes
    ]
    rng = random.Random(f"{seed}:{state.state_id}:{budget}")
    samples: dict[str, list[tuple[str, ...]]] = {
        "concentrated_sampling": [],
        "uniform_sampling": [],
        "uniform_distinct": [],
    }
    for _ in range(draws):
        samples["concentrated_sampling"].append(
            tuple(rng.choices(modes, weights=concentrated_weights, k=budget))
        )
        samples["uniform_sampling"].append(
            tuple(rng.choice(modes) for _ in range(budget))
        )
        samples["uniform_distinct"].append(tuple(rng.sample(modes, target_size)))

    scores = {
        "collapsed": 1.0 - medoid_error,
        "pairwise_oracle": 1.0 - matrix.representation_error(pairwise_modes),
        "representative_oracle": 1.0 - representative_error,
    }
    for policy, generated_sets in samples.items():
        scores[policy] = _mean(
            [
                1.0 - matrix.representation_error(generated)
                for generated in generated_sets
            ]
        )
    return scores


def _write_plot(summary: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = [float(summary[policy]["mean"]) for policy in POLICIES]
    lower_errors = [
        means[index] - float(summary[policy]["ci95"][0])
        for index, policy in enumerate(POLICIES)
    ]
    upper_errors = [
        float(summary[policy]["ci95"][1]) - means[index]
        for index, policy in enumerate(POLICIES)
    ]
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    positions = list(range(len(POLICIES)))
    axis.bar(
        positions,
        means,
        color=("#9CA3AF", "#D97706", "#2563EB", "#059669", "#7C3AED", "#111827"),
        width=0.72,
    )
    axis.errorbar(
        positions,
        means,
        yerr=(lower_errors, upper_errors),
        fmt="none",
        ecolor="#111827",
        capsize=3,
        linewidth=1,
    )
    axis.set_xticks(positions, [POLICY_LABELS[policy] for policy in POLICIES])
    axis.tick_params(axis="x", rotation=24)
    axis.set_ylabel("Predictive coverage AUC")
    axis.set_ylim(0.6, 0.95)
    axis.set_title(
        "The geometry-aware benchmark distinguishes reference strategies",
        loc="left",
        fontweight="bold",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def run_audit(
    *,
    states_path: Path,
    output_dir: Path,
    budget: int,
    draws: int,
    bootstrap_samples: int,
    seed: int,
    concentration_temperature: float,
) -> dict[str, Any]:
    states = load_states(states_path)
    rows = []
    for state in states:
        scores = _policy_scores(
            state,
            budget=budget,
            draws=draws,
            seed=seed,
            concentration_temperature=concentration_temperature,
        )
        for policy, score in scores.items():
            rows.append(
                {
                    "state_id": state.state_id,
                    "M": state.valid_mode_count,
                    "K": budget,
                    "policy": policy,
                    "predictive_coverage_auc": score,
                }
            )

    rng = random.Random(seed)
    summary = {}
    for policy in POLICIES:
        values = [
            float(row["predictive_coverage_auc"])
            for row in rows
            if row["policy"] == policy
        ]
        point, low, high = _bootstrap(
            values,
            rng=rng,
            samples=bootstrap_samples,
        )
        summary[policy] = {
            "mean": point,
            "ci95": [low, high],
            "states": len(values),
        }

    expected_order = [summary[policy]["mean"] for policy in POLICIES]
    rankability_passed = all(
        right > left for left, right in zip(expected_order, expected_order[1:])
    )
    result = {
        "states": len(states),
        "budget": budget,
        "draws_per_stochastic_policy": draws,
        "distance": "full_outcome_disagreement_v3",
        "metric": "predictive_coverage_auc",
        "summary": summary,
        "rankability_passed": rankability_passed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "state_policy_scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_plot(summary, output_dir / "reference_policy_rankability.png")
    if not rankability_passed:
        raise RuntimeError(
            "representative-coverage reference policies were not rankable"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--states",
        type=Path,
        default=Path("eval_sets/causal_micro_lab/final_v3/states.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/causal_micro_lab_environment_characterization/representative_v3"
        ),
    )
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--concentration-temperature", type=float, default=0.08)
    args = parser.parse_args()
    result = run_audit(
        states_path=args.states,
        output_dir=args.output_dir,
        budget=args.budget,
        draws=args.draws,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        concentration_temperature=args.concentration_temperature,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
