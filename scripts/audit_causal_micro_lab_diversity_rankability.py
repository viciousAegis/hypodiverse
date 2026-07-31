#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any

from scattered_discovery.envs.causal_micro_lab.benchmark_v2 import (
    select_continuous_states,
)
from scattered_discovery.envs.causal_micro_lab.parser import parse_record_state
from scattered_discovery.envs.causal_micro_lab.predictive_diversity import (
    PredictiveDistanceMatrix,
)
from scattered_discovery.envs.causal_micro_lab.state_generator import EvidenceState


POLICY_ORDER = (
    "collapsed",
    "concentrated_sampling",
    "uniform_sampling",
    "uniform_distinct",
    "oracle_dispersed",
)


def _load_states(path: Path) -> list[EvidenceState]:
    return [
        parse_record_state(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _weighted_sample(
    rng: random.Random,
    population: tuple[str, ...],
    weights: list[float],
    budget: int,
) -> tuple[str, ...]:
    return tuple(rng.choices(population, weights=weights, k=budget))


def _policy_scores(
    state: EvidenceState,
    *,
    budget: int,
    draws: int,
    seed: int,
    concentration_temperature: float,
) -> dict[str, float]:
    matrix = PredictiveDistanceMatrix(
        state.valid_mode_ids,
        state.observed_experiment_ids(),
    )
    modes = tuple(sorted(state.valid_mode_ids))
    oracle_mass, oracle_modes = matrix.optimal_subset(budget)
    if oracle_mass <= 0.0:
        return {policy: 0.0 for policy in POLICY_ORDER}

    medoid = min(
        modes,
        key=lambda mode_id: (
            sum(matrix.distance(mode_id, other) for other in modes),
            mode_id,
        ),
    )
    concentrated_weights = [
        math.exp(
            -matrix.distance(medoid, mode_id) / max(1e-9, concentration_temperature)
        )
        for mode_id in modes
    ]
    rng = random.Random(f"{seed}:{state.state_id}:{budget}")
    scores: dict[str, list[float]] = defaultdict(list)
    scores["collapsed"].append(
        matrix.predictive_diversity_recovery((medoid,) * budget).score
    )
    scores["oracle_dispersed"].append(
        matrix.predictive_diversity_recovery(oracle_modes, budget=budget).score
    )
    for _ in range(draws):
        scores["concentrated_sampling"].append(
            matrix.predictive_diversity_recovery(
                _weighted_sample(
                    rng,
                    modes,
                    concentrated_weights,
                    budget,
                )
            ).score
        )
        scores["uniform_sampling"].append(
            matrix.predictive_diversity_recovery(
                tuple(rng.choice(modes) for _ in range(budget))
            ).score
        )
        distinct_size = min(budget, len(modes))
        scores["uniform_distinct"].append(
            matrix.predictive_diversity_recovery(
                tuple(rng.sample(modes, distinct_size)),
                budget=budget,
            ).score
        )
    return {policy: mean(scores[policy]) for policy in POLICY_ORDER}


def _bootstrap_interval(
    values: list[float],
    *,
    rng: random.Random,
    samples: int,
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    estimates = sorted(mean(rng.choice(values) for _ in values) for _ in range(samples))
    low = estimates[int(0.025 * (samples - 1))]
    high = estimates[int(0.975 * (samples - 1))]
    return (low, high)


def _summarize(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    by_policy: dict[str, list[float]] = defaultdict(list)
    by_state: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        policy = str(row["policy"])
        score = float(row["pdr_at_k"])
        by_policy[policy].append(score)
        by_state[str(row["state_id"])][policy] = score

    rng = random.Random(seed)
    overall = {}
    for policy in POLICY_ORDER:
        values = by_policy[policy]
        low, high = _bootstrap_interval(
            values,
            rng=rng,
            samples=bootstrap_samples,
        )
        overall[policy] = {
            "mean": mean(values),
            "ci95": [low, high],
            "states": len(values),
        }

    adjacent = []
    for left, right in zip(POLICY_ORDER, POLICY_ORDER[1:]):
        differences = [
            scores[right] - scores[left]
            for scores in by_state.values()
            if left in scores and right in scores
        ]
        low, high = _bootstrap_interval(
            differences,
            rng=rng,
            samples=bootstrap_samples,
        )
        adjacent.append(
            {
                "left": left,
                "right": right,
                "mean_difference": mean(differences),
                "ci95": [low, high],
                "strictly_positive_ci": low > 0.0,
            }
        )

    return {
        "overall": overall,
        "adjacent_policy_differences": adjacent,
        "rankability_passed": all(item["strictly_positive_ci"] for item in adjacent),
    }


def _write_plot(summary: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [policy.replace("_", " ").title() for policy in POLICY_ORDER]
    means = [summary["overall"][policy]["mean"] for policy in POLICY_ORDER]
    lows = [
        means[index] - summary["overall"][policy]["ci95"][0]
        for index, policy in enumerate(POLICY_ORDER)
    ]
    highs = [
        summary["overall"][policy]["ci95"][1] - means[index]
        for index, policy in enumerate(POLICY_ORDER)
    ]
    fig, axis = plt.subplots(figsize=(10, 5.5))
    colors = ["#9CA3AF", "#D97706", "#3B82F6", "#14B8A6", "#166534"]
    axis.bar(labels, means, color=colors, width=0.72)
    axis.errorbar(
        range(len(labels)),
        means,
        yerr=[lows, highs],
        fmt="none",
        ecolor="#111827",
        capsize=4,
        linewidth=1.2,
    )
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Predictive Diversity Recovery at K")
    axis.set_title(
        "Synthetic policies test whether the benchmark is rankable", loc="left"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_audit(
    *,
    states_path: Path,
    output_dir: Path,
    states_per_m: int,
    budget: int,
    draws: int,
    bootstrap_samples: int,
    seed: int,
    concentration_temperature: float,
) -> dict[str, Any]:
    states = select_continuous_states(
        _load_states(states_path),
        states_per_m=states_per_m,
        seed=seed,
    )
    rows = []
    for index, state in enumerate(states, start=1):
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
                    "available_predictive_separation": state.mean_separation,
                    "K": budget,
                    "policy": policy,
                    "pdr_at_k": score,
                }
            )
        if index == 1 or index % 25 == 0 or index == len(states):
            print(f"rankability audit: {index}/{len(states)} states", flush=True)

    summary = _summarize(
        rows,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    summary.update(
        {
            "states_path": str(states_path),
            "states": len(states),
            "states_per_M": states_per_m,
            "budget": budget,
            "draws_per_stochastic_policy": draws,
            "bootstrap_samples": bootstrap_samples,
            "concentration_temperature": concentration_temperature,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "policy_state_scores.csv"
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_path = output_dir / "rankability.png"
    try:
        _write_plot(summary, plot_path)
    except (ImportError, OSError) as exc:
        summary["plot_error"] = str(exc)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"plot_skipped={exc}")
    print(f"rankability_passed={summary['rankability_passed']}")
    print(f"summary={summary_path}")
    if plot_path.exists():
        print(f"plot={plot_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether predictive-diversity recovery separates synthetic "
            "policies with known diversity behavior."
        )
    )
    parser.add_argument(
        "--states",
        type=Path,
        default=Path(
            "artifacts/causal_micro_lab_environment_characterization/"
            "predictive_v2/states.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/causal_micro_lab_diversity_rankability/v2"),
    )
    parser.add_argument("--states-per-m", type=int, default=48)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--draws", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--concentration-temperature", type=float, default=0.04)
    args = parser.parse_args()
    run_audit(
        states_path=args.states,
        output_dir=args.output_dir,
        states_per_m=args.states_per_m,
        budget=args.budget,
        draws=args.draws,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        concentration_temperature=args.concentration_temperature,
    )


if __name__ == "__main__":
    main()
