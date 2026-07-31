#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODE_COUNTS = (4, 8, 12, 16)
DEFAULT_ORDER = (
    "collapsed",
    "concentrated_sampling",
    "uniform_sampling",
    "uniform_distinct",
    "oracle_dispersed",
)
LABELS = {
    "collapsed": "Collapsed",
    "concentrated_sampling": "Concentrated sampling",
    "uniform_sampling": "Uniform sampling",
    "uniform_distinct": "Uniform distinct",
    "oracle_dispersed": "Oracle dispersed",
    "base": "Base model",
    "validity": "Validity GRPO",
    "ips": "IPS-GRPO",
    "latent_ips": "Latent IPS-GRPO",
}
COLORS = {
    "collapsed": "#64748B",
    "concentrated_sampling": "#D97706",
    "uniform_sampling": "#2563EB",
    "uniform_distinct": "#0F766E",
    "oracle_dispersed": "#166534",
    "base": "#64748B",
    "validity": "#2563EB",
    "ips": "#D97706",
    "latent_ips": "#7C3AED",
}
LINESTYLES = {
    "oracle_dispersed": "--",
}
MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">")


@dataclass(frozen=True)
class Score:
    method: str
    mode_count: int
    separation: float
    value: float


def _load_scores(path: Path) -> tuple[list[Score], int]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")

    columns = set(rows[0])
    method_column = "method" if "method" in columns else "policy"
    separation_column = (
        "available_predictive_separation"
        if "available_predictive_separation" in columns
        else "separation"
    )
    value_column = "pdr_at_k" if "pdr_at_k" in columns else "pdr"
    required = {method_column, "M", separation_column, value_column}
    missing = required - columns
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    scores = [
        Score(
            method=str(row[method_column]),
            mode_count=int(row["M"]),
            separation=float(row[separation_column]),
            value=float(row[value_column]),
        )
        for row in rows
        if int(row["M"]) in MODE_COUNTS
    ]
    budgets = {int(row["K"]) for row in rows if row.get("K")}
    budget = budgets.pop() if len(budgets) == 1 else 0
    return scores, budget


def _method_order(scores: list[Score]) -> list[str]:
    present = {score.method for score in scores}
    ordered = [method for method in DEFAULT_ORDER if method in present]
    return ordered + sorted(present - set(ordered))


def _quantile_bins(scores: list[Score], bins: int) -> list[list[Score]]:
    ordered = sorted(scores, key=lambda score: score.separation)
    return [list(chunk) for chunk in np.array_split(ordered, bins) if len(chunk)]


def _bootstrap_ci(
    values: list[float],
    *,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        return float(array[0]), float(array[0])
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    estimates = array[indices].mean(axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def build_plot(
    *,
    scores_path: Path,
    output_path: Path,
    bins: int,
    bootstrap_samples: int,
) -> None:
    scores, budget = _load_scores(scores_path)
    methods = _method_order(scores)
    reference_preview = set(methods).issubset(DEFAULT_ORDER)
    grouped: dict[tuple[int, str], list[Score]] = defaultdict(list)
    for score in scores:
        grouped[(score.mode_count, score.method)].append(score)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.8), sharex=True, sharey=True)
    rng = np.random.default_rng(20260731)

    for axis, mode_count in zip(axes.flat, MODE_COUNTS, strict=True):
        state_separations = sorted(
            {
                score.separation
                for score in scores
                if score.mode_count == mode_count
            }
        )
        axis.vlines(
            state_separations,
            ymin=-0.015,
            ymax=0.008,
            color="#CBD5E1",
            linewidth=0.65,
            clip_on=False,
        )

        for method_index, method in enumerate(methods):
            method_scores = grouped[(mode_count, method)]
            if not method_scores:
                continue
            chunks = _quantile_bins(method_scores, bins)
            x_values = [
                float(np.mean([score.separation for score in chunk]))
                for chunk in chunks
            ]
            y_values = [
                float(np.mean([score.value for score in chunk])) for chunk in chunks
            ]
            intervals = [
                _bootstrap_ci(
                    [score.value for score in chunk],
                    rng=rng,
                    samples=bootstrap_samples,
                )
                for chunk in chunks
            ]
            low = [interval[0] for interval in intervals]
            high = [interval[1] for interval in intervals]
            color = COLORS.get(method, plt.get_cmap("tab10")(method_index))
            axis.fill_between(x_values, low, high, color=color, alpha=0.10)
            axis.plot(
                x_values,
                y_values,
                color=color,
                marker=MARKERS[method_index % len(MARKERS)],
                markersize=4.5,
                linewidth=1.8,
                linestyle=LINESTYLES.get(method, "-"),
                label=LABELS.get(method, method.replace("_", " ").title()),
            )

        axis.set_title(f"M = {mode_count}", loc="left", fontweight="bold")
        axis.set_xlim(0.025, 0.505)
        axis.set_ylim(-0.02, 1.04)
        axis.grid(color="#CBD5E1", alpha=0.62, linewidth=0.7)
        axis.text(
            0.98,
            0.04,
            f"{max(len(grouped[(mode_count, method)]) for method in methods)} states",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.8,
            color="#64748B",
        )

    for axis in axes[-1, :]:
        axis.set_xlabel("Available predictive separation")
    for axis in axes[:, 0]:
        axis.set_ylabel(f"Predictive Diversity Recovery@{budget or 'K'}")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=min(len(methods), 5),
        frameon=False,
        fontsize=9.5,
    )
    fig.suptitle(
        "Diversity recovery across continuously controlled separation",
        x=0.075,
        y=0.995,
        ha="left",
        fontsize=15,
    )
    fig.text(
        0.075,
        0.935,
        (
            "Reference-policy preview for benchmark validation"
            if reference_preview
            else "Trained methods evaluated on the frozen continuous benchmark"
        ),
        ha="left",
        fontsize=10,
        color="#475569",
    )
    fig.text(
        0.075,
        0.018,
        (
            "Points are means over equal-count intervals of the continuous axis; "
            "bands are state-bootstrap 95% confidence intervals. Rug marks show "
            "the evaluated states."
        ),
        ha="left",
        fontsize=9.2,
        color="#475569",
    )
    fig.tight_layout(rect=(0.04, 0.055, 1, 0.86), h_pad=2.2, w_pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path(
            "artifacts/causal_micro_lab_diversity_rankability/final_v2/"
            "policy_state_scores.csv"
        ),
        help=(
            "Per-state CSV with method/policy, M, K, separation, and PDR columns."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/figures/causal_micro_lab_performance_vs_separation.png"
        ),
    )
    parser.add_argument("--bins", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    build_plot(
        scores_path=args.scores,
        output_path=args.output,
        bins=args.bins,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
