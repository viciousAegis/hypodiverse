#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("base", "validity", "ips", "latent_ips")
LABELS = {
    "base": "Base",
    "validity": "Validity GRPO",
    "ips": "IPS-GRPO",
    "latent_ips": "Latent-IPS v2",
}
COLORS = {
    "base": "#64748B",
    "validity": "#2563EB",
    "ips": "#D97706",
    "latent_ips": "#7C3AED",
}
MARKERS = {"base": "o", "validity": "s", "ips": "^", "latent_ips": "D"}
KS = (4, 8, 12, 16)
MODE_COUNTS = (4, 8, 12, 16)
SUMMARY_METRICS = (
    "pass_at_k",
    "valid_mode_rate",
    "num_unique_valid_modes",
    "exact_coverage",
    "predictive_diversity_recovery",
    "effective_mode_count",
    "duplicity",
)
PAIRED_METRICS = (
    "pass_at_k",
    "valid_mode_rate",
    "exact_coverage",
    "predictive_diversity_recovery",
)


@dataclass(frozen=True)
class Record:
    method: str
    state_id: str
    k: int
    mode_count: int
    separation: float
    values: dict[str, float]


def _read_records(root: Path) -> list[Record]:
    records: list[Record] = []
    for method in METHODS:
        path = root / method / "state_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            records.append(
                Record(
                    method=method,
                    state_id=str(row["state_id"]),
                    k=int(row["K"]),
                    mode_count=int(row["M"]),
                    separation=float(row["available_predictive_separation"]),
                    values={
                        key: float(value)
                        for key, value in row.items()
                        if key
                        not in {
                            "state_id",
                            "K",
                            "M",
                            "family_bucket",
                            "separation_bucket",
                        }
                        and value not in {None, ""}
                    },
                )
            )
    _validate_alignment(records)
    return records


def _validate_alignment(records: list[Record]) -> None:
    by_method: dict[str, dict[tuple[str, int], Record]] = defaultdict(dict)
    for record in records:
        key = (record.state_id, record.k)
        if key in by_method[record.method]:
            raise ValueError(f"Duplicate row for {record.method}: {key}")
        by_method[record.method][key] = record
    expected = set(by_method[METHODS[0]])
    for method in METHODS:
        observed = set(by_method[method])
        if observed != expected:
            raise ValueError(
                f"State/K mismatch for {method}: "
                f"missing={len(expected - observed)}, extra={len(observed - expected)}"
            )
        for key in expected:
            reference = by_method[METHODS[0]][key]
            current = by_method[method][key]
            if (
                current.mode_count != reference.mode_count
                or abs(current.separation - reference.separation) > 1e-12
            ):
                raise ValueError(f"State metadata mismatch for {method}: {key}")


def _bootstrap_mean(
    values: Iterable[float],
    *,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return (0.0, 0.0, 0.0)
    mean = float(array.mean())
    if len(array) == 1:
        return (mean, mean, mean)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    estimates = array[indices].mean(axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return (mean, float(low), float(high))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(
    records: list[Record], *, samples: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    output = []
    for method in METHODS:
        for k in KS:
            items = [r for r in records if r.method == method and r.k == k]
            row: dict[str, Any] = {
                "method": method,
                "K": k,
                "states": len(items),
            }
            for metric in SUMMARY_METRICS:
                mean, low, high = _bootstrap_mean(
                    (r.values[metric] for r in items), rng=rng, samples=samples
                )
                row[metric] = mean
                row[f"{metric}_ci95_low"] = low
                row[f"{metric}_ci95_high"] = high
            output.append(row)
    return output


def _paired_rows(
    records: list[Record], *, samples: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    indexed = {(r.method, r.state_id, r.k): r for r in records}
    state_ids = sorted({r.state_id for r in records})
    output = []
    for baseline in ("base", "validity"):
        for method in METHODS:
            if method == baseline:
                continue
            for k in KS:
                for metric in PAIRED_METRICS:
                    differences = [
                        indexed[(method, state_id, k)].values[metric]
                        - indexed[(baseline, state_id, k)].values[metric]
                        for state_id in state_ids
                    ]
                    mean, low, high = _bootstrap_mean(
                        differences, rng=rng, samples=samples
                    )
                    output.append(
                        {
                            "baseline": baseline,
                            "method": method,
                            "K": k,
                            "metric": metric,
                            "states": len(differences),
                            "mean_difference": mean,
                            "ci95_low": low,
                            "ci95_high": high,
                            "strictly_positive": low > 0,
                            "strictly_negative": high < 0,
                        }
                    )
    return output


def _decomposition_rows(
    records: list[Record], *, samples: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    k16 = [r for r in records if r.k == 16]
    by_method = {
        method: {r.state_id: r for r in k16 if r.method == method} for method in METHODS
    }
    common_multimode = set.intersection(
        *[
            {
                state_id
                for state_id, record in items.items()
                if record.values["num_unique_valid_modes"] >= 2
            }
            for items in by_method.values()
        ]
    )
    output = []
    for method in METHODS:
        items = list(by_method[method].values())
        successes = [r for r in items if r.values["pass_at_k"] > 0]
        common = [by_method[method][state_id] for state_id in common_multimode]
        specs = (
            ("pass_at_k", items, "pass_at_k"),
            ("valid_mode_rate", items, "valid_mode_rate"),
            ("predictive_diversity_recovery", items, "predictive_diversity_recovery"),
            ("coverage_given_success", successes, "exact_coverage"),
            (
                "relative_pairwise_separation_common_multimode",
                common,
                "generated_to_available_separation",
            ),
        )
        row: dict[str, Any] = {
            "method": method,
            "states": len(items),
            "successful_states": len(successes),
            "common_multimode_states": len(common),
        }
        for output_metric, source, source_metric in specs:
            mean, low, high = _bootstrap_mean(
                (r.values[source_metric] for r in source), rng=rng, samples=samples
            )
            row[output_metric] = mean
            row[f"{output_metric}_ci95_low"] = low
            row[f"{output_metric}_ci95_high"] = high
        output.append(row)
    return output


def _separation_association_rows(
    records: list[Record], *, samples: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    output = []
    for method in METHODS:
        for k in KS:
            for mode_count in MODE_COUNTS:
                items = [
                    record
                    for record in records
                    if record.method == method
                    and record.k == k
                    and record.mode_count == mode_count
                ]
                x = np.asarray([record.separation for record in items], dtype=float)
                y = np.asarray(
                    [
                        record.values["predictive_diversity_recovery"]
                        for record in items
                    ],
                    dtype=float,
                )

                def slope(indices: np.ndarray) -> float:
                    sample_x = x[indices]
                    sample_y = y[indices]
                    variance = float(np.var(sample_x))
                    if variance == 0:
                        return 0.0
                    return float(np.cov(sample_x, sample_y, ddof=0)[0, 1] / variance)

                all_indices = np.arange(len(items))
                point = slope(all_indices)
                bootstrap_indices = rng.integers(
                    0, len(items), size=(samples, len(items))
                )
                estimates = np.asarray(
                    [slope(indices) for indices in bootstrap_indices], dtype=float
                )
                low, high = np.quantile(estimates, (0.025, 0.975))
                correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(y) else 0.0
                output.append(
                    {
                        "method": method,
                        "K": k,
                        "M": mode_count,
                        "states": len(items),
                        "pdr_change_per_0_1_separation": 0.1 * point,
                        "ci95_low": 0.1 * float(low),
                        "ci95_high": 0.1 * float(high),
                        "pearson_r": correlation,
                    }
                )
    return output


def _combined_ranking(
    summary_rows: list[dict[str, Any]], reference_summary: Path
) -> list[dict[str, Any]]:
    reference = json.loads(reference_summary.read_text(encoding="utf-8"))["overall"]
    rows = [
        {
            "name": policy,
            "kind": "privileged_reference_policy",
            "pdr_at_4": float(values["mean"]),
            "ci95_low": float(values["ci95"][0]),
            "ci95_high": float(values["ci95"][1]),
        }
        for policy, values in reference.items()
    ]
    rows.extend(
        {
            "name": row["method"],
            "kind": "model",
            "pdr_at_4": row["predictive_diversity_recovery"],
            "ci95_low": row["predictive_diversity_recovery_ci95_low"],
            "ci95_high": row["predictive_diversity_recovery_ci95_high"],
        }
        for row in summary_rows
        if row["K"] == 4
    )
    return sorted(rows, key=lambda row: float(row["pdr_at_4"]), reverse=True)


def _style_axis(axis: Any) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(color="#CBD5E1", alpha=0.62, linewidth=0.7)


def _bins(items: list[Record], count: int) -> list[list[Record]]:
    ordered = sorted(items, key=lambda item: item.separation)
    return [list(chunk) for chunk in np.array_split(ordered, count) if len(chunk)]


def _plot_separation(
    records: list[Record],
    *,
    k: int,
    output: Path,
    samples: int,
    rng: np.random.Generator,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6), sharex=True, sharey=True)
    for axis, mode_count in zip(axes.flat, MODE_COUNTS, strict=True):
        state_x = sorted(
            {
                r.separation
                for r in records
                if r.method == "base" and r.k == k and r.mode_count == mode_count
            }
        )
        axis.vlines(
            state_x,
            ymin=-0.004,
            ymax=0.003,
            color="#CBD5E1",
            linewidth=0.6,
            clip_on=False,
        )
        for method in METHODS:
            items = [
                r
                for r in records
                if r.method == method and r.k == k and r.mode_count == mode_count
            ]
            chunks = _bins(items, 6)
            xs = [float(np.mean([r.separation for r in chunk])) for chunk in chunks]
            intervals = [
                _bootstrap_mean(
                    (r.values["predictive_diversity_recovery"] for r in chunk),
                    rng=rng,
                    samples=samples,
                )
                for chunk in chunks
            ]
            means = [item[0] for item in intervals]
            lows = [item[1] for item in intervals]
            highs = [item[2] for item in intervals]
            axis.fill_between(xs, lows, highs, color=COLORS[method], alpha=0.10)
            axis.plot(
                xs,
                means,
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=1.8,
                markersize=4.2,
                label=LABELS[method],
            )
        axis.set_title(f"M = {mode_count}", loc="left", fontweight="bold")
        axis.set_xlim(0.025, 0.505)
        axis.set_ylim(-0.01, 0.42)
        _style_axis(axis)
    for axis in axes[-1, :]:
        axis.set_xlabel("Available predictive separation")
    for axis in axes[:, 0]:
        axis.set_ylabel(f"Predictive Diversity Recovery@{k}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        f"Model diversity recovery across continuous separation (K={k})",
        x=0.075,
        y=0.995,
        ha="left",
        fontsize=15,
    )
    fig.text(
        0.075,
        0.018,
        "Points average eight adjacent states on the continuous axis; bands are state-bootstrap 95% confidence intervals.",
        fontsize=9.2,
        color="#475569",
    )
    fig.tight_layout(rect=(0.04, 0.055, 1, 0.88), h_pad=2.2, w_pad=2.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_scaling(summary: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    specs = (
        ("pass_at_k", "At least one valid hypothesis", "Pass@K"),
        (
            "predictive_diversity_recovery",
            "Predictive diversity recovered",
            "PDR@K",
        ),
    )
    for axis, (metric, title, ylabel) in zip(axes, specs, strict=True):
        for method in METHODS:
            rows = [row for row in summary if row["method"] == method]
            rows.sort(key=lambda row: int(row["K"]))
            xs = [int(row["K"]) for row in rows]
            ys = [float(row[metric]) for row in rows]
            lows = [float(row[f"{metric}_ci95_low"]) for row in rows]
            highs = [float(row[f"{metric}_ci95_high"]) for row in rows]
            axis.fill_between(xs, lows, highs, color=COLORS[method], alpha=0.10)
            axis.plot(
                xs,
                ys,
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=1.9,
                label=LABELS[method],
            )
        axis.set_title(title, loc="left")
        axis.set_xlabel("Generation budget K")
        axis.set_ylabel(ylabel)
        axis.set_xticks(KS)
        axis.set_ylim(0, 1.02 if metric == "pass_at_k" else 0.14)
        _style_axis(axis)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("More samples improve validity more than diversity", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_decomposition(rows: list[dict[str, Any]], output: Path) -> None:
    specs = (
        ("pass_at_k", "Pass@16", (0, 1.05)),
        ("valid_mode_rate", "Valid-output rate", (0, 0.72)),
        ("predictive_diversity_recovery", "PDR@16", (0, 0.14)),
        (
            "relative_pairwise_separation_common_multimode",
            "Relative pairwise separation\n(common 85 states)",
            (0, 1.15),
        ),
    )
    fig, axes = plt.subplots(1, 4, figsize=(13.4, 4.1))
    x = np.arange(len(METHODS))
    for axis, (metric, title, ylim) in zip(axes, specs, strict=True):
        means = [
            float(next(r for r in rows if r["method"] == m)[metric]) for m in METHODS
        ]
        lows = [
            float(next(r for r in rows if r["method"] == m)[f"{metric}_ci95_low"])
            for m in METHODS
        ]
        highs = [
            float(next(r for r in rows if r["method"] == m)[f"{metric}_ci95_high"])
            for m in METHODS
        ]
        axis.bar(x, means, color=[COLORS[m] for m in METHODS], width=0.68)
        axis.errorbar(
            x,
            means,
            yerr=[
                np.asarray(means) - np.asarray(lows),
                np.asarray(highs) - np.asarray(means),
            ],
            fmt="none",
            ecolor="#111827",
            capsize=3,
            linewidth=1,
        )
        axis.set_xticks(x, [LABELS[m] for m in METHODS], rotation=28, ha="right")
        axis.set_title(title, fontsize=10.5)
        axis.set_ylim(*ylim)
        _style_axis(axis)
    fig.suptitle(
        "Validity gains do not translate proportionally into predictive diversity",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_paired(paired: list[dict[str, Any]], output: Path) -> None:
    rows = [
        row
        for row in paired
        if row["baseline"] == "validity"
        and row["metric"] == "predictive_diversity_recovery"
    ]
    methods = ("base", "ips", "latent_ips")
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    for method in methods:
        items = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["K"]),
        )
        xs = [int(row["K"]) for row in items]
        ys = [float(row["mean_difference"]) for row in items]
        lows = [float(row["ci95_low"]) for row in items]
        highs = [float(row["ci95_high"]) for row in items]
        axis.fill_between(xs, lows, highs, color=COLORS[method], alpha=0.11)
        axis.plot(
            xs,
            ys,
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.9,
            label=f"{LABELS[method]} - Validity GRPO",
        )
    axis.axhline(0, color="#111827", linewidth=1, linestyle="--")
    axis.set_xticks(KS)
    axis.set_xlabel("Generation budget K")
    axis.set_ylabel("Paired difference in PDR")
    axis.set_title("No diversity method reliably beats Validity GRPO", loc="left")
    axis.legend(frameon=False, fontsize=9)
    _style_axis(axis)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("artifacts/causal_micro_lab_final_eval_v2/wandb_reports"),
    )
    parser.add_argument(
        "--reference-summary",
        type=Path,
        default=Path(
            "artifacts/causal_micro_lab_diversity_rankability/final_v2/summary.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/causal_micro_lab_final_eval_v2/comparison"),
    )
    parser.add_argument(
        "--figure-dir", type=Path, default=Path("docs/figures/causal_micro_lab_v2")
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()

    rng = np.random.default_rng(20260801)
    records = _read_records(args.reports_root)
    summary = _summary_rows(records, samples=args.bootstrap_samples, rng=rng)
    paired = _paired_rows(records, samples=args.bootstrap_samples, rng=rng)
    decomposition = _decomposition_rows(
        records, samples=args.bootstrap_samples, rng=rng
    )
    separation_association = _separation_association_rows(
        records, samples=args.bootstrap_samples, rng=rng
    )
    ranking = _combined_ranking(summary, args.reference_summary)

    _write_csv(args.output_dir / "headline_by_k.csv", summary)
    _write_csv(args.output_dir / "paired_differences.csv", paired)
    _write_csv(args.output_dir / "decomposition_k16.csv", decomposition)
    _write_csv(args.output_dir / "separation_association.csv", separation_association)
    _write_csv(args.output_dir / "combined_rank_k4.csv", ranking)
    _plot_separation(
        records,
        k=4,
        output=args.figure_dir / "pdr_vs_separation_k4.png",
        samples=args.bootstrap_samples,
        rng=rng,
    )
    _plot_separation(
        records,
        k=16,
        output=args.figure_dir / "pdr_vs_separation_k16.png",
        samples=args.bootstrap_samples,
        rng=rng,
    )
    _plot_scaling(summary, args.figure_dir / "validity_and_pdr_by_k.png")
    _plot_decomposition(
        decomposition, args.figure_dir / "validity_diversity_decomposition_k16.png"
    )
    _plot_paired(paired, args.figure_dir / "paired_pdr_vs_validity.png")

    print(f"validated_methods={len(METHODS)} states=192 rows={len(records)}")
    print(f"tables={args.output_dir}")
    print(f"figures={args.figure_dir}")


if __name__ == "__main__":
    main()
