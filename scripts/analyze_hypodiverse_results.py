#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import random
from statistics import fmean
from typing import Any

from scattered_discovery.plotting import (
    configure_publication_style,
    method_colors,
    save_publication_figure,
)


HEADLINE_METRICS = (
    ("pass_at_k", "all_states", False),
    ("valid_mode_rate", "all_states", False),
    ("num_unique_valid_modes", "all_states", False),
    ("diversity_at_4", "diversity_defined", False),
)

METHOD_LABELS = {
    "base": "Base",
    "grpo": "GRPO",
    "lifpo": "LIFPO",
}

VALIDITY_METRICS = {
    "valid_outputs_per_state": "num_evidence_consistent",
    "valid_output_rate": "valid_mode_rate",
    "distinct_valid_hypotheses": "num_unique_valid_modes",
    "duplicate_valid_outputs": "duplicate_valid_modes",
}


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _interval(
    values: list[float],
    *,
    rng: random.Random,
    samples: int,
) -> tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    estimates = sorted(
        _mean([rng.choice(values) for _ in values]) for _ in range(samples)
    )
    return (
        _mean(values),
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_reports(specs: list[str]) -> dict[str, Path]:
    reports = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"report must be METHOD=PATH, got: {spec}")
        method, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        if not (path / "state_metrics.csv").exists():
            raise FileNotFoundError(f"missing report state metrics: {path}")
        reports[method] = path
    if len(reports) < 2:
        raise ValueError("at least two --report METHOD=PATH arguments are required")
    return reports


def _headline_rows(
    state_rows: dict[str, list[dict[str, str]]],
    *,
    rng: random.Random,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    output = []
    for method, rows in state_rows.items():
        by_k: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_k[int(row["K"])].append(row)
        for k, items in sorted(by_k.items()):
            result: dict[str, Any] = {
                "method": method,
                "K": k,
                "states": len(items),
                "successful_states": sum(
                    float(item["pass_at_k"]) > 0 for item in items
                ),
            }
            successes = [item for item in items if float(item["pass_at_k"]) > 0]
            diversity_defined = [
                item
                for item in items
                if float(item.get("diversity_at_4_defined", 0.0)) > 0
            ]
            for metric, conditioning, _lower_is_better in HEADLINE_METRICS:
                source = {
                    "all_states": items,
                    "successful_states": successes,
                    "diversity_defined": diversity_defined,
                }[conditioning]
                point, low, high = _interval(
                    [float(item[metric]) for item in source],
                    rng=rng,
                    samples=bootstrap_samples,
                )
                result[metric] = point
                result[f"{metric}_ci95_low"] = low
                result[f"{metric}_ci95_high"] = high
            output.append(result)
    return output


def _paired_rows(
    state_rows: dict[str, list[dict[str, str]]],
    *,
    baseline: str,
    rng: random.Random,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    indexed = {
        method: {(row["state_id"], int(row["K"])): row for row in rows}
        for method, rows in state_rows.items()
    }
    output = []
    for method in state_rows:
        if method == baseline:
            continue
        common = sorted(set(indexed[baseline]) & set(indexed[method]))
        for metric, conditioning, lower_is_better in HEADLINE_METRICS:
            keys = []
            for key in common:
                left, right = indexed[baseline][key], indexed[method][key]
                if conditioning == "successful_states" and not (
                    float(left["pass_at_k"]) > 0 and float(right["pass_at_k"]) > 0
                ):
                    continue
                if conditioning == "diversity_defined" and not (
                    float(left.get("diversity_at_4_defined", 0.0)) > 0
                    and float(right.get("diversity_at_4_defined", 0.0)) > 0
                ):
                    continue
                keys.append(key)
            by_k: dict[int, list[float]] = defaultdict(list)
            for key in keys:
                difference = float(indexed[method][key][metric]) - float(
                    indexed[baseline][key][metric]
                )
                by_k[key[1]].append(difference)
            for k, differences in sorted(by_k.items()):
                point, low, high = _interval(
                    differences,
                    rng=rng,
                    samples=bootstrap_samples,
                )
                output.append(
                    {
                        "baseline": baseline,
                        "method": method,
                        "K": k,
                        "metric": metric,
                        "conditioning": {
                            "all_states": "all_common_states",
                            "successful_states": "common_success",
                            "diversity_defined": "common_diversity_defined",
                        }[conditioning],
                        "lower_is_better": lower_is_better,
                        "states": len(differences),
                        "mean_difference": point,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return output


def _validity_rows(
    state_rows: dict[str, list[dict[str, str]]],
    *,
    rng: random.Random,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    """Put validity, uniqueness, and duplication on one state denominator."""

    output = []
    for method, rows in state_rows.items():
        by_k: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_k[int(row["K"])].append(row)
        for k, items in sorted(by_k.items()):
            result: dict[str, Any] = {
                "method": method,
                "K": k,
                "states": len(items),
                "successful_states": sum(
                    float(item["pass_at_k"]) > 0 for item in items
                ),
            }
            for output_metric, source_metric in VALIDITY_METRICS.items():
                point, low, high = _interval(
                    [float(item[source_metric]) for item in items],
                    rng=rng,
                    samples=bootstrap_samples,
                )
                result[output_metric] = point
                result[f"{output_metric}_ci95_low"] = low
                result[f"{output_metric}_ci95_high"] = high
            output.append(result)
    return output


def _plot_headline(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    configure_publication_style()

    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    ks = sorted({int(row["K"]) for row in rows})
    metrics = (
        ("pass_at_k", r"Pass@$B$", (0.0, 1.05)),
        ("valid_mode_rate", r"Validity@$B$", (0.0, 1.05)),
        ("num_unique_valid_modes", "Distinct valid hypotheses", None),
        ("diversity_at_4", r"Diversity@4", (0.0, 1.0)),
    )
    colors = method_colors()
    markers = ("o", "s", "^")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True)
    indexed = {(str(row["method"]), int(row["K"])): row for row in rows}
    for axis, (metric, title, limits) in zip(axes.flat, metrics, strict=True):
        for index, method in enumerate(methods):
            values = [float(indexed[method, k][metric]) for k in ks]
            lows = [float(indexed[method, k][f"{metric}_ci95_low"]) for k in ks]
            highs = [float(indexed[method, k][f"{metric}_ci95_high"]) for k in ks]
            axis.plot(
                ks,
                values,
                marker=markers[index % len(markers)],
                linewidth=1.8,
                color=colors.get(method),
                label=METHOD_LABELS.get(method, method),
            )
            axis.fill_between(
                ks,
                lows,
                highs,
                color=colors.get(method),
                alpha=0.10,
            )
        axis.set_ylabel(title)
        if limits:
            axis.set_ylim(*limits)
    for axis in axes[-1]:
        axis.set_xlabel(r"Raw generations, $B$")
        axis.set_xticks(ks)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(methods), frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_publication_figure(fig, output)


def run_analysis(
    *,
    reports: dict[str, Path],
    baseline: str,
    output_dir: Path,
    figure_dir: Path,
    bootstrap_samples: int,
    seed: int,
) -> None:
    if baseline not in reports:
        raise ValueError(f"baseline {baseline!r} is not among the supplied reports")
    state_rows = {
        method: _read_csv(path / "state_metrics.csv")
        for method, path in reports.items()
    }
    expected_keys = {(row["state_id"], int(row["K"])) for row in state_rows[baseline]}
    for method, rows in state_rows.items():
        keys = {(row["state_id"], int(row["K"])) for row in rows}
        if keys != expected_keys:
            raise RuntimeError(
                f"{method} does not contain the same state/K support as {baseline}"
            )
    rng = random.Random(seed)
    headline = _headline_rows(
        state_rows,
        rng=rng,
        bootstrap_samples=bootstrap_samples,
    )
    validity = _validity_rows(
        state_rows,
        rng=random.Random(seed + 1),
        bootstrap_samples=bootstrap_samples,
    )
    paired = _paired_rows(
        state_rows,
        baseline=baseline,
        rng=rng,
        bootstrap_samples=bootstrap_samples,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "headline_by_k.csv", headline)
    _write_csv(output_dir / "validity_uniqueness_duplication_by_k.csv", validity)
    _write_csv(output_dir / "paired_differences.csv", paired)
    _plot_headline(headline, figure_dir / "method_comparison_by_k.png")
    print(f"wrote={output_dir / 'headline_by_k.csv'}")
    print(f"wrote={output_dir / 'validity_uniqueness_duplication_by_k.csv'}")
    print(f"wrote={output_dir / 'paired_differences.csv'}")
    print(f"wrote={figure_dir / 'method_comparison_by_k.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="METHOD=PATH to a report directory; repeat for each model.",
    )
    parser.add_argument("--baseline", default="grpo")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/hypodiverse_comparison"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("artifacts/hypodiverse_comparison/figures"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    run_analysis(
        reports=_load_reports(args.report),
        baseline=args.baseline,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
