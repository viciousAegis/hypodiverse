from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt

from plot_causal_micro_lab_wandb import (
    DEFAULT_ENTITY,
    DEFAULT_PROJECT,
    _load_env,
    _run_history,
)


SET_METRICS = (
    "states",
    "num_samples",
    "num_evidence_consistent",
    "pass_at_k",
    "valid_mode_rate",
    "exact_coverage",
    "budget_normalized_coverage",
    "family_coverage",
    "num_unique_valid_modes",
    "effective_mode_count",
    "mode_entropy",
    "dominant_mode_mass",
    "duplicity",
    "duplicate_rate",
    "generated_mode_separation",
    "generated_to_available_separation",
    "predictive_diversity_recovery",
)
SAMPLE_METRICS = (
    "parse_valid",
    "syntax_valid",
    "evidence_consistent",
    "length_cap_hit",
    "fallback_used",
    "initial_completion_tokens_mean",
    "total_completion_tokens_mean",
)
COLORS = ("#2563eb", "#dc2626", "#16a34a", "#7c3aed")


def _summary(run: dict[str, Any]) -> dict[str, Any]:
    value = run["summaryMetrics"]
    return json.loads(value) if isinstance(value, str) else dict(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _numeric(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _row(
    summary: dict[str, Any],
    *,
    prefix: str,
    labels: dict[str, Any],
    metrics: tuple[str, ...] = SET_METRICS,
) -> dict[str, Any]:
    row = dict(labels)
    for metric in metrics:
        value = _numeric(summary, f"{prefix}/{metric}")
        if value is not None:
            row[metric] = value
    pass_at_k = row.get("pass_at_k")
    if isinstance(pass_at_k, float) and pass_at_k > 0:
        for metric in (
            "exact_coverage",
            "budget_normalized_coverage",
            "family_coverage",
            "num_unique_valid_modes",
            "effective_mode_count",
            "dominant_mode_mass",
            "duplicity",
            "generated_mode_separation",
        ):
            if metric in row:
                row[f"{metric}_given_pass"] = float(row[metric]) / pass_at_k
    return row


def _extract_tables(
    summary: dict[str, Any],
    ks: tuple[int, ...],
    ms: tuple[int, ...],
) -> dict[str, list[dict[str, Any]]]:
    separation_labels = sorted(
        {
            match.group(1)
            for key in summary
            if (
                match := re.search(
                    r"/by_separation_bucket/([^/]+)/",
                    str(key),
                )
            )
        }
    ) or ["continuous"]
    overall = [_row(summary, prefix=f"set_summary_k{k}", labels={"K": k}) for k in ks]
    by_k_m = [
        _row(
            summary,
            prefix=f"set_summary_k{k}/by_M/{m}",
            labels={"K": k, "M": m},
        )
        for k in ks
        for m in ms
    ]
    by_k_separation = [
        _row(
            summary,
            prefix=f"set_summary_k{k}/by_separation_bucket/{bucket}",
            labels={"K": k, "separation_bucket": bucket},
        )
        for k in ks
        for bucket in separation_labels
    ]
    by_k_m_separation = [
        _row(
            summary,
            prefix=(f"set_summary_k{k}/by_M_and_separation_bucket/{m}/{bucket}"),
            labels={"K": k, "M": m, "separation_bucket": bucket},
        )
        for k in ks
        for m in ms
        for bucket in separation_labels
    ]
    by_k_family = [
        _row(
            summary,
            prefix=f"set_summary_k{k}/by_family_bucket/{bucket}",
            labels={"K": k, "family_bucket": bucket},
        )
        for k in ks
        for bucket in ("within_family", "mixed", "cross_family")
    ]
    generation_by_m = [
        _row(
            summary,
            prefix=f"eval_summary/by_M/{m}",
            labels={"M": m},
            metrics=SAMPLE_METRICS,
        )
        for m in ms
    ]
    primary_by_k_m = [
        {
            "K": row["K"],
            "M": row["M"],
            "support_states": row.get("states", 0.0),
            "pass_at_k": row.get("pass_at_k", 0.0),
            "predictive_diversity_recovery": row.get(
                "predictive_diversity_recovery",
                0.0,
            ),
            "modes_recovered_given_success": row.get(
                "num_unique_valid_modes_given_pass",
                0.0,
            ),
            "fraction_modes_recovered_given_success": row.get(
                "exact_coverage_given_pass",
                0.0,
            ),
        }
        for row in by_k_m
    ]
    primary_by_k_separation = [
        {
            "K": row["K"],
            "separation_bucket": row["separation_bucket"],
            "support_states": row.get("states", 0.0),
            "pass_at_k": row.get("pass_at_k", 0.0),
            "predictive_diversity_recovery": row.get(
                "predictive_diversity_recovery",
                0.0,
            ),
            "modes_recovered_given_success": row.get(
                "num_unique_valid_modes_given_pass",
                0.0,
            ),
            "fraction_modes_recovered_given_success": row.get(
                "exact_coverage_given_pass",
                0.0,
            ),
        }
        for row in by_k_separation
    ]
    return {
        "primary_metrics_by_k_m": primary_by_k_m,
        "primary_metrics_by_k_separation": primary_by_k_separation,
        "overall_by_k": overall,
        "metrics_by_k_m": by_k_m,
        "metrics_by_k_separation": by_k_separation,
        "metrics_by_k_m_separation": by_k_m_separation,
        "metrics_by_k_family": by_k_family,
        "generation_by_m": generation_by_m,
    }


def _series(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    group_key: str,
    group_value: Any,
) -> tuple[list[int], list[float]]:
    selected = sorted(
        (
            (int(row["K"]), float(row[metric]))
            for row in rows
            if row.get(group_key) == group_value and metric in row
        ),
        key=lambda item: item[0],
    )
    return [item[0] for item in selected], [item[1] for item in selected]


def _bootstrap_errors(
    bootstrap_rows: list[dict[str, str]] | None,
    *,
    slice_name: str,
    metric: str,
    xs: list[int],
    group_key: str,
    group_value: Any,
    ys: list[float],
) -> list[list[float]] | None:
    if not bootstrap_rows:
        return None
    intervals = {
        int(row["K"]): (float(row["ci95_low"]), float(row["ci95_high"]))
        for row in bootstrap_rows
        if row.get("slice") == slice_name
        and row.get("metric") == metric
        and row.get(group_key) == str(group_value)
    }
    if any(x not in intervals for x in xs):
        return None
    return [
        [max(0.0, y - intervals[x][0]) for x, y in zip(xs, ys, strict=True)],
        [max(0.0, intervals[x][1] - y) for x, y in zip(xs, ys, strict=True)],
    ]


def _save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_by_m(
    rows: list[dict[str, Any]],
    ms: tuple[int, ...],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for color, m in zip(COLORS, ms, strict=True):
        for axis, metric, title in (
            (axes[0], "exact_coverage", "Exact semantic-mode coverage"),
            (axes[1], "family_coverage", "Mechanism-family coverage"),
        ):
            xs, ys = _series(rows, metric=metric, group_key="M", group_value=m)
            axis.plot(xs, ys, marker="o", linewidth=2, color=color, label=f"M={m}")
            axis.set_title(title)
            axis.set_xlabel("Generation budget K")
            axis.set_ylabel("Coverage")
            axis.grid(alpha=0.25)
    for axis in axes:
        axis.relim()
        axis.autoscale_view()
        axis.set_ylim(bottom=0)
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Base Qwen3-4B: coverage versus generation budget")
    fig.tight_layout()
    _save_figure(fig, output_dir, "coverage_by_k_m")


def _plot_conditional_coverage(
    rows: list[dict[str, Any]],
    ms: tuple[int, ...],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for color, m in zip(COLORS, ms, strict=True):
        for axis, metric, title in (
            (
                axes[0],
                "exact_coverage_given_pass",
                "Exact coverage given success",
            ),
            (
                axes[1],
                "family_coverage_given_pass",
                "Family coverage given success",
            ),
        ):
            xs, ys = _series(rows, metric=metric, group_key="M", group_value=m)
            axis.plot(xs, ys, marker="o", linewidth=2, color=color, label=f"M={m}")
            axis.set_title(title)
            axis.set_xlabel("Generation budget K")
            axis.set_ylabel("Conditional coverage")
            axis.grid(alpha=0.25)
    for axis in axes:
        axis.relim()
        axis.autoscale_view()
        axis.set_ylim(bottom=0)
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Base Qwen3-4B: diversity among successful states")
    fig.tight_layout()
    _save_figure(fig, output_dir, "coverage_given_success_by_k_m")


def _plot_primary_metrics(
    rows: list[dict[str, Any]],
    ms: tuple[int, ...],
    output_dir: Path,
    bootstrap_rows: list[dict[str, str]] | None = None,
) -> None:
    panels = (
        ("pass_at_k", "Pass@K", "Probability"),
        (
            "predictive_diversity_recovery",
            "Predictive diversity recovery",
            "Oracle-normalized PDR@K",
        ),
        (
            "modes_recovered_given_success",
            "Modes recovered given success",
            "Distinct valid modes",
        ),
        (
            "fraction_modes_recovered_given_success",
            "Fraction recovered given success",
            "Fraction of M",
        ),
    )
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4), sharex=True)
    for axis, (metric, title, ylabel) in zip(axes, panels, strict=True):
        for color, m in zip(COLORS, ms, strict=True):
            xs, ys = _series(rows, metric=metric, group_key="M", group_value=m)
            yerr = _bootstrap_errors(
                bootstrap_rows,
                slice_name="by_k_m",
                metric=metric,
                xs=xs,
                group_key="M",
                group_value=m,
                ys=ys,
            )
            axis.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker="o",
                linewidth=2,
                capsize=3,
                color=color,
                label=f"M={m}",
            )
        axis.set_title(title)
        axis.set_xlabel("Generation budget K")
        axis.set_ylabel(ylabel)
        axis.set_xticks(sorted({int(row["K"]) for row in rows}))
        axis.grid(alpha=0.25)
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[3].set_ylim(0, 1)
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Base Qwen3-4B: validity and diversity")
    fig.tight_layout()
    _save_figure(fig, output_dir, "primary_validity_diversity_metrics")


def _plot_primary_metrics_by_separation(
    rows: list[dict[str, Any]],
    output_dir: Path,
    bootstrap_rows: list[dict[str, str]] | None = None,
) -> None:
    panels = (
        ("pass_at_k", "Pass@K", "Probability"),
        (
            "predictive_diversity_recovery",
            "Predictive diversity recovery",
            "Oracle-normalized PDR@K",
        ),
        (
            "modes_recovered_given_success",
            "Modes recovered given success",
            "Distinct valid modes",
        ),
        (
            "fraction_modes_recovered_given_success",
            "Fraction recovered given success",
            "Fraction of M",
        ),
    )
    buckets = ("low", "medium", "high")
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4), sharex=True)
    for axis, (metric, title, ylabel) in zip(axes, panels, strict=True):
        for color, bucket in zip(COLORS, buckets):
            xs, ys = _series(
                rows,
                metric=metric,
                group_key="separation_bucket",
                group_value=bucket,
            )
            yerr = _bootstrap_errors(
                bootstrap_rows,
                slice_name="by_k_separation",
                metric=metric,
                xs=xs,
                group_key="separation_bucket",
                group_value=bucket,
                ys=ys,
            )
            axis.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker="o",
                linewidth=2,
                capsize=3,
                color=color,
                label=bucket.title(),
            )
        axis.set_title(title)
        axis.set_xlabel("Generation budget K")
        axis.set_ylabel(ylabel)
        axis.set_xticks(sorted({int(row["K"]) for row in rows}))
        axis.grid(alpha=0.25)
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[3].set_ylim(0, 1)
    axes[0].legend(frameon=False)
    fig.suptitle("Base Qwen3-4B: validity and diversity by separation")
    fig.tight_layout()
    _save_figure(fig, output_dir, "primary_metrics_by_separation")


def _plot_diversity(
    rows: list[dict[str, Any]],
    ms: tuple[int, ...],
    output_dir: Path,
) -> None:
    panels = (
        ("num_unique_valid_modes", "Unique valid modes"),
        ("effective_mode_count", "Effective mode count"),
        ("dominant_mode_mass", "Dominant-mode mass"),
        ("duplicity", "Duplicity"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for axis, (metric, title) in zip(axes.flat, panels, strict=True):
        for color, m in zip(COLORS, ms, strict=True):
            xs, ys = _series(rows, metric=metric, group_key="M", group_value=m)
            axis.plot(xs, ys, marker="o", linewidth=2, color=color, label=f"M={m}")
        axis.set_title(title)
        axis.set_xlabel("Generation budget K")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle("Base Qwen3-4B: mode discovery and collapse")
    fig.tight_layout()
    _save_figure(fig, output_dir, "diversity_by_k_m")


def _plot_separation(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    panels = (
        ("exact_coverage", "Exact coverage"),
        ("effective_mode_count", "Effective mode count"),
        ("predictive_diversity_recovery", "Predictive diversity recovery"),
        ("duplicity", "Duplicity"),
    )
    buckets = ("low", "medium", "high")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for axis, (metric, title) in zip(axes.flat, panels, strict=True):
        for color, bucket in zip(COLORS, buckets):
            xs, ys = _series(
                rows,
                metric=metric,
                group_key="separation_bucket",
                group_value=bucket,
            )
            axis.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                color=color,
                label=bucket.title(),
            )
        axis.set_title(title)
        axis.set_xlabel("Generation budget K")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Base Qwen3-4B: performance by target-set separation")
    fig.tight_layout()
    _save_figure(fig, output_dir, "metrics_by_separation")


def _plot_family(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    panels = (
        ("exact_coverage", "Exact coverage"),
        ("family_coverage", "Mechanism-family coverage"),
        ("effective_mode_count", "Effective mode count"),
        ("duplicity", "Duplicity"),
    )
    buckets = ("within_family", "mixed", "cross_family")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for axis, (metric, title) in zip(axes.flat, panels, strict=True):
        for color, bucket in zip(COLORS, buckets):
            xs, ys = _series(
                rows,
                metric=metric,
                group_key="family_bucket",
                group_value=bucket,
            )
            axis.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                color=color,
                label=bucket.replace("_", " ").title(),
            )
        axis.set_title(title)
        axis.set_xlabel("Generation budget K")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Base Qwen3-4B: performance by target-set family composition")
    fig.tight_layout()
    _save_figure(fig, output_dir, "metrics_by_family_bucket")


def _plot_generation(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    ms = [int(row["M"]) for row in rows]
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for offset, metric, label, color in (
        (-width, "parse_valid", "Parse valid", "#2563eb"),
        (0.0, "evidence_consistent", "Evidence valid", "#16a34a"),
        (width, "length_cap_hit", "Initial cap hit", "#dc2626"),
    ):
        axes[0].bar(
            [m + offset for m in ms],
            [float(row.get(metric, 0.0)) for row in rows],
            width=width,
            label=label,
            color=color,
        )
    axes[0].set_xticks(ms)
    axes[0].set_xlabel("M")
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Validity and truncation")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        ms,
        [float(row.get("initial_completion_tokens_mean", 0.0)) for row in rows],
        color="#7c3aed",
    )
    axes[1].axhline(4096, color="#dc2626", linestyle="--", label="Token cap")
    axes[1].set_xticks(ms)
    axes[1].set_xlabel("M")
    axes[1].set_ylabel("Mean initial completion tokens")
    axes[1].set_title("Response length")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Base Qwen3-4B: generation behavior")
    fig.tight_layout()
    _save_figure(fig, output_dir, "generation_behavior_by_m")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--input-run-json")
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output-dir")
    parser.add_argument("--ks", default="4,8,12,16")
    parser.add_argument("--ms", default="4,8,12,16")
    parser.add_argument("--bootstrap-ci-csv")
    args = parser.parse_args()

    if args.input_run_json:
        run = json.loads(Path(args.input_run_json).read_text(encoding="utf-8"))
        run_id = args.run_id or str(run["name"])
    else:
        if not args.run_id:
            parser.error("--run-id is required without --input-run-json")
        _load_env(Path(args.env_file))
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            raise SystemExit("WANDB_API_KEY is not set.")
        run_id = args.run_id
        run = _run_history(
            api_key,
            entity=args.entity,
            project=args.project,
            run_name=run_id,
        )
    summary = _summary(run)
    output_dir = Path(
        args.output_dir or f"artifacts/causal_micro_lab_final_eval/wandb_{run_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ks = tuple(int(value) for value in args.ks.split(","))
    ms = tuple(int(value) for value in args.ms.split(","))
    tables = _extract_tables(summary, ks, ms)
    bootstrap_rows = None
    if args.bootstrap_ci_csv:
        with Path(args.bootstrap_ci_csv).open(
            encoding="utf-8",
            newline="",
        ) as handle:
            bootstrap_rows = list(csv.DictReader(handle))

    for name, rows in tables.items():
        _write_csv(output_dir / f"{name}.csv", rows)
    _write_csv(
        output_dir / "raw_summary_metrics.csv",
        [
            {"metric": key, "value": value}
            for key, value in sorted(summary.items())
            if isinstance(value, (bool, int, float))
        ],
    )
    metadata = {
        "run_id": run_id,
        "display_name": run["displayName"],
        "state": run["state"],
        "updated_at": run["updatedAt"],
        "wandb_url": (f"https://wandb.ai/{args.entity}/{args.project}/runs/{run_id}"),
    }
    _write_csv(output_dir / "run_metadata.csv", [metadata])
    _plot_by_m(tables["metrics_by_k_m"], ms, output_dir)
    _plot_conditional_coverage(tables["metrics_by_k_m"], ms, output_dir)
    _plot_primary_metrics(
        tables["primary_metrics_by_k_m"],
        ms,
        output_dir,
        bootstrap_rows,
    )
    separation_labels = {
        str(row["separation_bucket"]) for row in tables["metrics_by_k_separation"]
    }
    if len(separation_labels) > 1:
        _plot_primary_metrics_by_separation(
            tables["primary_metrics_by_k_separation"],
            output_dir,
            bootstrap_rows,
        )
    _plot_diversity(tables["metrics_by_k_m"], ms, output_dir)
    if len(separation_labels) > 1:
        _plot_separation(tables["metrics_by_k_separation"], output_dir)
    _plot_family(tables["metrics_by_k_family"], output_dir)
    _plot_generation(tables["generation_by_m"], output_dir)
    print(json.dumps(metadata, indent=2))
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
