from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


METHODS = ("base", "validity", "ips", "latent_ips")
LABELS = {
    "base": "Base",
    "validity": "Validity GRPO",
    "ips": "IPS-GRPO",
    "latent_ips": "Latent IPS",
}
COLORS = {
    "base": "#777777",
    "validity": "#2563EB",
    "ips": "#E07A1F",
    "latent_ips": "#159A75",
}
MARKERS = {
    "base": "o",
    "validity": "s",
    "ips": "^",
    "latent_ips": "D",
}
KS = (4, 8, 12, 16)
MS = (4, 8, 12, 16)
SEPARATIONS = ("low", "medium", "high")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else 0.0


def _select(
    rows: list[dict[str, str]],
    *,
    method: str,
    k: int | None = None,
    m: int | None = None,
    separation: str | None = None,
) -> dict[str, str]:
    for row in rows:
        if row["method"] != method:
            continue
        if k is not None and int(row["K"]) != k:
            continue
        if m is not None and int(row["M"]) != m:
            continue
        if separation is not None and row["separation"] != separation:
            continue
        return row
    raise KeyError((method, k, m, separation))


def _percent(value: float) -> str:
    return f"{100 * value:.1f}"


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _style_axis(axis: Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", alpha=0.22, linewidth=0.8)
    axis.set_axisbelow(True)


def _annotate_bars(axis: Axes, *, percent: bool = False) -> None:
    for patch in axis.patches:
        value = patch.get_height()
        label = f"{100 * value:.1f}" if percent else f"{value:.2f}"
        axis.annotate(
            label,
            (patch.get_x() + patch.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _plot_headline(
    overall: list[dict[str, str]],
    output_dir: Path,
) -> None:
    metrics = (
        ("valid_mode_rate", "Valid output rate", True),
        (
            "num_unique_valid_modes_given_pass",
            "Modes recovered | success",
            False,
        ),
        (
            "exact_coverage_given_pass",
            "Available modes recovered | success",
            True,
        ),
        (
            "generated_mode_separation_given_pass",
            "Behavioral separation | success",
            False,
        ),
    )
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.7))
    for axis, (metric, title, percent) in zip(axes, metrics, strict=True):
        values = [
            _number(_select(overall, method=method, k=16), metric)
            for method in METHODS
        ]
        axis.bar(
            range(len(METHODS)),
            values,
            color=[COLORS[method] for method in METHODS],
            width=0.72,
        )
        axis.set_xticks(
            range(len(METHODS)),
            [LABELS[method] for method in METHODS],
            rotation=24,
            ha="right",
        )
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("Percent" if percent else "Value")
        if percent:
            axis.set_ylim(0, max(values) * 1.24)
            axis.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda value, _: f"{100 * value:.0f}")
            )
        else:
            axis.set_ylim(0, max(values) * 1.24)
        _annotate_bars(axis, percent=percent)
        _style_axis(axis)
    fig.suptitle("Validity and diversity at K=16", fontsize=14, y=1.03)
    fig.tight_layout()
    _save(fig, output_dir, "figure_1_headline_k16")


def _plot_k_scaling(
    overall: list[dict[str, str]],
    output_dir: Path,
) -> None:
    metrics = (
        ("pass_at_k", "Pass@K", True),
        (
            "num_unique_valid_modes_given_pass",
            "Modes recovered | success",
            False,
        ),
        (
            "exact_coverage_given_pass",
            "Available modes recovered | success",
            True,
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for axis, (metric, title, percent) in zip(axes, metrics, strict=True):
        for method in METHODS:
            values = [
                _number(_select(overall, method=method, k=k), metric)
                for k in KS
            ]
            axis.plot(
                KS,
                values,
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2,
                markersize=5,
                label=LABELS[method],
            )
        axis.set_xticks(KS)
        axis.set_xlabel("Generation budget K")
        axis.set_title(title, fontsize=10)
        if percent:
            axis.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda value, _: f"{100 * value:.0f}")
            )
        _style_axis(axis)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Scaling with inference budget", fontsize=14, y=1.03)
    fig.tight_layout()
    _save(fig, output_dir, "figure_2_scaling_with_k")


def _plot_m_scaling(
    by_k_m: list[dict[str, str]],
    output_dir: Path,
) -> None:
    metrics = (
        (
            "num_unique_valid_modes_given_pass",
            "Modes recovered | success",
            False,
        ),
        (
            "exact_coverage_given_pass",
            "Available modes recovered | success",
            True,
        ),
        ("pass_at_k", "Pass@16", True),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for axis, (metric, title, percent) in zip(axes, metrics, strict=True):
        for method in METHODS:
            values = [
                _number(_select(by_k_m, method=method, k=16, m=m), metric)
                for m in MS
            ]
            axis.plot(
                MS,
                values,
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2,
                markersize=5,
                label=LABELS[method],
            )
        axis.set_xticks(MS)
        axis.set_xlabel("Number of valid modes M")
        axis.set_title(title, fontsize=10)
        if percent:
            axis.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda value, _: f"{100 * value:.0f}")
            )
        _style_axis(axis)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Scaling with ambiguity at K=16", fontsize=14, y=1.03)
    fig.tight_layout()
    _save(fig, output_dir, "figure_3_scaling_with_m_k16")


def _plot_separation(
    by_separation: list[dict[str, str]],
    output_dir: Path,
) -> None:
    metrics = (
        (
            "num_unique_valid_modes_given_pass",
            "Modes recovered | success",
            False,
        ),
        (
            "exact_coverage_given_pass",
            "Available modes recovered | success",
            True,
        ),
        (
            "generated_mode_separation_given_pass",
            "Behavioral separation | success",
            False,
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    x = range(len(SEPARATIONS))
    for axis, (metric, title, percent) in zip(axes, metrics, strict=True):
        for method in METHODS:
            values = [
                _number(
                    _select(
                        by_separation,
                        method=method,
                        k=16,
                        separation=separation,
                    ),
                    metric,
                )
                for separation in SEPARATIONS
            ]
            axis.plot(
                x,
                values,
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2,
                markersize=5,
                label=LABELS[method],
            )
        axis.set_xticks(x, [item.title() for item in SEPARATIONS])
        axis.set_xlabel("Available-mode separation")
        axis.set_title(title, fontsize=10)
        if percent:
            axis.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda value, _: f"{100 * value:.0f}")
            )
        _style_axis(axis)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Performance by state separation at K=16", fontsize=14, y=1.03)
    fig.tight_layout()
    _save(fig, output_dir, "figure_4_by_separation_k16")


def _plot_generation(
    generation: list[dict[str, str]],
    output_dir: Path,
) -> None:
    metrics = (
        ("parse_valid", "Parse valid"),
        ("evidence_consistent", "Evidence valid"),
        ("length_cap_hit", "Length cap hit"),
    )
    width = 0.19
    x = range(len(metrics))
    fig, axis = plt.subplots(figsize=(8.5, 4.2))
    for index, method in enumerate(METHODS):
        row = next(item for item in generation if item["method"] == method)
        values = [_number(row, metric) for metric, _ in metrics]
        offsets = [value + (index - 1.5) * width for value in x]
        axis.bar(
            offsets,
            values,
            width=width,
            color=COLORS[method],
            label=LABELS[method],
        )
    axis.set_xticks(x, [label for _, label in metrics])
    axis.set_ylabel("Percent")
    axis.set_ylim(0, 1.08)
    axis.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{100 * value:.0f}")
    )
    axis.legend(frameon=False, ncol=4, loc="upper center")
    axis.set_title("Generation outcomes")
    _style_axis(axis)
    fig.tight_layout()
    _save(fig, output_dir, "figure_5_generation_diagnostics")


def _plot_frontier(
    overall: list[dict[str, str]],
    output_dir: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(6.2, 4.7))
    for method in METHODS:
        row = _select(overall, method=method, k=16)
        x = _number(row, "valid_mode_rate")
        y = _number(row, "num_unique_valid_modes_given_pass")
        separation = _number(row, "generated_mode_separation_given_pass")
        axis.scatter(
            x,
            y,
            s=1800 * separation,
            color=COLORS[method],
            marker=MARKERS[method],
            edgecolor="white",
            linewidth=1,
            zorder=3,
        )
        axis.annotate(
            LABELS[method],
            (x, y),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=9,
        )
    axis.set_xlabel("Valid output rate")
    axis.set_ylabel("Modes recovered | success")
    axis.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{100 * value:.0f}%")
    )
    axis.set_title("Quality-diversity frontier at K=16\nMarker area: behavioral separation")
    _style_axis(axis)
    fig.tight_layout()
    _save(fig, output_dir, "figure_6_quality_diversity_frontier_k16")


def _checkpoint(display_name: str) -> str:
    match = re.search(r"_global_step_(\d+)_", display_name)
    return match.group(1) if match else "base"


def _write_reader_tables(
    *,
    overall: list[dict[str, str]],
    by_k_m: list[dict[str, str]],
    by_separation: list[dict[str, str]],
    generation: list[dict[str, str]],
    metadata: list[dict[str, str]],
    output_dir: Path,
) -> None:
    meta = {row["method"]: row for row in metadata}
    gen = {row["method"]: row for row in generation}
    headline = []
    for method in METHODS:
        row = _select(overall, method=method, k=16)
        headline.append(
            {
                "method": LABELS[method],
                "checkpoint_step": _checkpoint(meta[method]["display_name"]),
                "primary_token_budget": 4096 if method == "base" else 6000,
                "valid_output_rate_pct": _percent(
                    _number(row, "valid_mode_rate")
                ),
                "pass_at_16_pct": _percent(_number(row, "pass_at_k")),
                "modes_recovered_given_success": (
                    f"{_number(row, 'num_unique_valid_modes_given_pass'):.3f}"
                ),
                "available_modes_recovered_given_success_pct": _percent(
                    _number(row, "exact_coverage_given_pass")
                ),
                "family_coverage_given_success_pct": _percent(
                    _number(row, "family_coverage_given_pass")
                ),
                "effective_mode_count_given_success": (
                    f"{_number(row, 'effective_mode_count_given_pass'):.3f}"
                ),
                "dominant_mode_mass_given_success_pct": _percent(
                    _number(row, "dominant_mode_mass_given_pass")
                ),
                "generated_mode_separation_given_success": (
                    f"{_number(row, 'generated_mode_separation_given_pass'):.3f}"
                ),
                "parse_valid_pct": _percent(_number(gen[method], "parse_valid")),
                "length_cap_hit_pct": _percent(
                    _number(gen[method], "length_cap_hit")
                ),
            }
        )
    _write_csv(output_dir / "table_1_headline_k16.csv", headline)

    k_rows = []
    for method in METHODS:
        for k in KS:
            row = _select(overall, method=method, k=k)
            k_rows.append(
                {
                    "method": LABELS[method],
                    "K": k,
                    "pass_at_k_pct": _percent(_number(row, "pass_at_k")),
                    "modes_recovered_given_success": (
                        f"{_number(row, 'num_unique_valid_modes_given_pass'):.3f}"
                    ),
                    "available_modes_recovered_given_success_pct": _percent(
                        _number(row, "exact_coverage_given_pass")
                    ),
                }
            )
    _write_csv(output_dir / "table_2_scaling_with_k.csv", k_rows)

    m_rows = []
    for method in METHODS:
        for m in MS:
            row = _select(by_k_m, method=method, k=16, m=m)
            m_rows.append(
                {
                    "method": LABELS[method],
                    "M": m,
                    "support_states": int(_number(row, "states")),
                    "pass_at_16_pct": _percent(_number(row, "pass_at_k")),
                    "modes_recovered_given_success": (
                        f"{_number(row, 'num_unique_valid_modes_given_pass'):.3f}"
                    ),
                    "available_modes_recovered_given_success_pct": _percent(
                        _number(row, "exact_coverage_given_pass")
                    ),
                }
            )
    _write_csv(output_dir / "table_3_scaling_with_m_k16.csv", m_rows)

    separation_rows = []
    for method in METHODS:
        for separation in SEPARATIONS:
            row = _select(
                by_separation,
                method=method,
                k=16,
                separation=separation,
            )
            separation_rows.append(
                {
                    "method": LABELS[method],
                    "separation": separation,
                    "support_states": int(_number(row, "states")),
                    "pass_at_16_pct": _percent(_number(row, "pass_at_k")),
                    "modes_recovered_given_success": (
                        f"{_number(row, 'num_unique_valid_modes_given_pass'):.3f}"
                    ),
                    "available_modes_recovered_given_success_pct": _percent(
                        _number(row, "exact_coverage_given_pass")
                    ),
                    "generated_mode_separation_given_success": (
                        f"{_number(row, 'generated_mode_separation_given_pass'):.3f}"
                    ),
                }
            )
    _write_csv(
        output_dir / "table_4_by_separation_k16.csv",
        separation_rows,
    )


def _write_notes(output_dir: Path) -> None:
    text = """# Causal Micro-Lab comparison figures

## Recommended main-text presentation

1. Figure 1 and Table 1: headline validity and diversity at K=16.
2. Figure 2: whether additional generations rescue failures or discover modes.
3. Figure 4: the key separation analysis for the latent method.

Figure 3 belongs in the main text if scaling with M is a central research
question; otherwise place it in the appendix. Figure 5 is diagnostic and
belongs in the appendix. Figure 6 is a compact overview suitable for the
discussion, but it should not replace the decomposed primary metrics.

## Interpretation rules

- Report pass@K over all states.
- Report mode count and exact coverage conditional on at least one valid output.
- Use generated behavioral separation to distinguish genuinely different
  mechanisms from merely distinct mode IDs.
- Do not interpret lower duplicity alone as better diversity: methods producing
  fewer valid samples have fewer opportunities to duplicate.

## Current comparison caveats

- Base used a 4,096-token primary budget; trained checkpoints used 6,000.
- Validity GRPO is checkpoint 55; IPS-GRPO and latent IPS are checkpoint 30.
- Current figures are descriptive point estimates. Add paired bootstrap
  confidence intervals from the stored per-state report CSVs before using them
  as final thesis figures. No model evaluation rerun is required.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="artifacts/causal_micro_lab_final_eval/comparison_latest",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/causal_micro_lab_final_eval/comparison_latest/reader",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overall = _read_csv(input_dir / "overall_by_k.csv")
    by_k_m = _read_csv(input_dir / "by_k_m.csv")
    by_separation = _read_csv(input_dir / "by_k_separation.csv")
    generation = _read_csv(input_dir / "generation_overall.csv")
    metadata = _read_csv(input_dir / "run_metadata.csv")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.facecolor": "white",
        }
    )
    _plot_headline(overall, output_dir)
    _plot_k_scaling(overall, output_dir)
    _plot_m_scaling(by_k_m, output_dir)
    _plot_separation(by_separation, output_dir)
    _plot_generation(generation, output_dir)
    _plot_frontier(overall, output_dir)
    _write_reader_tables(
        overall=overall,
        by_k_m=by_k_m,
        by_separation=by_separation,
        generation=generation,
        metadata=metadata,
        output_dir=output_dir,
    )
    _write_notes(output_dir)
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
