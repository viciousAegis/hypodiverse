#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scattered_discovery.envs.causal_micro_lab.parser import parse_record_state


MODE_COUNTS = (4, 8, 12, 16)
COLORS = {
    4: "#2563EB",
    8: "#0F766E",
    12: "#B45309",
    16: "#7C3AED",
}


def _load_separation(path: Path) -> dict[int, list[float]]:
    values = {mode_count: [] for mode_count in MODE_COUNTS}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        state = parse_record_state(json.loads(line))
        if state.valid_mode_count in values:
            values[state.valid_mode_count].append(state.mean_separation)
    return {mode_count: sorted(items) for mode_count, items in values.items()}


def _largest_gap(values: list[float]) -> float:
    return max(
        (right - left for left, right in zip(values, values[1:], strict=False)),
        default=0.0,
    )


def build_plot(
    *,
    characterization_path: Path,
    eval_path: Path,
    output_path: Path,
) -> None:
    population = _load_separation(characterization_path)
    selected = _load_separation(eval_path)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, (distribution_axis, coverage_axis) = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.2),
        gridspec_kw={"width_ratios": (1.12, 1.0)},
    )

    positions = np.arange(len(MODE_COUNTS))
    violins = distribution_axis.violinplot(
        [population[mode_count] for mode_count in MODE_COUNTS],
        positions=positions,
        orientation="horizontal",
        widths=0.72,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        bw_method=0.22,
    )
    for body in violins["bodies"]:
        body.set_facecolor("#CBD5E1")
        body.set_edgecolor("#64748B")
        body.set_alpha(0.72)
        body.set_linewidth(0.8)
    violins["cmedians"].set_color("#334155")
    violins["cmedians"].set_linewidth(1.2)

    for position, mode_count in zip(positions, MODE_COUNTS, strict=True):
        values = selected[mode_count]
        offsets = np.linspace(-0.18, 0.18, len(values))
        order = np.argsort(
            [((index * 29) % len(values), value) for index, value in enumerate(values)],
            axis=0,
        )[:, 0]
        jitter = offsets[order]
        distribution_axis.scatter(
            values,
            position + jitter,
            s=15,
            color=COLORS[mode_count],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.9,
            zorder=3,
        )

    distribution_axis.set_yticks(positions, [f"M = {value}" for value in MODE_COUNTS])
    distribution_axis.invert_yaxis()
    distribution_axis.set_xlim(0.0, 0.52)
    distribution_axis.set_xlabel("Predictive separation")
    distribution_axis.set_title(
        "Candidate density and frozen evaluation states",
        loc="left",
    )
    distribution_axis.grid(axis="x", color="#CBD5E1", alpha=0.65, linewidth=0.7)
    distribution_axis.scatter(
        [],
        [],
        s=22,
        color="#2563EB",
        label="48 selected states per M",
    )
    distribution_axis.fill_between(
        [],
        [],
        [],
        color="#CBD5E1",
        label="All 252 candidate states per M",
    )
    distribution_axis.legend(frameon=False, loc="lower right", fontsize=9)

    for mode_count in MODE_COUNTS:
        values = selected[mode_count]
        coverage_axis.plot(
            range(1, len(values) + 1),
            values,
            marker="o",
            markersize=3.2,
            linewidth=1.5,
            color=COLORS[mode_count],
            label=f"M={mode_count}; max gap {_largest_gap(values):.3f}",
        )
    coverage_axis.set_xlim(1, 48)
    coverage_axis.set_ylim(0.0, 0.52)
    coverage_axis.set_xlabel("States ordered by separation")
    coverage_axis.set_ylabel("Predictive separation")
    coverage_axis.set_title("Continuous coverage of the selected set", loc="left")
    coverage_axis.grid(color="#CBD5E1", alpha=0.65, linewidth=0.7)
    coverage_axis.legend(frameon=False, loc="upper left", fontsize=8.8)

    fig.suptitle(
        "The benchmark samples predictive separation as a continuous variable",
        x=0.07,
        y=1.01,
        ha="left",
        fontsize=15,
        fontweight="normal",
    )
    fig.text(
        0.07,
        -0.01,
        (
            "Separation is the mean fraction of unanswered experiments on which "
            "two valid hypotheses predict different Y outcomes."
        ),
        ha="left",
        fontsize=9.5,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96), w_pad=3.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--characterization",
        type=Path,
        default=Path(
            "artifacts/causal_micro_lab_environment_characterization/"
            "predictive_v2/states.jsonl"
        ),
    )
    parser.add_argument(
        "--eval-states",
        type=Path,
        default=Path("eval_sets/causal_micro_lab/final_v2/states.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/causal_micro_lab_continuous_separation.png"),
    )
    args = parser.parse_args()
    build_plot(
        characterization_path=args.characterization,
        eval_path=args.eval_states,
        output_path=args.output,
    )
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
