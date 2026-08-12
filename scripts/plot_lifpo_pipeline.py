from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from scattered_discovery.plotting import (
    DOUBLE_COLUMN_WIDTH,
    configure_publication_style,
    save_publication_figure,
    science_colors,
)

configure_publication_style()
SCIENCE_COLORS = science_colors()
COLORS = {
    "text": "#111111",
    "line": "#666666",
    "muted": "#666666",
    "shared_fill": "#F2F2F2",
    "shared_edge": "#777777",
    "train_fill": "#E7F0F8",
    "train_edge": SCIENCE_COLORS[0],
    "frozen_fill": "#FBE9E7",
    "frozen_edge": SCIENCE_COLORS[3],
    "method_fill": "#FFF1DD",
    "method_edge": SCIENCE_COLORS[2],
    "score_fill": "#FFF1DD",
    "score_edge": SCIENCE_COLORS[2],
}


def add_box(
    axis: plt.Axes,
    *,
    center: tuple[float, float],
    width: float,
    height: float,
    text: str,
    kind: str = "shared",
    fontsize: float = 10.5,
) -> FancyBboxPatch:
    fills = {
        "shared": COLORS["shared_fill"],
        "train": COLORS["train_fill"],
        "frozen": COLORS["frozen_fill"],
        "method": COLORS["method_fill"],
        "score": COLORS["score_fill"],
    }
    edges = {
        "shared": COLORS["shared_edge"],
        "train": COLORS["train_edge"],
        "frozen": COLORS["frozen_edge"],
        "method": COLORS["method_edge"],
        "score": COLORS["score_edge"],
    }
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.6,
        edgecolor=edges[kind],
        facecolor=fills[kind],
        zorder=3,
    )
    axis.add_patch(patch)
    axis.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color=COLORS["text"],
        fontsize=fontsize,
        linespacing=1.18,
        zorder=4,
    )
    return patch


def add_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str | None = None,
    dashed: bool = False,
    connection: str = "arc3",
    linewidth: float = 1.6,
    zorder: int = 2,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=linewidth,
            linestyle=(0, (4, 3)) if dashed else "-",
            color=color or COLORS["line"],
            connectionstyle=connection,
            shrinkA=2,
            shrinkB=2,
            zorder=zorder,
        )
    )


def add_phase(axis: plt.Axes, x: float, y: float, label: str) -> None:
    axis.text(
        x,
        y,
        label.upper(),
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=COLORS["muted"],
    )


def build_figure() -> plt.Figure:
    fig, axis = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 3.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    fig.patch.set_facecolor("white")
    axis.set_facecolor("white")

    add_phase(axis, 0.25, 0.955, "1. Generate a rollout group")
    add_phase(axis, 0.76, 0.955, "2. Construct method credit")
    add_phase(axis, 0.57, 0.355, "3. Shared GRPO update")

    add_box(
        axis,
        center=(0.065, 0.75),
        width=0.09,
        height=0.09,
        text="Evidence\nstate $x$",
    )
    add_box(
        axis,
        center=(0.195, 0.75),
        width=0.13,
        height=0.11,
        text="Neutral labels\n$z_1,\\ldots,z_G$",
        kind="method",
    )
    add_box(
        axis,
        center=(0.355, 0.75),
        width=0.13,
        height=0.12,
        text="Policy model\n$\\pi_\\theta$",
        kind="train",
        fontsize=11.5,
    )
    add_box(
        axis,
        center=(0.515, 0.75),
        width=0.14,
        height=0.12,
        text="$G$ hypotheses\n$y_1,\\ldots,y_G$",
    )

    add_arrow(axis, (0.112, 0.75), (0.128, 0.75))
    add_arrow(
        axis,
        (0.26, 0.75),
        (0.288, 0.75),
        color=COLORS["method_edge"],
    )
    add_arrow(axis, (0.42, 0.75), (0.445, 0.75))

    add_box(
        axis,
        center=(0.695, 0.84),
        width=0.20,
        height=0.12,
        text="Verifier +\nlength shaping\n$v_i,\\ r_i^{\\mathrm{base}}$",
        fontsize=9.0,
    )
    add_box(
        axis,
        center=(0.695, 0.66),
        width=0.20,
        height=0.12,
        text="Semantic multiplicity\n$n_i/G\\;\\rightarrow\\;b_i\\in[0,0.5]$",
        kind="method",
        fontsize=9.0,
    )
    add_box(
        axis,
        center=(0.695, 0.47),
        width=0.20,
        height=0.12,
        text="Counterfactual scoring\n$z_i$ vs. $z_i^-$\n$\\mu_i\\in[-1,1]$",
        kind="method",
        fontsize=9.0,
    )

    add_arrow(axis, (0.585, 0.775), (0.603, 0.825))
    add_arrow(
        axis,
        (0.585, 0.735),
        (0.603, 0.675),
        color=COLORS["method_edge"],
    )
    add_arrow(
        axis,
        (0.535, 0.688),
        (0.62, 0.525),
        color=COLORS["method_edge"],
        connection="arc3,rad=0.12",
    )
    add_arrow(
        axis,
        (0.39, 0.69),
        (0.62, 0.49),
        color=COLORS["method_edge"],
        connection="arc3,rad=-0.10",
    )

    add_box(
        axis,
        center=(0.905, 0.66),
        width=0.17,
        height=0.13,
        text=("Shaped score\n$s_i=r_i^{\\mathrm{base}}$\n$+b_i+0.5\\mu_i$"),
        kind="score",
        fontsize=8.8,
    )
    add_arrow(axis, (0.785, 0.84), (0.83, 0.715), connection="arc3,rad=0.08")
    add_arrow(
        axis,
        (0.785, 0.66),
        (0.823, 0.66),
        color=COLORS["method_edge"],
    )
    add_arrow(
        axis,
        (0.785, 0.47),
        (0.84, 0.605),
        color=COLORS["method_edge"],
        connection="arc3,rad=-0.08",
    )

    add_box(
        axis,
        center=(0.825, 0.25),
        width=0.18,
        height=0.105,
        text="Group\nnormalisation\n$s_i\\;\\rightarrow\\;A_i$",
        kind="score",
        fontsize=9.0,
    )
    add_box(
        axis,
        center=(0.585, 0.25),
        width=0.17,
        height=0.11,
        text="Clipped GRPO\nactor loss",
        kind="shared",
        fontsize=11,
    )
    add_box(
        axis,
        center=(0.35, 0.25),
        width=0.15,
        height=0.105,
        text="Reference model\n$\\pi_{\\mathrm{ref}}$",
        kind="frozen",
    )
    add_box(
        axis,
        center=(0.825, 0.105),
        width=0.15,
        height=0.075,
        text="Cap-hit token mask",
        kind="shared",
        fontsize=9.5,
    )

    add_arrow(axis, (0.905, 0.585), (0.85, 0.31), connection="arc3,rad=-0.08")
    add_arrow(
        axis,
        (0.745, 0.25),
        (0.672, 0.25),
        color=COLORS["score_edge"],
    )
    add_arrow(axis, (0.425, 0.25), (0.497, 0.25), color=COLORS["frozen_edge"])
    axis.text(
        0.46,
        0.275,
        "KL",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=COLORS["muted"],
    )
    add_arrow(axis, (0.78, 0.13), (0.67, 0.21), connection="arc3,rad=0.08")

    add_arrow(
        axis,
        (0.51, 0.22),
        (0.355, 0.685),
        color=COLORS["train_edge"],
        dashed=True,
        connection="arc3,rad=-0.42",
        linewidth=1.8,
        zorder=1,
    )
    axis.text(
        0.235,
        0.43,
        "policy update",
        ha="center",
        va="center",
        fontsize=9,
        color=COLORS["train_edge"],
        rotation=79,
    )

    legend_y = 0.035
    legend_items = (
        ("train", "trained policy"),
        ("frozen", "frozen reference"),
        ("method", "LIFPO"),
        ("shared", "shared GRPO / verifier"),
    )
    start_x = 0.16
    for offset, (kind, label) in enumerate(legend_items):
        x = start_x + offset * 0.205
        fills = {
            "train": COLORS["train_fill"],
            "frozen": COLORS["frozen_fill"],
            "method": COLORS["method_fill"],
            "shared": COLORS["shared_fill"],
        }
        edges = {
            "train": COLORS["train_edge"],
            "frozen": COLORS["frozen_edge"],
            "method": COLORS["method_edge"],
            "shared": COLORS["shared_edge"],
        }
        swatch = FancyBboxPatch(
            (x, legend_y - 0.014),
            0.025,
            0.028,
            boxstyle="round,pad=0.003,rounding_size=0.004",
            facecolor=fills[kind],
            edgecolor=edges[kind],
            linewidth=1.2,
        )
        axis.add_patch(swatch)
        axis.text(
            x + 0.032,
            legend_y,
            label,
            ha="left",
            va="center",
            fontsize=8.7,
            color=COLORS["text"],
        )

    fig.subplots_adjust(left=0.02, right=0.985, top=0.985, bottom=0.02)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/thesis_figures/lifpo-overview.png"),
    )
    parser.add_argument("--pdf-output", type=Path)
    args = parser.parse_args()

    fig = build_figure()
    save_publication_figure(fig, args.output)


if __name__ == "__main__":
    main()
