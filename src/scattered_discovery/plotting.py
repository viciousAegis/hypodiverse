"""Shared publication plotting conventions for thesis figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  # Registers the SciencePlots styles.


FIGURE_DPI = 300
SINGLE_COLUMN_WIDTH = 3.35
DOUBLE_COLUMN_WIDTH = 6.90


def configure_publication_style() -> None:
    """Apply one legible, LaTeX-independent SciencePlots configuration."""

    plt.style.use(["science", "grid", "no-latex"])
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def science_colors() -> tuple[str, ...]:
    """Return the active SciencePlots colour cycle."""

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return tuple(colors)


def method_colors() -> dict[str, str]:
    """Map method identity to a stable colour from the SciencePlots cycle."""

    colors = science_colors()
    return {
        "base": colors[0],
        "grpo": colors[1],
        "lifpo": colors[2],
    }


def save_publication_figure(
    figure: Any,
    output: Path,
    *,
    png: bool = True,
    pdf: bool = True,
) -> None:
    """Save a cropped 300 dpi PNG and/or vector PDF with the same stem."""

    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.with_suffix("")
    if png:
        figure.savefig(stem.with_suffix(".png"), dpi=FIGURE_DPI, bbox_inches="tight")
    if pdf:
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
