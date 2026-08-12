#!/usr/bin/env python3
"""Generate the reader-facing Boolean Causal Micro-Lab environment figure pack."""

from __future__ import annotations

import argparse
import csv
import math
from itertools import combinations, product
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter

from scattered_discovery.envs.causal_micro_lab.dsl import PARENTS, Hypothesis
from scattered_discovery.envs.causal_micro_lab.enumerate_hypotheses import (
    enumerate_hypotheses,
)
from scattered_discovery.envs.causal_micro_lab.eval import load_states
from scattered_discovery.envs.causal_micro_lab.planner import (
    available_experiment_ids,
    experiment_entropy,
)
from scattered_discovery.envs.causal_micro_lab.predictive_diversity import (
    mode_full_outcome_distance,
)
from scattered_discovery.envs.causal_micro_lab.signatures import (
    ModeTable,
    build_mode_table,
    mode_id_for_signature,
)
from scattered_discovery.envs.causal_micro_lab.simulator import prediction_signature
from scattered_discovery.envs.causal_micro_lab.state_generator import (
    EvidenceState,
    make_state,
)
from scattered_discovery.plotting import (
    DOUBLE_COLUMN_WIDTH,
    SINGLE_COLUMN_WIDTH,
    configure_publication_style,
    save_publication_figure,
    science_colors,
)


configure_publication_style()
SCIENCE_COLORS = science_colors()
COLORS = {
    "ink": "#111111",
    "muted": "#666666",
    "grid": "#D9D9D9",
    "blue": SCIENCE_COLORS[0],
    "green": SCIENCE_COLORS[1],
    "orange": SCIENCE_COLORS[2],
    "red": SCIENCE_COLORS[3],
    "purple": SCIENCE_COLORS[4],
    "teal": SCIENCE_COLORS[5],
    "gray": "#777777",
    "light": "#F2F2F2",
}
M_VALUES = (4, 8, 12, 16)
M_COLORS = {
    mode_count: SCIENCE_COLORS[index] for index, mode_count in enumerate(M_VALUES)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-states",
        type=Path,
        default=Path("eval_sets/causal_micro_lab/canonical_eval/states.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/figures/causal_micro_lab_intro"),
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=Path("artifacts/causal_micro_lab_environment_figures"),
    )
    return parser.parse_args()


def configure_plotting() -> None:
    configure_publication_style()


def save(fig: Any, path: Path) -> None:
    save_publication_figure(fig, path)


def arrow(
    axis: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["muted"],
    width: float = 1.4,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=width,
            color=color,
        )
    )


def card(
    axis: Any,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    edge: str,
    face: str = "white",
    radius: float = 0.02,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.5,
    )
    axis.add_patch(patch)
    return patch


def style_axis(axis: Any, *, grid_axis: str = "both") -> None:
    axis.grid(False)
    axis.grid(True, axis=grid_axis)


def full_outcome_distance_matrix(
    state: EvidenceState,
    table: ModeTable,
) -> tuple[tuple[str, ...], tuple[int, ...], np.ndarray]:
    query_ids = available_experiment_ids(state, mode_table=table)
    mode_ids = tuple(
        sorted(
            state.valid_mode_ids,
            key=lambda mode_id: tuple(
                table.modes_by_id[mode_id].signature[query_id] for query_id in query_ids
            ),
        )
    )
    matrix = np.zeros((len(mode_ids), len(mode_ids)), dtype=float)
    for left, right in combinations(range(len(mode_ids)), 2):
        distance = mode_full_outcome_distance(
            mode_ids[left],
            mode_ids[right],
            query_ids,
            mode_table=table,
        )
        matrix[left, right] = distance
        matrix[right, left] = distance
    return mode_ids, query_ids, matrix


def mean_full_outcome_separation(state: EvidenceState, table: ModeTable) -> float:
    _, _, matrix = full_outcome_distance_matrix(state, table)
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    return float(upper.mean()) if len(upper) else 0.0


def selected_edges(hypothesis: Hypothesis) -> set[tuple[str, str]]:
    return {
        (input_name, rule.target)
        for rule in hypothesis.rules
        for input_name in rule.inputs
    }


def draw_candidate_graph(axis: Any, hypothesis: Hypothesis) -> None:
    axis.set_xlim(-0.08, 1.08)
    axis.set_ylim(0.05, 0.95)
    axis.axis("off")
    positions = {
        "X1": (0.02, 0.78),
        "X2": (0.02, 0.50),
        "X3": (0.02, 0.22),
        "Z1": (0.37, 0.65),
        "Z2": (0.66, 0.42),
        "Y": (0.98, 0.53),
    }
    chosen = selected_edges(hypothesis)
    for target, parents in PARENTS.items():
        for source in parents:
            x1, y1 = positions[source]
            x2, y2 = positions[target]
            selected = (source, target) in chosen
            axis.plot(
                [x1, x2],
                [y1, y2],
                color=COLORS["green"] if selected else COLORS["grid"],
                linewidth=2.2 if selected else 0.8,
                alpha=1.0 if selected else 0.65,
                zorder=1,
            )
    for name, (x, y) in positions.items():
        color = (
            COLORS["blue"]
            if name.startswith("X")
            else COLORS["orange"]
            if name.startswith("Z")
            else COLORS["green"]
        )
        axis.add_patch(
            Circle((x, y), 0.055, facecolor="white", edgecolor=color, linewidth=2)
        )
        axis.text(x, y, name, ha="center", va="center", weight="bold", fontsize=9)


def plot_task_anatomy(
    outputs: tuple[Path, Path, Path],
    state: EvidenceState,
    table: ModeTable,
) -> None:
    hidden = table.modes_by_id[state.hidden_mode_id]

    fig, axis = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.45))
    draw_candidate_graph(axis, hidden.canonical)
    fig.tight_layout()
    save(fig, outputs[0])

    fig, axis = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.8))
    grid = np.zeros((8, 5))
    observed = set(state.observed_experiment_ids())
    for experiment_id in observed:
        grid[experiment_id // 5, experiment_id % 5] = 1
    cmap = ListedColormap(["#EDF1F5", SCIENCE_COLORS[2]])
    axis.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    axis.grid(False)
    axis.set_xticks(
        range(5),
        ["Observe", "do Z1=0", "do Z1=1", "do Z2=0", "do Z2=1"],
        rotation=42,
        ha="right",
        fontsize=7.5,
    )
    axis.set_yticks(
        range(8),
        ["".join(map(str, values)) for values in product((0, 1), repeat=3)],
        fontsize=8,
    )
    axis.set_ylabel("Input assignment X1 X2 X3")
    for row in range(8):
        for column in range(5):
            experiment_id = row * 5 + column
            if experiment_id in observed:
                axis.text(
                    column,
                    row,
                    "visible",
                    ha="center",
                    va="center",
                    fontsize=6.7,
                    color="white",
                    weight="bold",
                )
    fig.tight_layout()
    save(fig, outputs[1])

    fig, axis = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.5))
    axis.axis("off")
    card(axis, (0.04, 0.43), 0.92, 0.42, edge=COLORS["green"], face="#F0F8F2")
    axis.text(
        0.10,
        0.76,
        hidden.canonical.render_flat_rules(),
        family="monospace",
        fontsize=9.5,
        va="top",
        linespacing=1.55,
    )
    axis.text(
        0.50,
        0.30,
        "parse  ->  syntax  ->  evidence  ->  hypothesis",
        ha="center",
        family="monospace",
        fontsize=8.3,
        color=COLORS["muted"],
    )
    fig.tight_layout()
    save(fig, outputs[2])


def equivalent_program_pair(
    table: ModeTable,
) -> tuple[Hypothesis, Hypothesis, str]:
    target = min(
        (mode for mode in table.modes if mode.syntactic_count > 1),
        key=lambda mode: mode.mode_id,
    )
    programs: list[Hypothesis] = []
    for hypothesis in enumerate_hypotheses():
        signature = prediction_signature(hypothesis, table.experiments)
        if mode_id_for_signature(signature) == target.mode_id:
            programs.append(hypothesis)
            if len(programs) == 2:
                return programs[0], programs[1], target.mode_id
    raise AssertionError("expected at least one non-trivial semantic equivalence")


def plot_programs_to_modes(output: Path, table: ModeTable) -> None:
    first, second, mode_id = equivalent_program_pair(table)
    sample_experiments = (0, 4, 39)
    signature = table.modes_by_id[mode_id].signature
    fig, axis = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 3.25))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    card(axis, (0.01, 0.27), 0.15, 0.52, edge=COLORS["blue"], face="#F3F7FD")
    axis.text(0.085, 0.70, "Finite DSL", ha="center", weight="bold", fontsize=9)
    axis.text(
        0.085,
        0.56,
        "COPY  NOT\nAND  OR  XOR",
        ha="center",
        va="center",
        family="monospace",
        fontsize=7.5,
        linespacing=1.5,
    )
    axis.text(
        0.085,
        0.38,
        f"{table.all_hypotheses_count:,}\nvalid programs",
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
        color=COLORS["blue"],
    )
    arrow(axis, (0.17, 0.53), (0.20, 0.53))

    for y, label, hypothesis in (
        (0.56, "Program A", first),
        (0.25, "Program B", second),
    ):
        card(axis, (0.21, y), 0.27, 0.24, edge=COLORS["orange"], face="#FFF8F0")
        axis.text(0.23, y + 0.19, label, weight="bold", fontsize=8.5)
        axis.text(
            0.23,
            y + 0.14,
            hypothesis.render_flat_rules(),
            family="monospace",
            fontsize=7.2,
            va="top",
            linespacing=1.25,
        )
    arrow(axis, (0.49, 0.53), (0.53, 0.53))

    card(axis, (0.54, 0.27), 0.25, 0.52, edge=COLORS["teal"], face="#F0F9F8")
    axis.text(
        0.665,
        0.70,
        "Same intervention\nsignature",
        ha="center",
        weight="bold",
        fontsize=8.5,
    )
    for row, experiment_id in enumerate(sample_experiments):
        outcome = signature[experiment_id]
        y = 0.57 - row * 0.10
        axis.text(
            0.665,
            y,
            f"P{experiment_id:02d}  ->  {''.join(map(str, outcome))}",
            ha="center",
            family="monospace",
            fontsize=7.5,
        )
    axis.text(
        0.665,
        0.31,
        "identical on all 40 probes",
        ha="center",
        fontsize=7.5,
        color=COLORS["blue"],
        weight="bold",
    )
    arrow(axis, (0.80, 0.53), (0.83, 0.53))

    card(axis, (0.84, 0.35), 0.15, 0.36, edge=COLORS["purple"], face="#F6F1FF")
    axis.text(0.915, 0.60, "Semantic mode", ha="center", weight="bold", fontsize=8.5)
    axis.text(
        0.915,
        0.48,
        mode_id[:8] + "...",
        ha="center",
        family="monospace",
        fontsize=7.5,
        color=COLORS["purple"],
    )

    fig.tight_layout()
    save(fig, output)


def minimum_subset_separation(
    state: EvidenceState,
    table: ModeTable,
    returned_size: int,
) -> tuple[float, tuple[str, ...]]:
    mode_ids, _, matrix = full_outcome_distance_matrix(state, table)
    if len(mode_ids) <= returned_size:
        return subset_separation(mode_ids, mode_ids, matrix), mode_ids
    best_score = math.inf
    best_modes: tuple[str, ...] = ()
    for candidate in combinations(mode_ids, returned_size):
        score = subset_separation(candidate, mode_ids, matrix)
        if score < best_score - 1e-12 or (
            math.isclose(score, best_score, abs_tol=1e-12)
            and (not best_modes or candidate < best_modes)
        ):
            best_score = score
            best_modes = candidate
    return best_score, best_modes


def hypothesis_set_trajectory(
    state: EvidenceState,
    hidden_mode_id: str,
    *,
    maximize_diversity: bool,
    returned_size: int,
    steps: int,
    table: ModeTable,
) -> tuple[int, ...]:
    """Run one exact loop while holding the experiment planner fixed."""
    hidden = table.modes_by_id[hidden_mode_id]
    current = make_state(
        hidden_mode=hidden,
        evidence_ids=state.observed_experiment_ids(),
        mode_table=table,
    )
    counts = [current.valid_mode_count]
    for _ in range(steps):
        if current.valid_mode_count <= 1:
            counts.append(current.valid_mode_count)
            continue
        if maximize_diversity:
            _, selected_modes = maximum_subset_separation(
                current,
                table,
                min(returned_size, current.valid_mode_count),
            )
        else:
            _, selected_modes = minimum_subset_separation(
                current,
                table,
                returned_size,
            )
        experiment_id = max(
            available_experiment_ids(current, mode_table=table),
            key=lambda candidate: (
                experiment_entropy(
                    selected_modes,
                    candidate,
                    mode_table=table,
                ),
                -candidate,
            ),
        )
        current = make_state(
            hidden_mode=hidden,
            evidence_ids=tuple(
                sorted((*current.observed_experiment_ids(), experiment_id))
            ),
            mode_table=table,
        )
        counts.append(current.valid_mode_count)
    return tuple(counts)


def expected_hypothesis_set_trajectory(
    state: EvidenceState,
    *,
    maximize_diversity: bool,
    returned_size: int,
    steps: int,
    table: ModeTable,
) -> np.ndarray:
    trajectories = np.asarray(
        [
            hypothesis_set_trajectory(
                state,
                hidden_mode_id,
                maximize_diversity=maximize_diversity,
                returned_size=returned_size,
                steps=steps,
                table=table,
            )
            for hidden_mode_id in state.valid_mode_ids
        ],
        dtype=float,
    )
    return trajectories.mean(axis=0)


def select_hypothesis_set_example(
    states: list[EvidenceState],
    table: ModeTable,
    *,
    returned_size: int,
    steps: int,
) -> tuple[EvidenceState, np.ndarray, np.ndarray]:
    """Select an illustrative state and return its paired exact trajectories."""
    candidates = []
    for state in states:
        if state.valid_mode_count != 16:
            continue
        diverse = expected_hypothesis_set_trajectory(
            state,
            maximize_diversity=True,
            returned_size=returned_size,
            steps=steps,
            table=table,
        )
        clustered = expected_hypothesis_set_trajectory(
            state,
            maximize_diversity=False,
            returned_size=returned_size,
            steps=steps,
            table=table,
        )
        first_step_advantage = clustered[1] - diverse[1]
        cumulative_advantage = float(np.sum(clustered[1:] - diverse[1:]))
        identifies_within_two = math.isclose(diverse[2], 1.0, abs_tol=1e-12)
        candidates.append(
            (
                identifies_within_two,
                cumulative_advantage,
                first_step_advantage,
                state.state_id,
                state,
                diverse,
                clustered,
            )
        )
    *_, state, _, _ = max(candidates)
    diverse = np.asarray(
        hypothesis_set_trajectory(
            state,
            state.hidden_mode_id,
            maximize_diversity=True,
            returned_size=returned_size,
            steps=steps,
            table=table,
        ),
        dtype=float,
    )
    clustered = np.asarray(
        hypothesis_set_trajectory(
            state,
            state.hidden_mode_id,
            maximize_diversity=False,
            returned_size=returned_size,
            steps=steps,
            table=table,
        ),
        dtype=float,
    )
    return state, diverse, clustered


def plot_evidence_shrinks_space(
    output: Path,
    states: list[EvidenceState],
    table: ModeTable,
) -> None:
    returned_size = 4
    steps = 5
    state, diverse, clustered = select_hypothesis_set_example(
        states,
        table,
        returned_size=returned_size,
        steps=steps,
    )
    fig, axis = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.6))
    x = np.arange(steps + 1)
    axis.plot(
        x,
        diverse,
        color=COLORS["blue"],
        marker="o",
        markerfacecolor="white",
        markeredgecolor=COLORS["blue"],
        markersize=6,
        linewidth=1.5,
        label="Diverse sets",
        zorder=2,
    )
    axis.plot(
        x,
        clustered,
        color=COLORS["orange"],
        marker="s",
        markerfacecolor="white",
        markeredgecolor=COLORS["orange"],
        markersize=5.5,
        linewidth=1.5,
        label="Clustered sets",
        zorder=2,
    )
    axis.set_xticks(x)
    axis.set_xlabel("Experiments performed")
    axis.set_ylim(0, state.valid_mode_count + 1)
    axis.set_yticks(range(0, state.valid_mode_count + 1, 2))
    axis.set_ylabel(r"Compatible hypotheses, $M$")
    axis.legend(loc="best", frameon=False)
    style_axis(axis, grid_axis="y")
    fig.tight_layout()
    save(fig, output)


def plot_same_m_different_diversity(
    outputs: tuple[Path, Path],
    states: list[EvidenceState],
    table: ModeTable,
) -> None:
    returned_size = 4
    m16_states = [state for state in states if state.valid_mode_count == 16]
    scored = sorted(
        (
            *maximum_subset_separation(state, table, returned_size),
            state.state_id,
            state,
        )
        for state in m16_states
    )
    examples = (scored[0], scored[-1])
    for output, (_, oracle_modes, _, state) in zip(outputs, examples, strict=True):
        fig, axis = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.7))
        mode_ids, _, full_matrix = full_outcome_distance_matrix(state, table)
        index = {mode_id: position for position, mode_id in enumerate(mode_ids)}
        selected = np.asarray([index[mode_id] for mode_id in oracle_modes])
        matrix = full_matrix[np.ix_(selected, selected)]
        image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=0.90)
        axis.grid(False)
        labels = [f"H{i + 1}" for i in range(returned_size)]
        axis.set_xticks(range(returned_size), labels, fontsize=9)
        axis.set_yticks(range(returned_size), labels, fontsize=9)
        axis.tick_params(length=0)
        axis.set_xlabel("Hypothesis")
        axis.set_ylabel("Hypothesis")
        colorbar = fig.colorbar(image, ax=axis, fraction=0.05, pad=0.04)
        colorbar.set_label("Predictive distance")
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        fig.tight_layout()
        save(fig, output)


def subset_separation(
    selected_modes: tuple[str, ...],
    all_modes: tuple[str, ...],
    matrix: np.ndarray,
    *,
    denominator_size: int | None = None,
) -> float:
    target_size = denominator_size or len(selected_modes)
    if target_size < 2 or len(selected_modes) < 2:
        return 0.0
    index = {mode_id: position for position, mode_id in enumerate(all_modes)}
    pair_mass = sum(
        matrix[index[left], index[right]]
        for left, right in combinations(selected_modes, 2)
    )
    return float(pair_mass / math.comb(target_size, 2))


def maximum_subset_separation(
    state: EvidenceState,
    table: ModeTable,
    returned_size: int,
) -> tuple[float, tuple[str, ...]]:
    mode_ids, _, matrix = full_outcome_distance_matrix(state, table)
    if returned_size < 2 or returned_size > len(mode_ids):
        raise ValueError(
            f"returned_size must be in [2, {len(mode_ids)}], got {returned_size}"
        )
    best_score = -1.0
    best_modes: tuple[str, ...] = ()
    for candidate in combinations(mode_ids, returned_size):
        score = subset_separation(candidate, mode_ids, matrix)
        if score > best_score + 1e-12 or (
            math.isclose(score, best_score, abs_tol=1e-12)
            and (not best_modes or candidate < best_modes)
        ):
            best_score = score
            best_modes = candidate
    return best_score, best_modes


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    samples: int = 3000,
) -> tuple[float, float, float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    draws = values[indices].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(values.mean()), float(low), float(high)


def plot_continuous_coverage(
    outputs: tuple[Path, Path],
    states: list[EvidenceState],
    table: ModeTable,
) -> None:
    returned_size = 4
    fig, axis = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.6))
    rng = np.random.default_rng(20260802)
    values_by_m = [
        np.asarray(
            [
                maximum_subset_separation(state, table, returned_size)[0]
                for state in states
                if state.valid_mode_count == mode_count
            ]
        )
        for mode_count in M_VALUES
    ]
    violins = axis.violinplot(
        values_by_m,
        positions=np.arange(4),
        widths=0.72,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body, mode_count in zip(violins["bodies"], M_VALUES, strict=True):
        body.set_facecolor(M_COLORS[mode_count])
        body.set_edgecolor(M_COLORS[mode_count])
        body.set_alpha(0.18)
    violins["cmedians"].set_color(COLORS["ink"])
    violins["cmedians"].set_linewidth(2)
    for index, (mode_count, values) in enumerate(
        zip(M_VALUES, values_by_m, strict=True)
    ):
        jitter = rng.uniform(-0.23, 0.23, size=len(values))
        axis.scatter(
            index + jitter,
            values,
            s=15,
            color=M_COLORS[mode_count],
            alpha=0.38,
            linewidth=0,
        )
    axis.set_xticks(range(4), [f"M={mode_count}" for mode_count in M_VALUES])
    axis.set_ylabel(r"Diversity@4")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_ylim(0, 1)
    style_axis(axis, grid_axis="y")
    fig.tight_layout()
    save(fig, outputs[0])

    fig, axis = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.6))
    rng = np.random.default_rng(20260803)
    for mode_count in M_VALUES:
        group = [state for state in states if state.valid_mode_count == mode_count]
        x_values: list[int] = []
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for candidate_size in range(2, mode_count):
            values = np.asarray(
                [
                    maximum_subset_separation(state, table, candidate_size)[0]
                    for state in group
                ]
            )
            mean, low, high = bootstrap_mean_interval(values, rng=rng)
            x_values.append(candidate_size)
            means.append(mean)
            lows.append(low)
            highs.append(high)
        x = np.asarray(x_values)
        axis.plot(
            x,
            means,
            marker="o",
            markersize=4,
            linewidth=2,
            color=M_COLORS[mode_count],
            label=f"M={mode_count}",
        )
        axis.fill_between(x, lows, highs, color=M_COLORS[mode_count], alpha=0.13)
    axis.set_xticks(range(2, 16, 2))
    axis.set_xlim(1.5, 15.5)
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xlabel("Retained-set size K")
    axis.set_ylabel(r"Diversity@$K$")
    axis.legend(
        frameon=False,
        loc="upper right",
        ncol=2,
        fontsize=8.5,
    )
    style_axis(axis, grid_axis="both")
    fig.tight_layout()
    save(fig, outputs[1])


def best_quartet(
    state: EvidenceState,
    table: ModeTable,
) -> tuple[tuple[str, ...], float]:
    score, modes = maximum_subset_separation(state, table, 4)
    return modes, score


def plot_evaluation_pipeline(
    output: Path,
    states: list[EvidenceState],
    table: ModeTable,
) -> None:
    del states, table
    fig, axis = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 2.35))
    axis.set_xlim(0, 1)
    axis.set_ylim(0.28, 0.84)
    axis.axis("off")
    stages = (
        (0.01, 0.20, "1. Raw generations", "B = 8", COLORS["blue"]),
        (0.27, 0.18, "2. Keep valid outputs", "N = 6", COLORS["green"]),
        (0.52, 0.18, "3. Collapse repeats", "U = 4", COLORS["orange"]),
        (0.77, 0.22, "4. Retain best subset", "K = 4", COLORS["purple"]),
    )
    y = 0.34
    height = 0.44
    for x, width, title, count, color in stages:
        card(axis, (x, y), width, height, edge=color, face=f"{color}0D")
        axis.text(
            x + width / 2, y + 0.36, title, ha="center", weight="bold", fontsize=10
        )
        axis.text(
            x + width / 2,
            y + 0.29,
            count,
            ha="center",
            weight="bold",
            fontsize=12,
            color=color,
        )
    for left, right in ((0.21, 0.27), (0.45, 0.52), (0.70, 0.77)):
        arrow(axis, (left + 0.01, 0.56), (right - 0.01, 0.56))

    palette = {
        "A": COLORS["blue"],
        "B": COLORS["teal"],
        "C": COLORS["orange"],
        "D": COLORS["purple"],
    }
    raw = ("A", "invalid", "A", "B", "invalid", "C", "D", "D")
    for index, label in enumerate(raw):
        x = 0.035 + (index % 4) * 0.045
        yy = 0.53 - (index // 4) * 0.11
        if label == "invalid":
            axis.text(
                x, yy, "x", ha="center", va="center", color=COLORS["red"], weight="bold"
            )
        else:
            axis.add_patch(
                Circle((x, yy), 0.015, facecolor=palette[label], edgecolor="white")
            )
            axis.text(
                x,
                yy,
                label,
                ha="center",
                va="center",
                color="white",
                fontsize=7,
                weight="bold",
            )

    valid = ("A", "A", "B", "C", "D", "D")
    for index, label in enumerate(valid):
        x = 0.292 + (index % 3) * 0.055
        yy = 0.53 - (index // 3) * 0.11
        axis.add_patch(
            Circle((x, yy), 0.018, facecolor=palette[label], edgecolor="white")
        )
        axis.text(
            x,
            yy,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=7.5,
            weight="bold",
        )

    for index, label in enumerate(("A", "B", "C", "D")):
        x = 0.555 + (index % 2) * 0.09
        yy = 0.53 - (index // 2) * 0.11
        axis.add_patch(
            Circle((x, yy), 0.022, facecolor=palette[label], edgecolor="white")
        )
        axis.text(
            x,
            yy,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            weight="bold",
        )

    retained_positions = ((0.825, 0.58), (0.93, 0.58), (0.825, 0.45), (0.93, 0.45))
    for (x, yy), label in zip(retained_positions, ("A", "B", "C", "D"), strict=True):
        axis.add_patch(
            Circle((x, yy), 0.028, facecolor=palette[label], edgecolor="white")
        )
        axis.text(
            x,
            yy,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            weight="bold",
        )
    fig.tight_layout()
    save(fig, output)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_characterization_tables(
    table_dir: Path,
    table: ModeTable,
    states: list[EvidenceState],
) -> None:
    specification = [
        {"feature": "Exogenous variables", "value": 3, "definition": "X1, X2, X3"},
        {"feature": "Intermediate variables", "value": 2, "definition": "Z1, Z2"},
        {"feature": "Outcome variables", "value": 1, "definition": "Y"},
        {
            "feature": "Boolean operators",
            "value": 5,
            "definition": "COPY, NOT, AND, OR, XOR",
        },
        {
            "feature": "Input assignments",
            "value": 8,
            "definition": "All binary settings of X1, X2, X3",
        },
        {
            "feature": "Intervention regimes",
            "value": 5,
            "definition": "Observe or force Z1/Z2 to 0/1",
        },
        {
            "feature": "Experiments",
            "value": len(table.experiments),
            "definition": "8 assignments x 5 regimes",
        },
        {
            "feature": "Syntactic programs",
            "value": table.all_hypotheses_count,
            "definition": "All valid three-rule programs",
        },
        {
            "feature": "Semantic modes",
            "value": len(table.modes),
            "definition": "Distinct 40-experiment signatures",
        },
    ]
    write_csv(table_dir / "environment_specification.csv", specification)

    rows = []
    for mode_count in M_VALUES:
        group = [state for state in states if state.valid_mode_count == mode_count]
        scores = np.asarray(
            [maximum_subset_separation(state, table, 4)[0] for state in group]
        )
        evidence = np.asarray([state.evidence_size for state in group])
        rows.append(
            {
                "M": mode_count,
                "final_eval_states": len(group),
                "evidence_size_min": int(evidence.min()),
                "evidence_size_median": float(np.median(evidence)),
                "evidence_size_max": int(evidence.max()),
                "oracle_best4_diversity_min": float(scores.min()),
                "oracle_best4_diversity_p05": float(np.quantile(scores, 0.05)),
                "oracle_best4_diversity_median": float(np.median(scores)),
                "oracle_best4_diversity_p95": float(np.quantile(scores, 0.95)),
                "oracle_best4_diversity_max": float(scores.max()),
            }
        )
    write_csv(table_dir / "environment_characterization_by_M.csv", rows)

    oracle_rows = []
    for mode_count in M_VALUES:
        group = [state for state in states if state.valid_mode_count == mode_count]
        for returned_size in range(2, mode_count):
            scores = np.asarray(
                [
                    maximum_subset_separation(state, table, returned_size)[0]
                    for state in group
                ]
            )
            oracle_rows.append(
                {
                    "M": mode_count,
                    "K": returned_size,
                    "states": len(group),
                    "oracle_diversity_mean": float(scores.mean()),
                    "oracle_diversity_median": float(np.median(scores)),
                    "oracle_diversity_min": float(scores.min()),
                    "oracle_diversity_max": float(scores.max()),
                }
            )
    write_csv(table_dir / "oracle_diversity_by_K_M.csv", oracle_rows)


def write_readme(output_dir: Path, table_dir: Path) -> None:
    content = f"""# Boolean Causal Micro-Lab Environment Figures

All figures are generated from the exact hypothesis engine and the frozen
192-state held-out evaluation set.

Composite figures are emitted as standalone panel files and assembled with
LaTeX `subfigure` environments. Every panel is written as a 300 dpi PNG and a
cropped vector PDF. Plot titles and explanatory prose belong in LaTeX captions,
not inside the graphics.

Pairwise distance is disagreement in the complete `(Z1, Z2, Y)` outcome over
experiments not visible in the evidence. For `K > 1`, Diversity@K is the maximum
mean pairwise distance among any `K` modes in the candidate bank. The Oracle is
scored with the same Diversity@K metric over the full valid version space.

Supporting tables are written to `{table_dir}`:

- `environment_specification.csv`
- `environment_characterization_by_M.csv`
- `oracle_diversity_by_K_M.csv`
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_plotting()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    table = build_mode_table()
    eval_states = load_states(args.eval_states)
    if {state.valid_mode_count for state in eval_states} != set(M_VALUES):
        raise ValueError("final evaluation states do not contain the expected M values")
    m8_states = [state for state in eval_states if state.valid_mode_count == 8]
    m8_scores = {
        state.state_id: maximum_subset_separation(state, table, 4)[0]
        for state in m8_states
    }
    median_m8 = float(np.median(list(m8_scores.values())))
    task_state = min(
        m8_states,
        key=lambda state: (
            abs(m8_scores[state.state_id] - median_m8),
            state.state_id,
        ),
    )

    task_paths = tuple(
        args.output_dir / name
        for name in (
            "figure-1a-mechanism-space.png",
            "figure-1b-experiment-library.png",
            "figure-1c-exact-verifier.png",
        )
    )
    program_path = args.output_dir / "figure-2-programs-to-semantic-modes.png"
    evidence_path = args.output_dir / "figure-3a-version-space-trajectory.png"
    example_paths = tuple(
        args.output_dir / name
        for name in (
            "figure-4a-low-oracle-diversity.png",
            "figure-4b-high-oracle-diversity.png",
        )
    )
    landscape_paths = tuple(
        args.output_dir / name
        for name in (
            "figure-5a-oracle-diversity-distribution.png",
            "figure-5b-oracle-diversity-by-k-m.png",
        )
    )
    pipeline_path = args.output_dir / "figure-6-evaluation-pipeline.png"
    plot_task_anatomy(task_paths, task_state, table)
    plot_programs_to_modes(program_path, table)
    plot_evidence_shrinks_space(evidence_path, eval_states, table)
    plot_same_m_different_diversity(example_paths, eval_states, table)
    plot_continuous_coverage(landscape_paths, eval_states, table)
    plot_evaluation_pipeline(pipeline_path, eval_states, table)
    write_characterization_tables(args.table_dir, table, eval_states)
    write_readme(args.output_dir, args.table_dir)
    paths = (
        *task_paths,
        program_path,
        evidence_path,
        *example_paths,
        *landscape_paths,
        pipeline_path,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
