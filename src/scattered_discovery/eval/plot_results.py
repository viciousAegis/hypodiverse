from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


TASK_FIELDS = (
    "dispersion",
    "num_branches",
    "branch_depth",
    "distractors_per_node",
    "base_budget",
)


def _load_records(path: Path) -> list[dict[str, Any]]:
    episodes_path = path
    if path.is_dir():
        episodes_path = path / "episodes.jsonl"
    rows = []
    with episodes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_all_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(_load_records(path))
    return records


def _task_value(record: dict[str, Any], field: str) -> Any:
    task = record.get("task")
    if not isinstance(task, dict):
        return None
    if field == "dispersion":
        return task.get("dispersion")
    world = task.get("world")
    if isinstance(world, dict):
        return world.get(field)
    return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _episode_summary(records: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = _task_value(record, field)
        grouped["missing" if value is None else str(value)].append(record)

    rows = []
    for value, group in sorted(grouped.items(), key=lambda item: item[0]):
        scores = [record["score"] for record in group]
        rows.append(
            {
                "field": field,
                "value": value,
                "episodes": len(group),
                "validity": _mean(
                    [float(score.get("valid_unique_count", 0) > 0) for score in scores]
                ),
                "valid_unique_count": _mean(
                    [float(score.get("valid_unique_count", 0)) for score in scores]
                ),
                "reward": _mean([float(score.get("reward", 0.0)) for score in scores]),
                "recovery": _mean(
                    [
                        float(score.get("metrics", {}).get("recovery", 0.0))
                        for score in scores
                    ]
                ),
                "false_count": _mean(
                    [float(score.get("false_count", 0)) for score in scores]
                ),
                "unsupported_count": _mean(
                    [float(score.get("unsupported_count", 0)) for score in scores]
                ),
                "parse_failures": _mean(
                    [float(score.get("parse_failures", 0)) for score in scores]
                ),
                "invalid_actions": _mean(
                    [float(score.get("invalid_actions", 0)) for score in scores]
                ),
                "early_stop_consecutive_invalid": _mean(
                    [
                        float(
                            score.get("metrics", {}).get("early_stop_reason")
                            == "consecutive_invalid_actions"
                        )
                        for score in scores
                    ]
                ),
            }
        )
    return rows


def _spec_groups(records: list[dict[str, Any]]) -> dict[tuple[Any, Any], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record.get("env_type"), record.get("spec_index"))].append(record)
    return grouped


def _spec_summary(records: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in _spec_groups(records).values():
        value = _task_value(group[0], field)
        grouped["missing" if value is None else str(value)].append(group)

    rows = []
    for value, groups in sorted(grouped.items(), key=lambda item: item[0]):
        pass_at_k = []
        unique_coverages = []
        valid_rollout_rates = []
        target_counts = []
        unique_counts = []
        for group in groups:
            target_count = int(
                group[0].get("score", {}).get("metrics", {}).get("target_count", 0)
            )
            valid_keys = {
                key
                for record in group
                for key in record.get("score", {}).get("valid_keys", [])
            }
            valid_rollouts = sum(
                record.get("score", {}).get("valid_unique_count", 0) > 0
                for record in group
            )
            pass_at_k.append(float(valid_rollouts > 0))
            valid_rollout_rates.append(valid_rollouts / len(group))
            target_counts.append(target_count)
            unique_counts.append(len(valid_keys))
            unique_coverages.append(
                len(valid_keys) / target_count if target_count else 0.0
            )
        rows.append(
            {
                "field": field,
                "value": value,
                "specs": len(groups),
                "pass_at_k": _mean(pass_at_k),
                "valid_rollout_rate": _mean(valid_rollout_rates),
                "unique_target_coverage": _mean(unique_coverages),
                "unique_targets_found": _mean([float(v) for v in unique_counts]),
                "target_count": _mean([float(v) for v in target_counts]),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_bar(
    *,
    rows: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    labels = [row["value"] for row in rows]
    values = [float(row[metric]) for row in rows]
    fig_width = max(6.0, 0.8 * len(labels) + 2.0)
    _, axis = plt.subplots(figsize=(fig_width, 4.0))
    axis.bar(labels, values, color="#3366aa")
    axis.set_ylim(0, max(1.0, max(values, default=0.0) * 1.15))
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        axis.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _plot_all(
    *,
    episode_rows: dict[str, list[dict[str, Any]]],
    spec_rows: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> list[Path]:
    written = []
    for field, rows in episode_rows.items():
        for metric, ylabel in (
            ("validity", "valid rollout rate"),
            ("recovery", "mean recovery"),
            ("false_count", "mean false commits"),
            ("unsupported_count", "mean unsupported commits"),
            ("parse_failures", "mean parse failures"),
            ("early_stop_consecutive_invalid", "early-stop rate"),
        ):
            path = output_dir / f"{field}_{metric}.png"
            _plot_bar(
                rows=rows,
                metric=metric,
                ylabel=ylabel,
                title=f"{ylabel} by {field}",
                output=path,
            )
            written.append(path)

    for field, rows in spec_rows.items():
        for metric, ylabel in (
            ("pass_at_k", "spec pass@K"),
            ("unique_target_coverage", "unique target coverage"),
            ("valid_rollout_rate", "valid rollout rate"),
        ):
            path = output_dir / f"{field}_{metric}.png"
            _plot_bar(
                rows=rows,
                metric=metric,
                ylabel=ylabel,
                title=f"{ylabel} by {field}",
                output=path,
            )
            written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result",
        nargs="+",
        help="One or more result directories or episodes.jsonl paths.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Defaults to <result>/plots for directories.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only write CSV/JSON summaries; do not require matplotlib.",
    )
    args = parser.parse_args()

    result_paths = [Path(result) for result in args.result]
    records = _load_all_records(result_paths)
    default_base = (
        result_paths[0] if result_paths[0].is_dir() else result_paths[0].parent
    )
    if len(result_paths) > 1:
        default_base = default_base.parent / "merged_plots"
    output_dir = (
        Path(args.out)
        if args.out
        else (default_base if len(result_paths) > 1 else default_base / "plots")
    )

    episode_rows = {field: _episode_summary(records, field) for field in TASK_FIELDS}
    spec_rows = {field: _spec_summary(records, field) for field in TASK_FIELDS}
    flat_episode_rows = [row for rows in episode_rows.values() for row in rows]
    flat_spec_rows = [row for rows in spec_rows.values() for row in rows]

    _write_csv(output_dir / "episode_metrics_by_task.csv", flat_episode_rows)
    _write_csv(output_dir / "spec_metrics_by_task.csv", flat_spec_rows)
    (output_dir / "plot_summary.json").write_text(
        json.dumps(
            {
                "episodes": len(records),
                "inputs": [str(path) for path in result_paths],
                "episode_metrics_by_task": episode_rows,
                "spec_metrics_by_task": spec_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    plot_paths: list[Path] = []
    if not args.no_plots:
        try:
            plot_paths = _plot_all(
                episode_rows=episode_rows,
                spec_rows=spec_rows,
                output_dir=output_dir,
            )
        except ImportError as exc:
            raise SystemExit(
                "Plotting requires matplotlib. Install it in the active environment "
                "or rerun with --no-plots to write CSV/JSON only."
            ) from exc

    print(
        json.dumps(
            {
                "episodes": len(records),
                "inputs": [str(path) for path in result_paths],
                "output_dir": str(output_dir),
                "csv": [
                    str(output_dir / "episode_metrics_by_task.csv"),
                    str(output_dir / "spec_metrics_by_task.csv"),
                ],
                "plots": [str(path) for path in plot_paths],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
