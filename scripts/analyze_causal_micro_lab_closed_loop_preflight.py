#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any

from scattered_discovery.envs.causal_micro_lab.closed_loop_preflight import (
    REFERENCE_POLICIES,
    planner_headroom,
    run_reference_trajectory,
    summarize_reference_trajectories,
    trajectory_metrics,
)
from scattered_discovery.envs.causal_micro_lab.eval import load_states
from scattered_discovery.envs.causal_micro_lab.signatures import build_mode_table
from scattered_discovery.envs.causal_micro_lab.state_generator import (
    EvidenceState,
    find_states,
)
from scattered_discovery.envs.causal_micro_lab.tables import (
    split_mode_ids,
    state_rows,
    write_table,
)


DEFAULT_EXCLUSION_GLOBS = (
    "data/causal_micro_lab/*/states_*.jsonl",
    "eval_sets/causal_micro_lab/*/states.jsonl",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _excluded_hidden_modes(root: Path) -> tuple[set[str], list[str]]:
    hidden_modes: set[str] = set()
    paths = []
    for pattern in DEFAULT_EXCLUSION_GLOBS:
        for path in sorted(root.glob(pattern)):
            paths.append(str(path.relative_to(root)))
            hidden_modes.update(
                state.hidden_mode_id for state in load_states(str(path))
            )
    return hidden_modes, paths


def _select_across_headroom(
    candidates: list[EvidenceState],
    *,
    count: int,
    mode_table,
) -> tuple[list[EvidenceState], dict[str, float]]:
    ranked = sorted(
        (
            (planner_headroom(state, mode_table=mode_table), state)
            for state in candidates
        ),
        key=lambda item: (item[0], item[1].state_id),
    )
    if len(ranked) < count:
        raise RuntimeError(f"found {len(ranked)} candidates, need {count}")
    if count == 1:
        indices = [len(ranked) // 2]
    else:
        indices = [
            round(index * (len(ranked) - 1) / (count - 1)) for index in range(count)
        ]
    selected = [ranked[index][1] for index in indices]
    selected_headrooms = [ranked[index][0] for index in indices]
    return selected, {
        "minimum": min(selected_headrooms),
        "mean": sum(selected_headrooms) / len(selected_headrooms),
        "maximum": max(selected_headrooms),
    }


def build_states(
    *,
    output_dir: Path,
    initial_mode_count: int,
    state_count: int,
    candidate_multiplier: int,
    split_seed: int,
    generation_seed: int,
    beam_width: int,
    max_evidence: int,
    repo_root: Path,
) -> tuple[list[EvidenceState], dict[str, Any]]:
    table = build_mode_table()
    excluded, exclusion_paths = _excluded_hidden_modes(repo_root)
    test_modes = sorted(
        split_mode_ids(seed=split_seed, mode_table=table)["test"] - excluded
    )
    random.Random(generation_seed).shuffle(test_modes)
    candidate_target = state_count * candidate_multiplier
    candidates = []
    searched = 0
    for hidden_mode_id in test_modes:
        searched += 1
        found = find_states(
            hidden_mode_id,
            initial_mode_count,
            max_evidence=max_evidence,
            beam_width=beam_width,
            mode_table=table,
            max_results=1,
        )
        if found:
            candidates.append(found[0])
        if len(candidates) >= candidate_target:
            break
        if searched % 20 == 0:
            print(
                f"state search: searched={searched} candidates={len(candidates)}/"
                f"{candidate_target}",
                flush=True,
            )
    selected, headroom = _select_across_headroom(
        candidates,
        count=state_count,
        mode_table=table,
    )
    selected = sorted(selected, key=lambda state: state.state_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    states_path = write_table(
        state_rows(selected, mode_table=table),
        output_dir / "states.jsonl",
    )
    manifest = {
        "name": "causal_micro_lab_closed_loop_preflight",
        "initial_mode_count": initial_mode_count,
        "states": len(selected),
        "candidate_pool": len(candidates),
        "hidden_modes_searched": searched,
        "selection_axis": "continuous_planner_headroom",
        "planner_headroom": headroom,
        "source_mode_split": "test",
        "split_seed": split_seed,
        "generation_seed": generation_seed,
        "excluded_paths": exclusion_paths,
        "files": {"states.jsonl": _sha256(states_path)},
    }
    (output_dir / "state_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return selected, manifest


def _write_curves(output_dir: Path, summary: dict[str, Any]) -> None:
    rows = [
        {"policy": policy, **row}
        for policy, values in summary["policies"].items()
        for row in values["curves"]
    ]
    with (output_dir / "curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _state_metric_means(trajectories, *, max_steps: int) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float | bool]]] = {}
    for trajectory in trajectories:
        grouped.setdefault(trajectory.initial_state_id, []).append(
            trajectory_metrics(trajectory, max_steps=max_steps)
        )
    return {
        state_id: {
            key: statistics.fmean(float(row[key]) for row in rows) for key in rows[0]
        }
        for state_id, rows in grouped.items()
    }


def _paired_comparison(
    greedy: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    state_ids = sorted(set(greedy) & set(baseline))
    orientations = {
        "identification_at_budget": ("identified", 1.0),
        "cumulative_information_gain_bits": (
            "cumulative_information_gain_bits",
            1.0,
        ),
        "normalized_log_version_space_auc": (
            "normalized_log_version_space_auc",
            -1.0,
        ),
        "entropy_regret": ("mean_entropy_regret", -1.0),
        "bank_coverage_score": ("mean_bank_coverage_score", 1.0),
    }
    rng = random.Random(seed)
    result = {}
    for label, (metric, orientation) in orientations.items():
        differences = [
            orientation * (greedy[state_id][metric] - baseline[state_id][metric])
            for state_id in state_ids
        ]
        samples = sorted(
            statistics.fmean(rng.choice(differences) for _ in differences)
            for _ in range(bootstrap_samples)
        )
        result[label] = {
            "greedy_improvement": statistics.fmean(differences),
            "ci95_low": samples[round(0.025 * (bootstrap_samples - 1))],
            "ci95_high": samples[round(0.975 * (bootstrap_samples - 1))],
        }
    return result


def _write_plot(output_dir: Path, summary: dict[str, Any]) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    labels = {
        "oracle_planner": "Full-space oracle",
        "greedy_representative": "Greedy representative bank",
        "uniform_distinct": "Uniform distinct bank",
        "uniform_with_replacement": "Uniform sampled bank",
        "collapsed": "Collapsed bank",
        "random_experiment": "Random experiment",
    }
    colors = {
        "oracle_planner": "#222222",
        "greedy_representative": "#008C8C",
        "uniform_distinct": "#3B73B9",
        "uniform_with_replacement": "#8C6BB1",
        "collapsed": "#D95F02",
        "random_experiment": "#999999",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for policy in REFERENCE_POLICIES:
        rows = summary["policies"][policy]["curves"]
        steps = [row["step"] for row in rows]
        axes[0].plot(
            steps,
            [row["identification_success"] for row in rows],
            marker="o",
            color=colors[policy],
            label=labels[policy],
        )
        axes[1].plot(
            steps,
            [row["mean_log2_version_space_size"] for row in rows],
            marker="o",
            color=colors[policy],
            label=labels[policy],
        )
    axes[0].set(
        xlabel="Experiments performed",
        ylabel="Hidden worlds identified",
        ylim=(-0.02, 1.02),
    )
    axes[1].set(
        xlabel="Experiments performed",
        ylabel="Mean remaining uncertainty (bits)",
    )
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    figure.suptitle("Does a representative hypothesis bank improve experiment choice?")
    figure.tight_layout(rect=(0, 0.14, 1, 0.94))
    path = output_dir / "reference_policy_curves.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return str(path)


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    states_path = output_dir / "states.jsonl"
    if states_path.exists() and not args.rebuild_states:
        states = load_states(str(states_path))
        manifest = json.loads(
            (output_dir / "state_manifest.json").read_text(encoding="utf-8")
        )
        print(f"reusing {len(states)} states from {states_path}", flush=True)
    else:
        states, manifest = build_states(
            output_dir=output_dir,
            initial_mode_count=args.initial_mode_count,
            state_count=args.states,
            candidate_multiplier=args.candidate_multiplier,
            split_seed=args.split_seed,
            generation_seed=args.generation_seed,
            beam_width=args.beam_width,
            max_evidence=args.max_evidence,
            repo_root=Path(args.repo_root).resolve(),
        )
    table = build_mode_table()
    policy_summaries = {}
    state_metrics = {}
    for policy in REFERENCE_POLICIES:
        replicates = (
            1
            if policy in {"oracle_planner", "greedy_representative"}
            else args.sampled_replicates
        )
        trajectories = []
        for state_index, state in enumerate(states, start=1):
            for replicate in range(replicates):
                trajectories.append(
                    run_reference_trajectory(
                        state,
                        policy=policy,
                        k=args.k,
                        max_steps=args.max_steps,
                        seed=args.seed,
                        replicate=replicate,
                        mode_table=table,
                    )
                )
            if state_index % 8 == 0:
                print(
                    f"policy={policy} states={state_index}/{len(states)}",
                    flush=True,
                )
        policy_summaries[policy] = summarize_reference_trajectories(
            trajectories,
            max_steps=args.max_steps,
        )
        state_metrics[policy] = _state_metric_means(
            trajectories,
            max_steps=args.max_steps,
        )
        values = policy_summaries[policy]
        print(
            f"{policy}: identification={values['identification_at_budget']:.3f} "
            f"information_gain={values['mean_cumulative_information_gain_bits']:.3f} "
            f"log_auc={values['mean_normalized_log_version_space_auc']:.3f}",
            flush=True,
        )
    summary = {
        "configuration": {
            "initial_mode_count": args.initial_mode_count,
            "K": args.k,
            "T": args.max_steps,
            "states": len(states),
            "sampled_replicates": args.sampled_replicates,
            "seed": args.seed,
        },
        "state_manifest": manifest,
        "policies": policy_summaries,
        "paired_greedy_improvement": {
            baseline: _paired_comparison(
                state_metrics["greedy_representative"],
                state_metrics[baseline],
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + index,
            )
            for index, baseline in enumerate(
                ("uniform_distinct", "uniform_with_replacement"),
                start=1,
            )
        },
        "plot": None,
    }
    _write_curves(output_dir, summary)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["plot"] = _write_plot(output_dir, summary)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU gate for causal micro-lab closed-loop bank utility."
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/causal_micro_lab_closed_loop_preflight/m512_k4_t5",
    )
    parser.add_argument("--initial-mode-count", type=int, default=512)
    parser.add_argument("--states", type=int, default=48)
    parser.add_argument("--candidate-multiplier", type=int, default=2)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--sampled-replicates", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--generation-seed", type=int, default=20260801)
    parser.add_argument("--beam-width", type=int, default=256)
    parser.add_argument("--max-evidence", type=int, default=8)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--rebuild-states", action="store_true")
    args = parser.parse_args()
    if args.states < 1 or args.k < 1 or args.max_steps < 1:
        parser.error("states, K, and max steps must be positive")
    summary = run_analysis(args)
    print(
        json.dumps(
            {
                policy: {
                    key: values[key]
                    for key in (
                        "identification_at_budget",
                        "mean_cumulative_information_gain_bits",
                        "mean_normalized_log_version_space_auc",
                        "mean_entropy_regret",
                        "mean_bank_coverage_score",
                    )
                }
                for policy, values in summary["policies"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
