from __future__ import annotations

import argparse
import json
from pathlib import Path

from scattered_discovery.envs.causal_micro_lab.tables import (
    build_split_dataset,
    experiment_rows,
    generate_states,
    mode_rows,
    sft_rows_for_states,
    state_rows,
    verl_rows_for_states,
    write_table,
)


PRESETS = {
    "smoke": {
        "output_dir": "data/causal_micro_lab/smoke",
        "target_counts": (4, 8, 16),
        "train_states_per_count": 2,
        "val_states_per_count": 1,
        "test_states_per_count": 1,
        "beam_width": 64,
        "targets_per_state": 2,
    },
    "pilot": {
        "output_dir": "data/causal_micro_lab/pilot",
        "target_counts": (4, 8, 16),
        "train_states_per_count": 128,
        "val_states_per_count": 32,
        "test_states_per_count": 32,
        "beam_width": 256,
        "targets_per_state": 2,
    },
    "canonical_eval": {
        "output_dir": "data/causal_micro_lab/canonical_eval",
        "target_counts": (4, 8, 16),
        "train_states_per_count": 0,
        "val_states_per_count": 128,
        "test_states_per_count": 128,
        "beam_width": 256,
        "targets_per_state": 1,
        "source_splits": {
            "train": "test",
            "val": "test",
            "test": "test",
        },
    },
    "trainable": {
        "output_dir": "data/causal_micro_lab/trainable",
        "target_counts": (4, 8, 16),
        "train_states_per_count": 2048,
        "val_states_per_count": 256,
        "test_states_per_count": 256,
        "beam_width": 256,
        "targets_per_state": 4,
    },
}


def _reward_task_overrides(args: argparse.Namespace) -> dict[str, float]:
    overrides = {}
    for cli_name, task_name in (
        ("nonempty_output_reward", "nonempty_output_reward"),
        ("rule_marker_reward", "rule_marker_reward"),
        ("parse_valid_reward", "parse_valid_reward"),
        ("syntax_valid_reward", "syntax_valid_reward"),
        ("evidence_consistent_reward", "evidence_consistent_reward"),
        ("valid_hypothesis_reward", "valid_hypothesis_reward"),
    ):
        value = getattr(args, cli_name, None)
        if value is not None:
            overrides[task_name] = float(value)
    return overrides


def _agent_overrides(args: argparse.Namespace) -> dict[str, float | bool]:
    overrides: dict[str, float | bool] = {}
    if getattr(args, "length_penalty_start", None) is not None:
        overrides["length_penalty_start"] = float(args.length_penalty_start)
    if getattr(args, "length_penalty_max", None) is not None:
        overrides["length_penalty_max"] = float(args.length_penalty_max)
    if getattr(args, "mask_truncated", None) is not None:
        overrides["mask_truncated"] = bool(args.mask_truncated)
    return overrides


def _add_agent_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--length-penalty-start", type=float)
    parser.add_argument("--length-penalty-max", type=float)
    parser.add_argument(
        "--mask-truncated",
        choices=(0, 1),
        type=int,
        help="Whether cap-hit responses should have response_mask zeroed.",
    )


def build_tables_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/causal_micro_lab")
    parser.add_argument("--suffix", choices=["jsonl", "parquet"], default="jsonl")
    args = parser.parse_args()
    root = Path(args.output_dir)
    mode_path = write_table(mode_rows(), root / f"modes.{args.suffix}")
    experiment_path = write_table(experiment_rows(), root / f"experiments.{args.suffix}")
    print(f"wrote {mode_path}")
    print(f"wrote {experiment_path}")


def generate_states_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/causal_micro_lab/states.jsonl")
    parser.add_argument("--states-per-count", type=int, default=8)
    parser.add_argument("--target-counts")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-evidence", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=256)
    args = parser.parse_args()
    target_counts = (
        tuple(int(item) for item in args.target_counts.split(",") if item)
        if args.target_counts
        else (4, 8, 16)
    )
    states = generate_states(
        target_counts=target_counts,
        states_per_count=args.states_per_count,
        seed=args.seed,
        max_evidence=args.max_evidence,
        beam_width=args.beam_width,
    )
    path = write_table(state_rows(states), args.output)
    print(f"wrote {len(states)} states to {path}")


def build_sft_dataset_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--output", default="data/causal_micro_lab/sft_train.jsonl")
    parser.add_argument("--targets-per-state", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    from scattered_discovery.envs.causal_micro_lab.parser import parse_record_state

    records = [
        json.loads(line)
        for line in Path(args.states).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    states = [parse_record_state(record) for record in records]
    rows = sft_rows_for_states(
        states,
        targets_per_state=args.targets_per_state,
        seed=args.seed,
    )
    path = write_table(rows, args.output)
    print(f"wrote {len(rows)} SFT rows to {path}")


def run_closed_loop_eval_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()
    from scattered_discovery.envs.causal_micro_lab.planner import run_oracle_closed_loop
    from scattered_discovery.envs.causal_micro_lab.parser import parse_record_state

    records = [
        json.loads(line)
        for line in Path(args.states).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    traces = [
        run_oracle_closed_loop(parse_record_state(record), max_steps=args.max_steps)
        for record in records
    ]
    summary = {}
    if traces:
        summary = {
            "episodes": len(traces),
            "identification_success": sum(trace.identified() for trace in traces)
            / len(traces),
            "mean_remaining_version_space_size": sum(
                trace.final_version_space_size() for trace in traces
            )
            / len(traces),
            "mean_steps": sum(len(trace.steps) for trace in traces) / len(traces),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_verl_dataset_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/causal_micro_lab/verl_train.jsonl")
    parser.add_argument("--states-per-count", type=int, default=8)
    parser.add_argument("--target-counts", default="4,8,16")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--nonempty-output-reward", type=float)
    parser.add_argument("--rule-marker-reward", type=float)
    parser.add_argument("--parse-valid-reward", type=float)
    parser.add_argument("--syntax-valid-reward", type=float)
    parser.add_argument("--evidence-consistent-reward", type=float)
    parser.add_argument("--valid-hypothesis-reward", type=float)
    _add_agent_override_args(parser)
    args = parser.parse_args()
    states = generate_states(
        target_counts=tuple(int(item) for item in args.target_counts.split(",")),
        states_per_count=args.states_per_count,
        seed=args.seed,
    )
    rows = verl_rows_for_states(
        states,
        task_overrides=_reward_task_overrides(args),
        agent_overrides=_agent_overrides(args),
    )
    path = write_table(rows, args.output)
    print(f"wrote {len(rows)} veRL rows to {path}")


def build_eval_rows_from_states_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--states",
        help="Single states JSONL file to convert, e.g. states_val.jsonl.",
    )
    parser.add_argument(
        "--output",
        help="Output JSONL for --states mode, e.g. verl_val.jsonl.",
    )
    parser.add_argument(
        "--input-dir",
        help="Directory containing states_train/val/test.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for verl_train/val/test.jsonl. Defaults to --input-dir.",
    )
    parser.add_argument("--agent-name", default="causal_micro_lab_agent_loop")
    parser.add_argument("--data-source-prefix", default="causal_micro_lab")
    parser.add_argument("--nonempty-output-reward", type=float)
    parser.add_argument("--rule-marker-reward", type=float)
    parser.add_argument("--parse-valid-reward", type=float)
    parser.add_argument("--syntax-valid-reward", type=float)
    parser.add_argument("--evidence-consistent-reward", type=float)
    parser.add_argument("--valid-hypothesis-reward", type=float)
    _add_agent_override_args(parser)
    args = parser.parse_args()
    from scattered_discovery.envs.causal_micro_lab.parser import parse_record_state

    if bool(args.states) == bool(args.input_dir):
        raise SystemExit("Provide exactly one of --states or --input-dir.")
    if args.states and not args.output:
        raise SystemExit("--states mode requires --output.")

    def convert(states_path: Path, output_path: Path, data_source: str) -> None:
        records = [
            json.loads(line)
            for line in states_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        states = [parse_record_state(record) for record in records]
        rows = verl_rows_for_states(
            states,
            agent_name=args.agent_name,
            data_source=data_source,
            task_overrides=_reward_task_overrides(args),
            agent_overrides=_agent_overrides(args),
        )
        path = write_table(rows, output_path)
        print(f"wrote {len(rows)} eval/RL rows to {path}")

    if args.states:
        convert(
            Path(args.states),
            Path(args.output),
            args.data_source_prefix,
        )
        return

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    for split in ("train", "val", "test"):
        states_path = input_dir / f"states_{split}.jsonl"
        if not states_path.exists():
            print(f"skipping missing {states_path}")
            continue
        convert(
            states_path,
            output_dir / f"verl_{split}.jsonl",
            f"{args.data_source_prefix}_{split}",
        )


def build_split_dataset_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--output-dir")
    parser.add_argument("--target-counts")
    parser.add_argument("--train-states-per-count", type=int)
    parser.add_argument("--val-states-per-count", type=int)
    parser.add_argument("--test-states-per-count", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-evidence", type=int, default=8)
    parser.add_argument("--beam-width", type=int)
    parser.add_argument("--targets-per-state", type=int)
    parser.add_argument("--suffix", choices=["jsonl", "parquet"], default="jsonl")
    parser.add_argument("--no-verl", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--source-splits",
        default="",
        help=(
            "Optional output-to-hidden-split mapping, e.g. "
            "train:train,val:test,test:test."
        ),
    )
    parser.add_argument(
        "--separation-buckets",
        default="",
        help="Optional comma-separated bucket balance request, e.g. low,medium,high.",
    )
    parser.add_argument("--nonempty-output-reward", type=float)
    parser.add_argument("--rule-marker-reward", type=float)
    parser.add_argument("--parse-valid-reward", type=float)
    parser.add_argument("--syntax-valid-reward", type=float)
    parser.add_argument("--evidence-consistent-reward", type=float)
    parser.add_argument("--valid-hypothesis-reward", type=float)
    _add_agent_override_args(parser)
    args = parser.parse_args()
    preset = PRESETS[args.preset]
    output_dir = args.output_dir or preset["output_dir"]
    counts = {
        "train": args.train_states_per_count
        if args.train_states_per_count is not None
        else preset["train_states_per_count"],
        "val": args.val_states_per_count
        if args.val_states_per_count is not None
        else preset["val_states_per_count"],
        "test": args.test_states_per_count
        if args.test_states_per_count is not None
        else preset["test_states_per_count"],
    }
    target_counts = (
        tuple(int(item) for item in args.target_counts.split(",") if item)
        if args.target_counts
        else tuple(preset.get("target_counts", (4, 8, 16)))
    )
    source_splits = dict(preset.get("source_splits", {}))
    if args.source_splits:
        source_splits = {}
        for item in args.source_splits.split(","):
            output_split, source_split = item.split(":", 1)
            source_splits[output_split.strip()] = source_split.strip()
    separation_buckets = tuple(
        item.strip()
        for item in args.separation_buckets.split(",")
        if item.strip()
    )
    outputs = build_split_dataset(
        output_dir=output_dir,
        target_counts=target_counts,
        states_per_count=counts,
        seed=args.seed,
        max_evidence=args.max_evidence,
        beam_width=args.beam_width or preset["beam_width"],
        targets_per_state=args.targets_per_state or preset["targets_per_state"],
        suffix=args.suffix,
        separation_buckets=separation_buckets,
        source_splits=source_splits or None,
        verl_task_overrides=_reward_task_overrides(args),
        verl_agent_overrides=_agent_overrides(args),
        include_verl=not args.no_verl,
        progress=None if args.quiet else lambda message: print(message, flush=True),
        progress_every=max(1, args.progress_every),
    )
    for name, path in sorted(outputs.items()):
        print(f"wrote {name}: {path}")
