from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scattered_discovery.envs.causal_micro_lab.eval import (
    _flatten_numeric,
    _set_verification_to_eval_dict,
    load_states,
    summarize_grouped_records,
    summarize_records,
)
from scattered_discovery.envs.causal_micro_lab.verifier import (
    verify_output,
    verify_output_set,
    verify_verbalized_output_set,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score saved causal micro-lab generations without inference."
    )
    parser.add_argument("--episodes", required=True)
    parser.add_argument(
        "--states",
        default="eval_sets/causal_micro_lab/final_v1/verl_test.jsonl",
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--output-mode",
        choices=["single", "multi_answer_rlvr", "verbalized_sampling"],
    )
    parser.add_argument("--answer-count", type=int, default=4)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    args = parser.parse_args()

    episodes_path = Path(args.episodes)
    states_path = Path(args.states)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else episodes_path.parent / "rescored_relaxed"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(episodes_path)
    states = load_states(states_path)
    states_by_id = {state.state_id: state for state in states}
    if not records:
        raise SystemExit(f"No episodes found in {episodes_path}")

    detected_mode = records[0].get("verification", {}).get("output_mode", "single")
    output_mode = args.output_mode or str(detected_mode)
    rescored = []
    for record in records:
        state_id = str(record["state_id"])
        if state_id not in states_by_id:
            raise SystemExit(f"Episode state_id is absent from state table: {state_id}")
        state = states_by_id[state_id]
        output = str(record.get("output") or "")
        if output_mode == "verbalized_sampling":
            verification = _set_verification_to_eval_dict(
                verify_verbalized_output_set(
                    output,
                    state,
                    expected_count=args.answer_count,
                ),
                state,
                output_mode=output_mode,
            )
        elif output_mode == "multi_answer_rlvr":
            verification = _set_verification_to_eval_dict(
                verify_output_set(
                    output,
                    state,
                    expected_count=args.answer_count,
                ),
                state,
                output_mode=output_mode,
            )
        else:
            verification = verify_output(output, state).as_dict()
        rescored.append({**record, "verification": verification})

    summary = summarize_records(rescored)
    summary.update(
        {
            "rescored_from": str(episodes_path),
            "states_source": str(states_path),
            "output_mode": output_mode,
            "answer_count": args.answer_count,
        }
    )
    set_summary = summarize_grouped_records(rescored, states)

    rescored_path = output_dir / "episodes.jsonl"
    summary_path = output_dir / "summary.json"
    set_summary_path = output_dir / "set_summary.json"
    _write_jsonl(rescored_path, rescored)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    set_summary_path.write_text(
        json.dumps(set_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.wandb_project:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            job_type="offline-rescore",
            config={
                "episodes": str(episodes_path),
                "states": str(states_path),
                "output_mode": output_mode,
                "answer_count": args.answer_count,
            },
        )
        run.log(_flatten_numeric(summary, prefix="eval_summary"))
        run.log(_flatten_numeric(set_summary, prefix="set_summary"))
        run.finish()

    print(f"episodes={rescored_path}")
    print(f"summary={summary_path}")
    print(f"set_summary={set_summary_path}")
    print(
        "candidate_parse_valid="
        f"{summary.get('candidate_parse_valid', 0.0):.6f} "
        "candidate_valid_mode="
        f"{summary.get('candidate_currently_valid_mode', 0.0):.6f} "
        f"pass_at_k={set_summary.get('pass_at_k', 0.0):.6f}"
    )


if __name__ == "__main__":
    main()
