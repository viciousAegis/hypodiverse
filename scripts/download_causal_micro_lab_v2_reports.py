#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_RUNS = {
    "base": "54un4aiu",
    "validity": "xhnajqs2",
    "ips": "omr0dfw6",
    "latent_ips": "nt544xbr",
}


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _parse_run(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Runs must use METHOD=RUN_ID syntax")
    method, run_id = value.split("=", 1)
    if not method or not run_id:
        raise argparse.ArgumentTypeError("Runs must use METHOD=RUN_ID syntax")
    return method, run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="akshitsinha3")
    parser.add_argument("--project", default="scattered-discovery")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/causal_micro_lab_final_eval_v2/wandb_reports"),
    )
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run,
        help="METHOD=RUN_ID; repeat to override the default four-run comparison",
    )
    args = parser.parse_args()

    _load_env(args.env_file)
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise SystemExit("WANDB_API_KEY is not set")

    import wandb

    runs = dict(args.run) if args.run else DEFAULT_RUNS
    api = wandb.Api(api_key=api_key)
    for method, run_id in runs.items():
        run = api.run(f"{args.entity}/{args.project}/{run_id}")
        reports = [
            artifact
            for artifact in run.logged_artifacts()
            if artifact.type == "evaluation-report"
        ]
        if len(reports) != 1:
            raise RuntimeError(
                f"Expected one evaluation report for {method}={run_id}, "
                f"found {len(reports)}"
            )
        target = args.output_dir / method
        target.mkdir(parents=True, exist_ok=True)
        reports[0].download(root=str(target))
        state_metrics = target / "state_metrics.csv"
        if not state_metrics.exists():
            raise RuntimeError(f"Downloaded report lacks {state_metrics}")
        print(f"{method}: run={run_id} report={target}")


if __name__ == "__main__":
    main()
