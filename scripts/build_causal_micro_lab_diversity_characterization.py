#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

from scattered_discovery.envs.causal_micro_lab.benchmark_v2 import (
    CONTINUOUS_SEPARATION_LABEL,
    DEFAULT_TARGET_COUNTS,
    separation_distribution_by_m,
)
from scattered_discovery.envs.causal_micro_lab.parser import parse_record_state
from scattered_discovery.envs.causal_micro_lab.predictive_diversity import (
    theoretical_binary_pairwise_max,
)
from scattered_discovery.envs.causal_micro_lab.signatures import build_mode_table
from scattered_discovery.envs.causal_micro_lab.state_generator import EvidenceState


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _state_from_row(row: dict[str, Any]) -> EvidenceState:
    if "env_spec_json" in row:
        spec = json.loads(str(row["env_spec_json"]))
        return parse_record_state(spec["task"]["state"])
    if "state_json" in row:
        return parse_record_state(json.loads(str(row["state_json"])))
    return parse_record_state(row)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_characterization(
    *,
    source_paths: tuple[Path, ...],
    output_dir: Path,
) -> dict[str, Any]:
    table = build_mode_table()
    states_by_id: dict[str, EvidenceState] = {}
    for source in source_paths:
        for row in _load_jsonl(source):
            state = _state_from_row(row)
            states_by_id[state.state_id] = state

    states = sorted(
        (
            replace(state, separation_bucket=CONTINUOUS_SEPARATION_LABEL)
            for state in states_by_id.values()
        ),
        key=lambda state: (
            state.valid_mode_count,
            state.mean_separation,
            state.state_id,
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    states_path = output_dir / "states.jsonl"
    with states_path.open("w", encoding="utf-8") as handle:
        for state in states:
            handle.write(
                json.dumps(
                    state.to_record(mode_table=table, include_private=True),
                    sort_keys=True,
                )
                + "\n"
            )

    distributions = separation_distribution_by_m(
        states,
        target_counts=DEFAULT_TARGET_COUNTS,
    )
    for mode_count, values in distributions.items():
        maximum = float(values["maximum"])
        theoretical = theoretical_binary_pairwise_max(int(mode_count))
        values["theoretical_binary_pairwise_max"] = theoretical
        values["maximum_observed_fraction_of_theoretical"] = maximum / theoretical
    manifest = {
        "name": "causal_micro_lab_diversity_characterization_v2",
        "source_paths": [str(path) for path in source_paths],
        "total_states": len(states),
        "prediction_targets": ["Y"],
        "separation_definition": "predictive_target_disagreement_v2",
        "separation_scale": "continuous",
        "separation_by_M": distributions,
        "files": {"states.jsonl": _sha256(states_path)},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"states={len(states)}")
    for mode_count, distribution in distributions.items():
        print(
            f"M={mode_count} count={distribution['count']} "
            f"range=[{distribution['minimum']:.4f}, "
            f"{distribution['maximum']:.4f}]"
        )
    print(f"wrote={states_path}")
    print(f"manifest={manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute target-outcome separation and build a CPU-only causal "
            "micro-lab characterization bank."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        default=[],
        help="State JSONL source. May be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/causal_micro_lab_environment_characterization/predictive_v2"
        ),
    )
    args = parser.parse_args()
    sources = tuple(args.source) or (
        Path(
            "artifacts/causal_micro_lab_environment_characterization/"
            "dense_v1/states.jsonl"
        ),
    )
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        parser.error(f"missing source files: {', '.join(missing)}")
    build_characterization(
        source_paths=sources,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
