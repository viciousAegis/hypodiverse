#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

from scattered_discovery.envs.causal_micro_lab.benchmark_v2 import (
    assert_clean_eval_audit,
    eval_overlap_audit,
    visible_evidence_key,
)
from scattered_discovery.envs.causal_micro_lab.benchmark_v3 import (
    DEFAULT_REPRESENTATIVE_BUDGET,
    annotate_state_bank,
    geometry_distribution_by_m,
    select_geometry_states,
)
from scattered_discovery.envs.causal_micro_lab.parser import parse_record_state
from scattered_discovery.envs.causal_micro_lab.signatures import build_mode_table
from scattered_discovery.envs.causal_micro_lab.state_generator import EvidenceState
from scattered_discovery.envs.causal_micro_lab.tables import (
    split_mode_ids,
    state_rows,
    verl_rows_for_states,
    write_table,
)


TARGET_COUNTS = (4, 8, 12, 16)
DEFAULT_EXCLUSION_GLOBS = (
    "data/causal_micro_lab/*/states_*.jsonl",
    "eval_sets/causal_micro_lab/final_v1/states.jsonl",
    "eval_sets/causal_micro_lab/final_v2/states.jsonl",
    "eval_sets/causal_micro_lab/closed_loop_v1/states.jsonl",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_state_row(row: dict[str, Any]) -> EvidenceState:
    if "env_spec_json" in row:
        return parse_record_state(
            json.loads(str(row["env_spec_json"]))["task"]["state"]
        )
    if "state_json" in row:
        return parse_record_state(json.loads(str(row["state_json"])))
    return parse_record_state(row)


def _load_states(path: Path) -> list[EvidenceState]:
    return [
        _load_state_row(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_exclusions(
    repo_root: Path,
    globs: tuple[str, ...],
) -> tuple[list[EvidenceState], list[str]]:
    states_by_id: dict[str, EvidenceState] = {}
    paths = []
    for pattern in globs:
        for path in sorted(repo_root.glob(pattern)):
            paths.append(str(path.relative_to(repo_root)))
            for state in _load_states(path):
                states_by_id[state.state_id] = state
    return sorted(states_by_id.values(), key=lambda state: state.state_id), paths


def build_final_eval_v3(
    *,
    state_bank_path: Path,
    output_dir: Path,
    states_per_m: int,
    split_seed: int,
    selection_seed: int,
    representative_budget: int,
    repo_root: Path,
) -> dict[str, Any]:
    table = build_mode_table()
    split_ids = split_mode_ids(seed=split_seed, mode_table=table)
    bank = annotate_state_bank(
        _load_states(state_bank_path),
        representative_budget=representative_budget,
        mode_table=table,
    )
    exclusions, exclusion_paths = _load_exclusions(repo_root, DEFAULT_EXCLUSION_GLOBS)
    selected = select_geometry_states(
        bank,
        states_per_m=states_per_m,
        seed=selection_seed,
        target_counts=TARGET_COUNTS,
        excluded_hidden_mode_ids={state.hidden_mode_id for state in exclusions},
        excluded_state_ids={state.state_id for state in exclusions},
        excluded_visible_keys={visible_evidence_key(state) for state in exclusions},
    )
    audit = eval_overlap_audit(
        selected,
        train_mode_ids=split_ids["train"],
        val_mode_ids=split_ids["val"],
        test_mode_ids=split_ids["test"],
        excluded_states=exclusions,
    )
    assert_clean_eval_audit(audit)

    output_dir.mkdir(parents=True, exist_ok=True)
    states_path = write_table(
        state_rows(selected, mode_table=table), output_dir / "states.jsonl"
    )
    verl_path = write_table(
        verl_rows_for_states(
            selected,
            data_source="causal_micro_lab_final_eval_v3",
            mode_table=table,
        ),
        output_dir / "verl_test.jsonl",
    )
    distribution = geometry_distribution_by_m(selected, target_counts=TARGET_COUNTS)
    manifest = {
        "name": "causal_micro_lab_final_eval_v3",
        "schema_version": 3,
        "state_bank": str(state_bank_path),
        "target_counts": list(TARGET_COUNTS),
        "states_per_M": states_per_m,
        "total_states": len(selected),
        "split_seed": split_seed,
        "selection_seed": selection_seed,
        "source_mode_split": "test",
        "one_state_per_hidden_mode": True,
        "separation_definition": "full_outcome_disagreement_v3",
        "prediction_outcome": ["Z1", "Z2", "Y"],
        "selection_method": "evenly_spaced_representative_coverage_opportunity",
        "representative_budget": representative_budget,
        "representative_coverage_definition": "full_outcome_facility_location_v3",
        "geometry_by_M": distribution,
        "counts_by_family": dict(
            sorted(Counter(state.family_bucket for state in selected).items())
        ),
        "evidence_size": {
            "minimum": min(state.evidence_size for state in selected),
            "mean": mean(state.evidence_size for state in selected),
            "maximum": max(state.evidence_size for state in selected),
        },
        "exclusion_paths": exclusion_paths,
        "excluded_states": len(exclusions),
        "overlap_audit": audit,
        "state_id_order_sha256": hashlib.sha256(
            ("\n".join(state.state_id for state in selected) + "\n").encode("ascii")
        ).hexdigest(),
        "files": {
            "states.jsonl": _sha256(states_path),
            "verl_test.jsonl": _sha256(verl_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"selected={len(selected)}")
    for mode_count, values in distribution.items():
        print(
            f"M={mode_count} count={values['count']} "
            f"opportunity=[{values['opportunity_minimum']:.4f}, "
            f"{values['opportunity_maximum']:.4f}] "
            f"separation=[{values['separation_minimum']:.4f}, "
            f"{values['separation_maximum']:.4f}]"
        )
    print(f"audit={json.dumps(audit, sort_keys=True)}")
    print(f"manifest={manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the geometry-controlled causal micro-lab v3 eval set."
    )
    parser.add_argument(
        "--state-bank",
        type=Path,
        default=Path(
            "artifacts/causal_micro_lab_environment_characterization/"
            "predictive_v2/states.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_sets/causal_micro_lab/final_v3"),
    )
    parser.add_argument("--states-per-m", type=int, default=48)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--selection-seed", type=int, default=20260801)
    parser.add_argument(
        "--representative-budget",
        type=int,
        default=DEFAULT_REPRESENTATIVE_BUDGET,
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    if not args.state_bank.exists():
        parser.error(f"missing state bank: {args.state_bank}")
    build_final_eval_v3(
        state_bank_path=args.state_bank,
        output_dir=args.output_dir,
        states_per_m=args.states_per_m,
        split_seed=args.split_seed,
        selection_seed=args.selection_seed,
        representative_budget=args.representative_budget,
        repo_root=args.repo_root.resolve(),
    )


if __name__ == "__main__":
    main()
