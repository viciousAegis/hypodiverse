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
    DEFAULT_TARGET_COUNTS,
    assert_clean_eval_audit,
    eval_overlap_audit,
    select_continuous_states,
    separation_distribution_by_m,
    visible_evidence_key,
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


DEFAULT_EXCLUSION_GLOBS = (
    "data/causal_micro_lab/*/states_*.jsonl",
    "eval_sets/causal_micro_lab/final_v1/states.jsonl",
    "eval_sets/causal_micro_lab/closed_loop_v1/states.jsonl",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_state_row(path: Path, row: dict[str, Any]) -> EvidenceState:
    if "env_spec_json" in row:
        spec = json.loads(str(row["env_spec_json"]))
        return parse_record_state(spec["task"]["state"])
    if "state_json" in row:
        return parse_record_state(json.loads(str(row["state_json"])))
    if "visible_experiments" in row:
        return parse_record_state(row)
    raise ValueError(f"Unsupported state row in {path}")


def _load_states(path: Path) -> list[EvidenceState]:
    return [
        _load_state_row(path, json.loads(line))
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


def _state_id_order_hash(states: list[EvidenceState]) -> str:
    payload = "\n".join(state.state_id for state in states) + "\n"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def build_final_eval_v2(
    *,
    state_bank_path: Path,
    output_dir: Path,
    states_per_m: int,
    split_seed: int,
    selection_seed: int,
    repo_root: Path,
    exclusion_globs: tuple[str, ...] = DEFAULT_EXCLUSION_GLOBS,
) -> dict[str, Any]:
    table = build_mode_table()
    split_ids = split_mode_ids(seed=split_seed, mode_table=table)
    bank = _load_states(state_bank_path)
    exclusions, exclusion_paths = _load_exclusions(repo_root, exclusion_globs)
    excluded_hidden = {state.hidden_mode_id for state in exclusions}
    excluded_state_ids = {state.state_id for state in exclusions}
    excluded_visible = {visible_evidence_key(state) for state in exclusions}
    selected = select_continuous_states(
        bank,
        states_per_m=states_per_m,
        seed=selection_seed,
        excluded_hidden_mode_ids=excluded_hidden,
        excluded_state_ids=excluded_state_ids,
        excluded_visible_keys=excluded_visible,
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
        state_rows(selected, mode_table=table),
        output_dir / "states.jsonl",
    )
    verl_path = write_table(
        verl_rows_for_states(
            selected,
            data_source="causal_micro_lab_final_eval_v2",
            mode_table=table,
        ),
        output_dir / "verl_test.jsonl",
    )
    counts_by_family = Counter(state.family_bucket for state in selected)
    evidence_sizes = [state.evidence_size for state in selected]
    separations = [state.mean_separation for state in selected]
    separation_by_m = separation_distribution_by_m(selected)
    manifest = {
        "name": "causal_micro_lab_final_eval_v2",
        "schema_version": 2,
        "state_bank": str(state_bank_path),
        "target_counts": list(DEFAULT_TARGET_COUNTS),
        "separation_definition": "predictive_target_disagreement_v2",
        "separation_scale": "continuous",
        "prediction_targets": ["Y"],
        "selection_method": "evenly_spaced_raw_separation",
        "requires_positive_separation": True,
        "states_per_M": states_per_m,
        "total_states": len(selected),
        "split_seed": split_seed,
        "selection_seed": selection_seed,
        "source_mode_split": "test",
        "one_state_per_hidden_mode": True,
        "separation_by_M": separation_by_m,
        "counts_by_family": dict(sorted(counts_by_family.items())),
        "evidence_size": {
            "minimum": min(evidence_sizes),
            "mean": mean(evidence_sizes),
            "maximum": max(evidence_sizes),
        },
        "mean_separation": {
            "minimum": min(separations),
            "mean": mean(separations),
            "maximum": max(separations),
        },
        "exclusion_paths": exclusion_paths,
        "excluded_states": len(exclusions),
        "overlap_audit": audit,
        "state_id_order_sha256": _state_id_order_hash(selected),
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
    for mode_count, distribution in separation_by_m.items():
        print(
            f"M={mode_count} count={distribution['count']} "
            f"range=[{distribution['minimum']:.4f}, "
            f"{distribution['maximum']:.4f}] "
            f"largest_gap={distribution['largest_adjacent_gap']:.4f}"
        )
    print(f"audit={json.dumps(audit, sort_keys=True)}")
    print(f"wrote={states_path}")
    print(f"wrote={verl_path}")
    print(f"manifest={manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the continuous-separation causal micro-lab v2 eval set."
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
        default=Path("eval_sets/causal_micro_lab/final_v2"),
    )
    parser.add_argument("--states-per-m", type=int, default=48)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--selection-seed", type=int, default=20260731)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional repo-relative state-file glob to exclude.",
    )
    args = parser.parse_args()
    if not args.state_bank.exists():
        parser.error(f"missing state bank: {args.state_bank}")
    build_final_eval_v2(
        state_bank_path=args.state_bank,
        output_dir=args.output_dir,
        states_per_m=args.states_per_m,
        split_seed=args.split_seed,
        selection_seed=args.selection_seed,
        repo_root=args.repo_root.resolve(),
        exclusion_globs=(*DEFAULT_EXCLUSION_GLOBS, *tuple(args.exclude)),
    )


if __name__ == "__main__":
    main()
