#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from scattered_discovery.envs.causal_micro_lab.parser import parse_record_state
from scattered_discovery.envs.causal_micro_lab.signatures import build_mode_table
from scattered_discovery.envs.causal_micro_lab.state_generator import (
    EvidenceState,
    find_states,
)
from scattered_discovery.envs.causal_micro_lab.tables import (
    split_mode_ids,
    state_rows,
    verl_rows_for_states,
    write_table,
)


DEFAULT_EXCLUSION_GLOBS = (
    "data/causal_micro_lab/canonical_eval/states_*.jsonl",
    "data/causal_micro_lab/pilot/states_*.jsonl",
    "data/causal_micro_lab/pilot_sft/states_*.jsonl",
    "data/causal_micro_lab/trainable/states_*.jsonl",
)


def _visible_key(state: EvidenceState) -> tuple[tuple[int, tuple[int, int, int]], ...]:
    return tuple(
        (item.experiment_id, tuple(item.outcome))
        for item in state.evidence
    )


def _load_state(path: Path, row: dict[str, Any]) -> EvidenceState:
    if "env_spec_json" in row:
        spec = json.loads(str(row["env_spec_json"]))
        return parse_record_state(spec["task"]["state"])
    if "state_json" in row:
        return parse_record_state(json.loads(str(row["state_json"])))
    if "visible_experiments" in row:
        return parse_record_state(row)
    raise ValueError(f"Unsupported state row in {path}")


def _load_exclusions(
    root: Path,
    globs: tuple[str, ...],
) -> tuple[set[str], set[str], set[tuple[tuple[int, tuple[int, int, int]], ...]], list[str]]:
    hidden_mode_ids: set[str] = set()
    state_ids: set[str] = set()
    visible_keys: set[tuple[tuple[int, tuple[int, int, int]], ...]] = set()
    paths: list[str] = []
    for pattern in globs:
        for path in sorted(root.glob(pattern)):
            try:
                paths.append(str(path.relative_to(root)))
            except ValueError:
                paths.append(str(path))
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                state = _load_state(path, json.loads(line))
                hidden_mode_ids.add(state.hidden_mode_id)
                state_ids.add(state.state_id)
                visible_keys.add(_visible_key(state))
    return hidden_mode_ids, state_ids, visible_keys, paths


def _assign_exact_separation_buckets(
    states: list[EvidenceState],
) -> list[EvidenceState]:
    ordered = sorted(states, key=lambda state: (state.mean_separation, state.state_id))
    size = len(ordered)
    updated = []
    for index, state in enumerate(ordered):
        bucket_index = min(2, (3 * index) // max(1, size))
        bucket = ("low", "medium", "high")[bucket_index]
        updated.append(replace(state, separation_bucket=bucket))
    return updated


def _balanced_select(
    states: list[EvidenceState],
    *,
    per_bucket: int,
    seed: int,
) -> list[EvidenceState]:
    rng = random.Random(seed)
    selected: list[EvidenceState] = []
    for separation_bucket in ("low", "medium", "high"):
        candidates = [
            state
            for state in states
            if state.separation_bucket == separation_bucket
        ]
        by_family: dict[str, list[EvidenceState]] = defaultdict(list)
        for state in candidates:
            by_family[state.family_bucket].append(state)
        for family_states in by_family.values():
            rng.shuffle(family_states)
        families = sorted(by_family)
        bucket_selected: list[EvidenceState] = []
        while families and len(bucket_selected) < per_bucket:
            remaining = []
            for family in families:
                if by_family[family]:
                    bucket_selected.append(by_family[family].pop())
                    if len(bucket_selected) >= per_bucket:
                        break
                if by_family[family]:
                    remaining.append(family)
            families = remaining
        if len(bucket_selected) != per_bucket:
            raise RuntimeError(
                f"Could only select {len(bucket_selected)}/{per_bucket} "
                f"{separation_bucket} states"
            )
        selected.extend(bucket_selected)
    return sorted(selected, key=lambda state: (state.valid_mode_count, state.state_id))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_final_eval(
    *,
    output_dir: Path,
    target_counts: tuple[int, ...],
    states_per_count: int,
    split_seed: int,
    generation_seed: int,
    beam_width: int,
    max_evidence: int,
    candidate_multiplier: int,
    repo_root: Path,
    exclusion_globs: tuple[str, ...],
) -> dict[str, Any]:
    if states_per_count % 3:
        raise ValueError("--states-per-count must be divisible by 3")
    table = build_mode_table()
    split_ids = split_mode_ids(seed=split_seed, mode_table=table)
    (
        excluded_hidden_modes,
        excluded_state_ids,
        excluded_visible_keys,
        exclusion_paths,
    ) = _load_exclusions(repo_root, exclusion_globs)
    eligible_hidden_modes = sorted(split_ids["test"] - excluded_hidden_modes)
    all_states: list[EvidenceState] = []
    used_hidden_modes = set(excluded_hidden_modes)
    used_state_ids = set(excluded_state_ids)
    used_visible_keys = set(excluded_visible_keys)

    for target_count in target_counts:
        rng = random.Random(generation_seed + target_count)
        hidden_modes = [
            mode_id
            for mode_id in eligible_hidden_modes
            if mode_id not in used_hidden_modes
        ]
        rng.shuffle(hidden_modes)
        pool_target = states_per_count * candidate_multiplier
        candidates: list[EvidenceState] = []
        searched = 0
        for hidden_mode_id in hidden_modes:
            searched += 1
            found = find_states(
                hidden_mode_id,
                target_count,
                max_evidence=max_evidence,
                beam_width=beam_width,
                mode_table=table,
                max_results=1,
            )
            if not found:
                continue
            state = found[0]
            visible_key = _visible_key(state)
            if state.state_id in used_state_ids or visible_key in used_visible_keys:
                continue
            candidates.append(state)
            used_hidden_modes.add(state.hidden_mode_id)
            used_state_ids.add(state.state_id)
            used_visible_keys.add(visible_key)
            if len(candidates) >= pool_target:
                break
        if len(candidates) < states_per_count:
            raise RuntimeError(
                f"M={target_count}: found only {len(candidates)} eligible states "
                f"after searching {searched} held-out modes"
            )
        bucketed = _assign_exact_separation_buckets(candidates)
        selected = _balanced_select(
            bucketed,
            per_bucket=states_per_count // 3,
            seed=generation_seed + 1000 + target_count,
        )
        all_states.extend(selected)
        print(
            f"M={target_count}: selected={len(selected)} candidates={len(candidates)} "
            f"searched_hidden_modes={searched}",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    states_path = write_table(
        state_rows(all_states, mode_table=table),
        output_dir / "states.jsonl",
    )
    verl_path = write_table(
        verl_rows_for_states(
            all_states,
            data_source="causal_micro_lab_final_eval_v1",
            mode_table=table,
        ),
        output_dir / "verl_test.jsonl",
    )
    counts_by_m = {
        str(target): sum(state.valid_mode_count == target for state in all_states)
        for target in target_counts
    }
    counts_by_separation = {
        bucket: sum(state.separation_bucket == bucket for state in all_states)
        for bucket in ("low", "medium", "high")
    }
    counts_by_family = {
        bucket: sum(state.family_bucket == bucket for state in all_states)
        for bucket in sorted({state.family_bucket for state in all_states})
    }
    manifest = {
        "name": "causal_micro_lab_final_eval_v1",
        "target_counts": list(target_counts),
        "states_per_count": states_per_count,
        "total_states": len(all_states),
        "split_seed": split_seed,
        "generation_seed": generation_seed,
        "source_mode_split": "test",
        "beam_width": beam_width,
        "max_evidence": max_evidence,
        "candidate_multiplier": candidate_multiplier,
        "one_state_per_hidden_mode": True,
        "excluded_paths": exclusion_paths,
        "excluded_hidden_mode_count": len(excluded_hidden_modes),
        "excluded_state_count": len(excluded_state_ids),
        "excluded_visible_prompt_count": len(excluded_visible_keys),
        "counts_by_M": counts_by_m,
        "counts_by_separation": counts_by_separation,
        "counts_by_family": counts_by_family,
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
    print(f"wrote {states_path}")
    print(f"wrote {verl_path}")
    print(f"wrote {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="eval_sets/causal_micro_lab/final_v1",
    )
    parser.add_argument("--target-counts", default="4,8,12,16")
    parser.add_argument("--states-per-count", type=int, default=24)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--generation-seed", type=int, default=20260726)
    parser.add_argument("--beam-width", type=int, default=256)
    parser.add_argument("--max-evidence", type=int, default=8)
    parser.add_argument("--candidate-multiplier", type=int, default=2)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional repo-relative glob of state files to exclude.",
    )
    args = parser.parse_args()
    target_counts = tuple(
        int(item) for item in args.target_counts.split(",") if item.strip()
    )
    manifest = build_final_eval(
        output_dir=Path(args.output_dir),
        target_counts=target_counts,
        states_per_count=args.states_per_count,
        split_seed=args.split_seed,
        generation_seed=args.generation_seed,
        beam_width=args.beam_width,
        max_evidence=args.max_evidence,
        candidate_multiplier=args.candidate_multiplier,
        repo_root=Path(args.repo_root).resolve(),
        exclusion_globs=(*DEFAULT_EXCLUSION_GLOBS, *tuple(args.exclude)),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
