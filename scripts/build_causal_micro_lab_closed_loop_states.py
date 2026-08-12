#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
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
    write_table,
)


DEFAULT_EXCLUSION_GLOBS = (
    "data/causal_micro_lab/*/states_*.jsonl",
    "eval_sets/causal_micro_lab/final_v1/states.jsonl",
)


def _visible_key(
    state: EvidenceState,
) -> tuple[tuple[int, tuple[int, int, int]], ...]:
    return tuple((item.experiment_id, tuple(item.outcome)) for item in state.evidence)


def _load_state(path: Path, row: dict[str, Any]) -> EvidenceState:
    if "env_spec_json" in row:
        spec = json.loads(str(row["env_spec_json"]))
        return parse_record_state(spec["task"]["state"])
    if "state_json" in row:
        return parse_record_state(json.loads(str(row["state_json"])))
    return parse_record_state(row)


def _load_exclusions(
    root: Path,
    patterns: tuple[str, ...],
) -> tuple[
    set[str],
    set[str],
    set[tuple[tuple[int, tuple[int, int, int]], ...]],
    list[str],
]:
    hidden_modes: set[str] = set()
    state_ids: set[str] = set()
    visible_keys: set[tuple[tuple[int, tuple[int, int, int]], ...]] = set()
    paths = []
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            paths.append(str(path.relative_to(root)))
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    state = _load_state(path, json.loads(line))
                    hidden_modes.add(state.hidden_mode_id)
                    state_ids.add(state.state_id)
                    visible_keys.add(_visible_key(state))
    return hidden_modes, state_ids, visible_keys, paths


def _assign_tertiles(states: list[EvidenceState]) -> list[EvidenceState]:
    ordered = sorted(states, key=lambda state: (state.mean_separation, state.state_id))
    return [
        replace(
            state,
            separation_bucket=("low", "medium", "high")[
                min(2, 3 * index // max(1, len(ordered)))
            ],
        )
        for index, state in enumerate(ordered)
    ]


def _select_by_separation(
    states: list[EvidenceState],
    *,
    count: int,
    seed: int,
) -> list[EvidenceState]:
    rng = random.Random(seed)
    quotas = {
        bucket: count // 3 + int(index < count % 3)
        for index, bucket in enumerate(("low", "medium", "high"))
    }
    selected = []
    for bucket, quota in quotas.items():
        candidates = [state for state in states if state.separation_bucket == bucket]
        rng.shuffle(candidates)
        if len(candidates) < quota:
            raise RuntimeError(
                f"{bucket}: found {len(candidates)} candidates, need {quota}"
            )
        selected.extend(candidates[:quota])
    return sorted(selected, key=lambda state: state.state_id)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_states(output_dir: Path) -> None:
    """Verify the frozen closed-loop worlds against their checked-in manifest."""
    states_path = output_dir / "states.jsonl"
    manifest_path = output_dir / "manifest.json"
    if not states_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing frozen closed-loop files under {output_dir}; restore them "
            "from the HypoDiverse repository"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = str(manifest["files"]["states.jsonl"])
    actual = _sha256(states_path)
    if actual != expected:
        raise RuntimeError(
            f"closed-loop states hash mismatch: expected {expected}, got {actual}"
        )
    print(f"verified {states_path}: {actual}")


def build_states(
    *,
    output_dir: Path,
    initial_mode_counts: tuple[int, ...],
    trajectories_per_count: int,
    split_seed: int,
    generation_seed: int,
    beam_width: int,
    max_evidence: int,
    candidate_multiplier: int,
    repo_root: Path,
    exclusion_globs: tuple[str, ...],
) -> dict[str, Any]:
    table = build_mode_table()
    test_modes = split_mode_ids(seed=split_seed, mode_table=table)["test"]
    (
        excluded_hidden_modes,
        excluded_state_ids,
        excluded_visible_keys,
        exclusion_paths,
    ) = _load_exclusions(repo_root, exclusion_globs)
    used_hidden_modes = set(excluded_hidden_modes)
    used_state_ids = set(excluded_state_ids)
    used_visible_keys = set(excluded_visible_keys)
    all_states: list[EvidenceState] = []

    for mode_count in initial_mode_counts:
        rng = random.Random(generation_seed + mode_count)
        hidden_modes = sorted(test_modes - used_hidden_modes)
        rng.shuffle(hidden_modes)
        candidate_target = trajectories_per_count * candidate_multiplier
        candidates = []
        searched = 0
        for hidden_mode_id in hidden_modes:
            searched += 1
            found = find_states(
                hidden_mode_id,
                mode_count,
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
            if len(candidates) >= candidate_target:
                break
        if len(candidates) < trajectories_per_count:
            raise RuntimeError(
                f"M={mode_count}: found {len(candidates)} eligible held-out states "
                f"after searching {searched} modes; need {trajectories_per_count}"
            )
        selected = _select_by_separation(
            _assign_tertiles(candidates),
            count=trajectories_per_count,
            seed=generation_seed + 1000 + mode_count,
        )
        all_states.extend(selected)
        used_hidden_modes.update(state.hidden_mode_id for state in selected)
        used_state_ids.update(state.state_id for state in selected)
        used_visible_keys.update(_visible_key(state) for state in selected)
        print(
            f"M={mode_count}: selected={len(selected)} "
            f"candidate_pool={len(candidates)} searched={searched}",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    states_path = write_table(
        state_rows(all_states, mode_table=table),
        output_dir / "states.jsonl",
    )
    manifest = {
        "name": "causal_micro_lab_closed_loop_v1",
        "initial_mode_counts": list(initial_mode_counts),
        "trajectories_per_count": trajectories_per_count,
        "total_trajectories": len(all_states),
        "one_state_per_hidden_mode": True,
        "balance_axis": "predictive_separation_tertile",
        "source_mode_split": "test",
        "split_seed": split_seed,
        "generation_seed": generation_seed,
        "beam_width": beam_width,
        "max_evidence": max_evidence,
        "candidate_multiplier": candidate_multiplier,
        "excluded_paths": exclusion_paths,
        "counts_by_initial_M": {
            str(mode_count): sum(
                state.valid_mode_count == mode_count for state in all_states
            )
            for mode_count in initial_mode_counts
        },
        "counts_by_separation": {
            bucket: sum(state.separation_bucket == bucket for state in all_states)
            for bucket in ("low", "medium", "high")
        },
        "files": {"states.jsonl": _sha256(states_path)},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {states_path}")
    print(f"wrote {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="eval_sets/causal_micro_lab/closed_loop_v1",
    )
    parser.add_argument("--initial-mode-counts", default="16,32")
    parser.add_argument("--trajectories-per-count", type=int, default=64)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--generation-seed", type=int, default=20260730)
    parser.add_argument("--beam-width", type=int, default=256)
    parser.add_argument("--max-evidence", type=int, default=8)
    parser.add_argument("--candidate-multiplier", type=int, default=2)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_frozen_states(Path(args.output_dir))
        return
    manifest = build_states(
        output_dir=Path(args.output_dir),
        initial_mode_counts=tuple(
            int(item) for item in args.initial_mode_counts.split(",") if item.strip()
        ),
        trajectories_per_count=args.trajectories_per_count,
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
