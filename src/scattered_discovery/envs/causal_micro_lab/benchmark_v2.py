from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
from statistics import mean
from typing import Iterable

from scattered_discovery.envs.causal_micro_lab.state_generator import (
    EvidenceState,
)


DEFAULT_TARGET_COUNTS = (4, 8, 12, 16)
CONTINUOUS_SEPARATION_LABEL = "continuous"
VisibleEvidenceKey = tuple[tuple[int, tuple[int, int, int]], ...]


def visible_evidence_key(state: EvidenceState) -> VisibleEvidenceKey:
    return tuple((item.experiment_id, tuple(item.outcome)) for item in state.evidence)


def separation_distribution_by_m(
    states: Iterable[EvidenceState],
    *,
    target_counts: tuple[int, ...] = DEFAULT_TARGET_COUNTS,
) -> dict[str, dict[str, float | int]]:
    by_m = {
        mode_count: sorted(
            state.mean_separation
            for state in states
            if state.valid_mode_count == mode_count
        )
        for mode_count in target_counts
    }
    output: dict[str, dict[str, float | int]] = {}
    for mode_count, values in by_m.items():
        if not values:
            output[str(mode_count)] = {"count": 0}
            continue

        def percentile(fraction: float) -> float:
            index = round(fraction * (len(values) - 1))
            return values[index]

        gaps = [right - left for left, right in zip(values, values[1:])]
        output[str(mode_count)] = {
            "count": len(values),
            "minimum": values[0],
            "p05": percentile(0.05),
            "p25": percentile(0.25),
            "median": percentile(0.5),
            "mean": mean(values),
            "p75": percentile(0.75),
            "p95": percentile(0.95),
            "maximum": values[-1],
            "largest_adjacent_gap": max(gaps, default=0.0),
        }
    return output


def _stable_random_key(seed: int, state_id: str) -> str:
    return hashlib.sha256(f"{seed}:{state_id}".encode("ascii")).hexdigest()


def _outside_in_indices(size: int) -> list[int]:
    indices = []
    left, right = 0, size - 1
    while left <= right:
        indices.append(left)
        if right != left:
            indices.append(right)
        left += 1
        right -= 1
    return indices


def select_continuous_states(
    states: Iterable[EvidenceState],
    *,
    states_per_m: int,
    seed: int,
    target_counts: tuple[int, ...] = DEFAULT_TARGET_COUNTS,
    excluded_hidden_mode_ids: set[str] | None = None,
    excluded_state_ids: set[str] | None = None,
    excluded_visible_keys: set[VisibleEvidenceKey] | None = None,
    require_positive_separation: bool = True,
) -> list[EvidenceState]:
    if states_per_m <= 0:
        raise ValueError("states_per_m must be positive")

    hidden_seen = set(excluded_hidden_mode_ids or ())
    state_seen = set(excluded_state_ids or ())
    visible_seen = set(excluded_visible_keys or ())
    candidates_by_m = {
        mode_count: [
            state
            for state in states
            if state.valid_mode_count == mode_count
            and (not require_positive_separation or state.mean_separation > 0.0)
            and state.hidden_mode_id not in hidden_seen
            and state.state_id not in state_seen
            and visible_evidence_key(state) not in visible_seen
        ]
        for mode_count in target_counts
    }
    selected: list[EvidenceState] = []
    for mode_count in target_counts:
        candidates = candidates_by_m[mode_count]
        if len(candidates) < states_per_m:
            raise RuntimeError(
                f"Could only find {len(candidates)}/{states_per_m} eligible "
                f"states for M={mode_count}"
            )
        minimum = min(state.mean_separation for state in candidates)
        maximum = max(state.mean_separation for state in candidates)
        if states_per_m == 1:
            targets = [(minimum + maximum) / 2.0]
        else:
            targets = [
                minimum + index * (maximum - minimum) / (states_per_m - 1)
                for index in range(states_per_m)
            ]
        family_counts: Counter[str] = Counter()
        evidence_counts: Counter[int] = Counter()
        chosen: list[EvidenceState] = []
        for target_index in _outside_in_indices(states_per_m):
            target = targets[target_index]
            eligible = [
                state
                for state in candidates
                if state.hidden_mode_id not in hidden_seen
                and state.state_id not in state_seen
                and visible_evidence_key(state) not in visible_seen
            ]
            if not eligible:
                raise RuntimeError(
                    f"Could only select {len(chosen)}/{states_per_m} states "
                    f"for M={mode_count}"
                )
            picked = min(
                eligible,
                key=lambda state: (
                    abs(state.mean_separation - target),
                    family_counts[state.family_bucket],
                    evidence_counts[state.evidence_size],
                    _stable_random_key(seed, state.state_id),
                    state.state_id,
                ),
            )
            chosen.append(
                replace(
                    picked,
                    separation_bucket=CONTINUOUS_SEPARATION_LABEL,
                )
            )
            hidden_seen.add(picked.hidden_mode_id)
            state_seen.add(picked.state_id)
            visible_seen.add(visible_evidence_key(picked))
            family_counts[picked.family_bucket] += 1
            evidence_counts[picked.evidence_size] += 1
        selected.extend(chosen)

    return sorted(
        selected,
        key=lambda state: (
            state.valid_mode_count,
            state.mean_separation,
            state.state_id,
        ),
    )


def eval_overlap_audit(
    states: Iterable[EvidenceState],
    *,
    train_mode_ids: set[str],
    val_mode_ids: set[str],
    test_mode_ids: set[str],
    excluded_states: Iterable[EvidenceState] = (),
) -> dict[str, int]:
    selected = list(states)
    excluded = list(excluded_states)
    hidden = [state.hidden_mode_id for state in selected]
    state_ids = [state.state_id for state in selected]
    visible = [visible_evidence_key(state) for state in selected]
    excluded_state_ids = {state.state_id for state in excluded}
    excluded_visible = {visible_evidence_key(state) for state in excluded}
    return {
        "states": len(selected),
        "duplicate_hidden_modes": len(hidden) - len(set(hidden)),
        "duplicate_state_ids": len(state_ids) - len(set(state_ids)),
        "duplicate_visible_prompts": len(visible) - len(set(visible)),
        "hidden_modes_outside_test_split": sum(
            mode_id not in test_mode_ids for mode_id in hidden
        ),
        "hidden_mode_overlap_with_train": sum(
            mode_id in train_mode_ids for mode_id in hidden
        ),
        "hidden_mode_overlap_with_val": sum(
            mode_id in val_mode_ids for mode_id in hidden
        ),
        "state_id_overlap_with_exclusions": sum(
            state_id in excluded_state_ids for state_id in state_ids
        ),
        "visible_prompt_overlap_with_exclusions": sum(
            key in excluded_visible for key in visible
        ),
    }


def assert_clean_eval_audit(audit: dict[str, int]) -> None:
    problems = {
        key: value for key, value in audit.items() if key != "states" and value != 0
    }
    if problems:
        details = ", ".join(f"{key}={value}" for key, value in sorted(problems.items()))
        raise RuntimeError(f"frozen eval overlap audit failed: {details}")
