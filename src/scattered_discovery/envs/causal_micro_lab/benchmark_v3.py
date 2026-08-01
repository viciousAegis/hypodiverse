from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
from statistics import mean
from typing import Iterable

from scattered_discovery.envs.causal_micro_lab.benchmark_v2 import (
    CONTINUOUS_SEPARATION_LABEL,
    VisibleEvidenceKey,
    visible_evidence_key,
)
from scattered_discovery.envs.causal_micro_lab.predictive_diversity import (
    OUTCOME_CHANNELS,
    RepresentativeCoverageMatrix,
)
from scattered_discovery.envs.causal_micro_lab.signatures import (
    ModeTable,
    build_mode_table,
)
from scattered_discovery.envs.causal_micro_lab.state_generator import EvidenceState


FULL_OUTCOME_SEPARATION_DEFINITION = "full_outcome_disagreement_v3"
DEFAULT_REPRESENTATIVE_BUDGET = 4


def annotate_representative_geometry(
    state: EvidenceState,
    *,
    representative_budget: int = DEFAULT_REPRESENTATIVE_BUDGET,
    mode_table: ModeTable | None = None,
) -> EvidenceState:
    table = mode_table or build_mode_table()
    matrix = RepresentativeCoverageMatrix(
        state.valid_mode_ids,
        state.observed_experiment_ids(),
        mode_table=table,
    )
    separation = matrix.separation_summary()
    singleton_error, _ = matrix.optimal_subset(1)
    budget_error, _ = matrix.optimal_subset(representative_budget)
    return replace(
        state,
        mean_separation=separation.mean,
        minimum_separation=separation.minimum,
        maximum_separation=separation.maximum,
        separation_bucket=CONTINUOUS_SEPARATION_LABEL,
        separation_definition=FULL_OUTCOME_SEPARATION_DEFINITION,
        separation_targets=OUTCOME_CHANNELS,
        representative_budget=representative_budget,
        oracle_singleton_representation_error=singleton_error,
        oracle_budget_representation_error=budget_error,
        representative_coverage_opportunity=max(0.0, singleton_error - budget_error),
    )


def annotate_state_bank(
    states: Iterable[EvidenceState],
    *,
    representative_budget: int = DEFAULT_REPRESENTATIVE_BUDGET,
    mode_table: ModeTable | None = None,
) -> list[EvidenceState]:
    table = mode_table or build_mode_table()
    return [
        annotate_representative_geometry(
            state,
            representative_budget=representative_budget,
            mode_table=table,
        )
        for state in states
    ]


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


def select_geometry_states(
    states: Iterable[EvidenceState],
    *,
    states_per_m: int,
    seed: int,
    target_counts: tuple[int, ...] = (4, 8, 12, 16),
    excluded_hidden_mode_ids: set[str] | None = None,
    excluded_state_ids: set[str] | None = None,
    excluded_visible_keys: set[VisibleEvidenceKey] | None = None,
) -> list[EvidenceState]:
    if states_per_m <= 0:
        raise ValueError("states_per_m must be positive")
    hidden_seen = set(excluded_hidden_mode_ids or ())
    state_seen = set(excluded_state_ids or ())
    visible_seen = set(excluded_visible_keys or ())
    bank = list(states)
    selected = []
    for mode_count in target_counts:
        candidates = [
            state
            for state in bank
            if state.valid_mode_count == mode_count
            and state.representative_coverage_opportunity is not None
            and state.representative_coverage_opportunity > 0.0
            and state.hidden_mode_id not in hidden_seen
            and state.state_id not in state_seen
            and visible_evidence_key(state) not in visible_seen
        ]
        if len(candidates) < states_per_m:
            raise RuntimeError(
                f"Could only find {len(candidates)}/{states_per_m} eligible "
                f"states for M={mode_count}"
            )
        opportunities = [
            float(state.representative_coverage_opportunity) for state in candidates
        ]
        minimum, maximum = min(opportunities), max(opportunities)
        targets = (
            [(minimum + maximum) / 2.0]
            if states_per_m == 1
            else [
                minimum + index * (maximum - minimum) / (states_per_m - 1)
                for index in range(states_per_m)
            ]
        )
        family_counts: Counter[str] = Counter()
        evidence_counts: Counter[int] = Counter()
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
                    f"Could only select {target_index}/{states_per_m} states "
                    f"for M={mode_count}"
                )
            picked = min(
                eligible,
                key=lambda state: (
                    abs(float(state.representative_coverage_opportunity) - target),
                    family_counts[state.family_bucket],
                    evidence_counts[state.evidence_size],
                    _stable_random_key(seed, state.state_id),
                    state.state_id,
                ),
            )
            selected.append(picked)
            hidden_seen.add(picked.hidden_mode_id)
            state_seen.add(picked.state_id)
            visible_seen.add(visible_evidence_key(picked))
            family_counts[picked.family_bucket] += 1
            evidence_counts[picked.evidence_size] += 1
    return sorted(
        selected,
        key=lambda state: (
            state.valid_mode_count,
            float(state.representative_coverage_opportunity or 0.0),
            state.state_id,
        ),
    )


def geometry_distribution_by_m(
    states: Iterable[EvidenceState],
    *,
    target_counts: tuple[int, ...] = (4, 8, 12, 16),
) -> dict[str, dict[str, float | int]]:
    output = {}
    items = list(states)
    for mode_count in target_counts:
        group = [state for state in items if state.valid_mode_count == mode_count]
        opportunities = sorted(
            float(state.representative_coverage_opportunity or 0.0) for state in group
        )
        separations = sorted(state.mean_separation for state in group)
        if not group:
            output[str(mode_count)] = {"count": 0}
            continue

        def percentile(values: list[float], fraction: float) -> float:
            return values[round(fraction * (len(values) - 1))]

        output[str(mode_count)] = {
            "count": len(group),
            "opportunity_minimum": opportunities[0],
            "opportunity_p05": percentile(opportunities, 0.05),
            "opportunity_median": percentile(opportunities, 0.5),
            "opportunity_mean": mean(opportunities),
            "opportunity_p95": percentile(opportunities, 0.95),
            "opportunity_maximum": opportunities[-1],
            "separation_minimum": separations[0],
            "separation_median": percentile(separations, 0.5),
            "separation_mean": mean(separations),
            "separation_maximum": separations[-1],
        }
    return output
