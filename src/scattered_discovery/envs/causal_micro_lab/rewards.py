from __future__ import annotations

import math
from collections import Counter

from scattered_discovery.envs.causal_micro_lab.signatures import build_mode_table
from scattered_discovery.envs.causal_micro_lab.state_generator import EvidenceState
from scattered_discovery.envs.causal_micro_lab.verifier import VerificationResult


def _mean_pairwise_mode_distance(
    mode_ids: set[str],
    state: EvidenceState,
) -> float:
    if len(mode_ids) < 2:
        return 0.0
    table = build_mode_table()
    observed = {item.experiment_id for item in state.evidence}
    unobserved = [
        experiment.experiment_id
        for experiment in table.experiments
        if experiment.experiment_id not in observed
    ]
    if not unobserved:
        return 0.0
    ordered = sorted(mode_ids)
    distances = []
    for left_index, left_id in enumerate(ordered):
        left = table.modes_by_id[left_id]
        for right_id in ordered[left_index + 1 :]:
            right = table.modes_by_id[right_id]
            disagreements = sum(
                int(left.signature[experiment_id][bit] != right.signature[experiment_id][bit])
                for experiment_id in unobserved
                for bit in range(3)
            )
            distances.append(disagreements / (3 * len(unobserved)))
    return sum(distances) / len(distances)


def group_metrics(
    results: list[VerificationResult],
    state: EvidenceState,
) -> dict[str, float]:
    valid_mode_ids = [
        result.semantic_mode_id
        for result in results
        if result.is_currently_valid_mode and result.semantic_mode_id is not None
    ]
    counts = Counter(valid_mode_ids)
    unique_valid = set(valid_mode_ids)
    available = max(1, state.valid_mode_count)
    budget = max(1, min(len(results), state.valid_mode_count))
    valid_count = len(valid_mode_ids)
    unavoidable_duplicates = max(0, valid_count - state.valid_mode_count)
    duplicate_valid_modes = sum(count - 1 for count in counts.values())
    probabilities = [count / len(valid_mode_ids) for count in counts.values()] if valid_mode_ids else []
    entropy = -sum(p * math.log(p) for p in probabilities) if probabilities else 0.0
    effective = math.exp(entropy) if probabilities else 0.0
    dominant_mode_mass = max(probabilities, default=0.0)
    duplicity = (
        1.0 - (len(unique_valid) / valid_count)
        if valid_count
        else 0.0
    )
    generated_separation = _mean_pairwise_mode_distance(unique_valid, state)
    families = {
        tuple(result.mechanism_family)
        for result in results
        if result.is_currently_valid_mode and result.mechanism_family is not None
    }
    table = build_mode_table()
    available_families = {
        table.modes_by_id[mode_id].family for mode_id in state.valid_mode_ids
    }
    return {
        "num_samples": float(len(results)),
        "num_parse_valid": float(sum(result.parse_valid for result in results)),
        "num_syntax_valid": float(sum(result.syntax_valid for result in results)),
        "num_evidence_consistent": float(
            sum(result.evidence_consistent for result in results)
        ),
        "num_unique_valid_modes": float(len(unique_valid)),
        "available_valid_modes": float(state.valid_mode_count),
        "pass_at_k": float(bool(unique_valid)),
        "exact_coverage": len(unique_valid) / available,
        "budget_normalized_coverage": len(unique_valid) / budget,
        "effective_mode_count": effective,
        "mode_entropy": entropy,
        "dominant_mode_mass": dominant_mode_mass,
        "duplicity": duplicity,
        "generated_mode_separation": generated_separation,
        "generated_to_available_separation": (
            generated_separation / state.mean_separation
            if state.mean_separation > 0
            else 0.0
        ),
        "duplicate_valid_modes": float(duplicate_valid_modes),
        "unavoidable_duplicate_valid_modes": float(unavoidable_duplicates),
        "extra_duplicate_valid_modes": float(
            max(0, duplicate_valid_modes - unavoidable_duplicates)
        ),
        "family_coverage": len(families) / max(1, len(available_families)),
    }


def validity_rewards(results: list[VerificationResult]) -> list[float]:
    return [1.0 if result.is_currently_valid_mode else 0.0 for result in results]


def duplicate_aware_rewards(
    results: list[VerificationResult], *, diversity_lambda: float = 1.0
) -> list[float]:
    mode_ids = [
        result.semantic_mode_id if result.is_currently_valid_mode else None
        for result in results
    ]
    counts = Counter(mode_id for mode_id in mode_ids if mode_id is not None)
    rewards = []
    for mode_id in mode_ids:
        if mode_id is None:
            rewards.append(0.0)
        else:
            rewards.append(1.0 + diversity_lambda / counts[mode_id])
    return rewards


def marginal_coverage_rewards(results: list[VerificationResult]) -> list[float]:
    valid_modes = [
        result.semantic_mode_id if result.is_currently_valid_mode else None
        for result in results
    ]
    full = {mode_id for mode_id in valid_modes if mode_id is not None}
    rewards = []
    for index, mode_id in enumerate(valid_modes):
        if mode_id is None:
            rewards.append(0.0)
            continue
        without = {
            other
            for other_index, other in enumerate(valid_modes)
            if other_index != index and other is not None
        }
        rewards.append(float(len(full) - len(without)))
    return rewards
