from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

from scattered_discovery.envs.causal_micro_lab.predictive_diversity import (
    PredictiveDistanceMatrix,
    RepresentativeCoverageMatrix,
)
from scattered_discovery.envs.causal_micro_lab.signatures import build_mode_table
from scattered_discovery.envs.causal_micro_lab.state_generator import EvidenceState
from scattered_discovery.envs.causal_micro_lab.verifier import VerificationResult


def _mean_pairwise_mode_distance(
    mode_ids: set[str],
    state: EvidenceState,
    *,
    matrix: PredictiveDistanceMatrix | None = None,
) -> float:
    if len(mode_ids) < 2:
        return 0.0
    distance_matrix = matrix or PredictiveDistanceMatrix(
        state.valid_mode_ids,
        state.observed_experiment_ids(),
    )
    pairs = math.comb(len(mode_ids), 2)
    return distance_matrix.dispersion_mass(mode_ids) / pairs


def group_metrics(
    results: list[VerificationResult],
    state: EvidenceState,
    *,
    representative_matrix: RepresentativeCoverageMatrix | None = None,
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
    probabilities = (
        [count / len(valid_mode_ids) for count in counts.values()]
        if valid_mode_ids
        else []
    )
    entropy = -sum(p * math.log(p) for p in probabilities) if probabilities else 0.0
    effective = math.exp(entropy) if probabilities else 0.0
    dominant_mode_mass = max(probabilities, default=0.0)
    duplicity = 1.0 - (len(unique_valid) / valid_count) if valid_count else 0.0
    distance_matrix = PredictiveDistanceMatrix(
        state.valid_mode_ids,
        state.observed_experiment_ids(),
    )
    available_separation = distance_matrix.separation_summary()
    generated_separation = _mean_pairwise_mode_distance(
        unique_valid,
        state,
        matrix=distance_matrix,
    )
    diversity_recovery = distance_matrix.predictive_diversity_recovery(
        [
            result.semantic_mode_id if result.is_currently_valid_mode else None
            for result in results
        ]
    )
    coverage_matrix = representative_matrix or RepresentativeCoverageMatrix(
        state.valid_mode_ids,
        state.observed_experiment_ids(),
    )
    full_outcome_separation = coverage_matrix.separation_summary()
    representative_coverage = coverage_matrix.representative_coverage(valid_mode_ids)
    full_outcome_generated_separation = (
        sum(
            coverage_matrix.distance(left, right)
            for left, right in combinations(unique_valid, 2)
        )
        / math.comb(len(unique_valid), 2)
        if len(unique_valid) >= 2
        else 0.0
    )
    oracle_singleton_error, _ = coverage_matrix.optimal_subset(1)
    oracle_budget_error, _ = coverage_matrix.optimal_subset(budget)
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
        "available_predictive_separation": available_separation.mean,
        "available_predictive_separation_normalized": (
            available_separation.normalized_mean
        ),
        "predictive_diversity_recovery": diversity_recovery.score,
        "recovered_predictive_dispersion_mass": diversity_recovery.recovered_mass,
        "oracle_predictive_dispersion_mass": diversity_recovery.oracle_mass,
        "predictive_diversity_target_size": float(diversity_recovery.target_size),
        "predictive_coverage_auc": representative_coverage.coverage_auc,
        "predictive_representation_error": (
            representative_coverage.representation_error
        ),
        "predictive_placement_regret": representative_coverage.placement_regret,
        "oracle_representation_error_same_cardinality": (
            representative_coverage.oracle_error_same_size
        ),
        "oracle_predictive_coverage_auc_same_cardinality": (
            1.0 - representative_coverage.oracle_error_same_size
        ),
        "full_outcome_generated_separation": full_outcome_generated_separation,
        "full_outcome_available_separation": full_outcome_separation.mean,
        "full_outcome_available_separation_normalized": (
            full_outcome_separation.normalized_mean
        ),
        "predictive_coverage_opportunity_at_budget": max(
            0.0,
            oracle_singleton_error - oracle_budget_error,
        ),
        "oracle_representation_error_at_budget": oracle_budget_error,
        "generated_to_available_separation": (
            generated_separation / available_separation.mean
            if available_separation.mean > 0
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
