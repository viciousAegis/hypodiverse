from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Iterable

from scattered_discovery.envs.causal_micro_lab.signatures import (
    ModeTable,
    build_mode_table,
)


OUTCOME_CHANNELS = ("Z1", "Z2", "Y")
DEFAULT_PREDICTION_TARGET = (2,)


def prediction_target_indices(channels: Iterable[str]) -> tuple[int, ...]:
    names = tuple(str(channel).upper() for channel in channels)
    if not names:
        raise ValueError("at least one prediction target is required")
    unknown = sorted(set(names) - set(OUTCOME_CHANNELS))
    if unknown:
        raise ValueError(f"unknown prediction targets: {', '.join(unknown)}")
    return tuple(OUTCOME_CHANNELS.index(name) for name in names)


def prediction_target_names(indices: Iterable[int]) -> tuple[str, ...]:
    values = tuple(int(index) for index in indices)
    if not values:
        raise ValueError("at least one prediction target is required")
    if any(index < 0 or index >= len(OUTCOME_CHANNELS) for index in values):
        raise ValueError("prediction target index is out of range")
    if len(set(values)) != len(values):
        raise ValueError("prediction target indices must be unique")
    return tuple(OUTCOME_CHANNELS[index] for index in values)


def available_query_ids(
    observed_experiment_ids: Iterable[int],
    *,
    mode_table: ModeTable | None = None,
) -> tuple[int, ...]:
    table = mode_table or build_mode_table()
    observed = {int(experiment_id) for experiment_id in observed_experiment_ids}
    return tuple(
        experiment.experiment_id
        for experiment in table.experiments
        if experiment.experiment_id not in observed
    )


def mode_prediction_distance(
    left_mode_id: str,
    right_mode_id: str,
    query_ids: Iterable[int],
    *,
    target_indices: tuple[int, ...] = DEFAULT_PREDICTION_TARGET,
    mode_table: ModeTable | None = None,
) -> float:
    table = mode_table or build_mode_table()
    prediction_target_names(target_indices)
    queries = tuple(int(query_id) for query_id in query_ids)
    if not queries:
        return 0.0
    left = table.modes_by_id[left_mode_id]
    right = table.modes_by_id[right_mode_id]
    disagreements = sum(
        left.signature[query_id][target] != right.signature[query_id][target]
        for query_id in queries
        for target in target_indices
    )
    return disagreements / (len(queries) * len(target_indices))


def theoretical_binary_pairwise_max(mode_count: int) -> float:
    """Maximum mean pairwise disagreement at one binary prediction query."""
    if mode_count < 2:
        return 0.0
    disagreeing_pairs = (mode_count * mode_count) // 4
    total_pairs = math.comb(mode_count, 2)
    return disagreeing_pairs / total_pairs


@dataclass(frozen=True)
class SeparationSummary:
    mean: float
    minimum: float
    maximum: float
    normalized_mean: float


@dataclass(frozen=True)
class PredictiveDiversityResult:
    score: float
    recovered_mass: float
    oracle_mass: float
    valid_unique_modes: int
    target_size: int
    oracle_mode_ids: tuple[str, ...]


class PredictiveDistanceMatrix:
    def __init__(
        self,
        mode_ids: Iterable[str],
        observed_experiment_ids: Iterable[int],
        *,
        target_indices: tuple[int, ...] = DEFAULT_PREDICTION_TARGET,
        mode_table: ModeTable | None = None,
    ) -> None:
        self.table = mode_table or build_mode_table()
        self.mode_ids = tuple(dict.fromkeys(str(mode_id) for mode_id in mode_ids))
        self.mode_set = frozenset(self.mode_ids)
        self.query_ids = available_query_ids(
            observed_experiment_ids,
            mode_table=self.table,
        )
        self.target_indices = tuple(target_indices)
        prediction_target_names(self.target_indices)
        self._distances: dict[tuple[str, str], float] = {}
        self._optimal_cache: dict[int, tuple[float, tuple[str, ...]]] = {}
        for left, right in combinations(self.mode_ids, 2):
            self._distances[self._key(left, right)] = mode_prediction_distance(
                left,
                right,
                self.query_ids,
                target_indices=self.target_indices,
                mode_table=self.table,
            )

    @staticmethod
    def _key(left: str, right: str) -> tuple[str, str]:
        return (left, right) if left < right else (right, left)

    def distance(self, left: str, right: str) -> float:
        if left == right:
            return 0.0
        return self._distances[self._key(left, right)]

    def dispersion_mass(self, mode_ids: Iterable[str]) -> float:
        unique = tuple(dict.fromkeys(str(mode_id) for mode_id in mode_ids))
        return sum(
            self.distance(left, right) for left, right in combinations(unique, 2)
        )

    def separation_summary(self) -> SeparationSummary:
        distances = list(self._distances.values())
        if not distances:
            return SeparationSummary(0.0, 0.0, 0.0, 0.0)
        mean = sum(distances) / len(distances)
        theoretical_max = theoretical_binary_pairwise_max(len(self.mode_ids))
        return SeparationSummary(
            mean=mean,
            minimum=min(distances),
            maximum=max(distances),
            normalized_mean=mean / theoretical_max if theoretical_max else 0.0,
        )

    def optimal_subset(self, budget: int) -> tuple[float, tuple[str, ...]]:
        target_size = min(max(0, int(budget)), len(self.mode_ids))
        cached = self._optimal_cache.get(target_size)
        if cached is not None:
            return cached
        if target_size == 0:
            return (0.0, ())
        if target_size == 1:
            return (0.0, (min(self.mode_ids),))
        if target_size == len(self.mode_ids):
            ordered = tuple(sorted(self.mode_ids))
            result = (self.dispersion_mass(ordered), ordered)
            self._optimal_cache[target_size] = result
            return result

        best_mass = -1.0
        best_subset: tuple[str, ...] = ()
        for subset in combinations(sorted(self.mode_ids), target_size):
            mass = self.dispersion_mass(subset)
            if mass > best_mass + 1e-15 or (
                math.isclose(mass, best_mass, abs_tol=1e-15) and subset < best_subset
            ):
                best_mass = mass
                best_subset = subset
        result = (max(0.0, best_mass), best_subset)
        self._optimal_cache[target_size] = result
        return result

    def predictive_diversity_recovery(
        self,
        generated_mode_ids: Iterable[str | None],
        *,
        budget: int | None = None,
    ) -> PredictiveDiversityResult:
        generated = tuple(generated_mode_ids)
        effective_budget = len(generated) if budget is None else int(budget)
        target_size = min(max(0, effective_budget), len(self.mode_ids))
        valid_unique = tuple(
            dict.fromkeys(
                mode_id
                for mode_id in generated[:effective_budget]
                if mode_id is not None and mode_id in self.mode_set
            )
        )
        recovered_mass = self.dispersion_mass(valid_unique)
        oracle_mass, oracle_ids = self.optimal_subset(effective_budget)
        score = recovered_mass / oracle_mass if oracle_mass > 0.0 else 0.0
        return PredictiveDiversityResult(
            score=min(1.0, max(0.0, score)),
            recovered_mass=recovered_mass,
            oracle_mass=oracle_mass,
            valid_unique_modes=len(valid_unique),
            target_size=target_size,
            oracle_mode_ids=oracle_ids,
        )


def separation_for_modes(
    mode_ids: Iterable[str],
    observed_experiment_ids: Iterable[int],
    *,
    target_indices: tuple[int, ...] = DEFAULT_PREDICTION_TARGET,
    mode_table: ModeTable | None = None,
) -> SeparationSummary:
    return PredictiveDistanceMatrix(
        mode_ids,
        observed_experiment_ids,
        target_indices=target_indices,
        mode_table=mode_table,
    ).separation_summary()
