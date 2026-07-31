from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from statistics import quantiles

from scattered_discovery.envs.causal_micro_lab.interventions import Experiment
from scattered_discovery.envs.causal_micro_lab.predictive_diversity import (
    DEFAULT_PREDICTION_TARGET,
    prediction_target_names,
    separation_for_modes as predictive_separation_for_modes,
    theoretical_binary_pairwise_max,
)
from scattered_discovery.envs.causal_micro_lab.signatures import (
    ModeRecord,
    ModeTable,
    build_mode_table,
)
from scattered_discovery.envs.causal_micro_lab.simulator import Outcome

_OUTCOME_INDEX_CACHE: dict[int, dict[tuple[int, Outcome], frozenset[int]]] = {}

DEFAULT_ABSOLUTE_SEPARATION_BANDS: tuple[tuple[str, float, float], ...] = (
    ("low", 0.03, 0.12),
    ("medium", 0.20, 0.30),
    ("high", 0.36, 0.50),
)


@dataclass(frozen=True)
class EvidenceItem:
    experiment_id: int
    outcome: Outcome

    def to_json(self, experiments: tuple[Experiment, ...]) -> dict[str, object]:
        experiment = experiments[self.experiment_id]
        return {
            "experiment_id": self.experiment_id,
            "inputs": experiment.inputs_dict(),
            "intervention": experiment.intervention,
            "observation": {
                "Z1": self.outcome[0],
                "Z2": self.outcome[1],
                "Y": self.outcome[2],
            },
        }


@dataclass(frozen=True)
class EvidenceState:
    state_id: str
    hidden_mode_id: str
    evidence: tuple[EvidenceItem, ...]
    valid_mode_ids: tuple[str, ...]
    mean_separation: float
    minimum_separation: float
    maximum_separation: float
    separation_bucket: str
    family_bucket: str

    @property
    def evidence_size(self) -> int:
        return len(self.evidence)

    @property
    def valid_mode_count(self) -> int:
        return len(self.valid_mode_ids)

    def observed_experiment_ids(self) -> tuple[int, ...]:
        return tuple(item.experiment_id for item in self.evidence)

    def to_record(
        self,
        *,
        mode_table: ModeTable | None = None,
        include_private: bool = True,
    ) -> dict[str, object]:
        table = mode_table or build_mode_table()
        record: dict[str, object] = {
            "state_id": self.state_id,
            "visible_experiments": [
                item.to_json(table.experiments) for item in self.evidence
            ],
            "available_experiment_ids": [
                experiment.experiment_id
                for experiment in table.experiments
                if experiment.experiment_id not in set(self.observed_experiment_ids())
            ],
            "metadata": {
                "valid_mode_count": self.valid_mode_count,
                "separation_bucket": self.separation_bucket,
                "mean_separation": self.mean_separation,
                "minimum_separation": self.minimum_separation,
                "maximum_separation": self.maximum_separation,
                "normalized_mean_separation": (
                    self.mean_separation
                    / theoretical_binary_pairwise_max(self.valid_mode_count)
                    if self.valid_mode_count > 1
                    else 0.0
                ),
                "separation_definition": "predictive_target_disagreement_v2",
                "separation_targets": list(
                    prediction_target_names(DEFAULT_PREDICTION_TARGET)
                ),
                "family_bucket": self.family_bucket,
                "evidence_size": self.evidence_size,
            },
        }
        if include_private:
            record["private"] = {
                "valid_mode_ids": list(self.valid_mode_ids),
                "hidden_mode_id": self.hidden_mode_id,
            }
        return record


def _state_id(hidden_mode_id: str, evidence_ids: tuple[int, ...]) -> str:
    payload = (
        hidden_mode_id + ":" + ",".join(str(item) for item in sorted(evidence_ids))
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:24]


def valid_modes_for_evidence(
    evidence: tuple[EvidenceItem, ...],
    *,
    mode_table: ModeTable | None = None,
) -> tuple[str, ...]:
    table = mode_table or build_mode_table()
    indices = valid_mode_indices_for_evidence(evidence, mode_table=table)
    return tuple(table.modes[index].mode_id for index in sorted(indices))


def _outcome_index(table: ModeTable) -> dict[tuple[int, Outcome], frozenset[int]]:
    cache_key = id(table)
    cached = _OUTCOME_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    index: dict[tuple[int, Outcome], set[int]] = {}
    for mode_index, mode in enumerate(table.modes):
        for experiment_id, outcome in enumerate(mode.signature):
            index.setdefault((experiment_id, outcome), set()).add(mode_index)
    frozen = {key: frozenset(value) for key, value in index.items()}
    _OUTCOME_INDEX_CACHE[cache_key] = frozen
    return frozen


def valid_mode_indices_for_evidence(
    evidence: tuple[EvidenceItem, ...],
    *,
    mode_table: ModeTable | None = None,
) -> frozenset[int]:
    table = mode_table or build_mode_table()
    if not evidence:
        return frozenset(range(len(table.modes)))
    index = _outcome_index(table)
    compatible: set[int] | None = None
    for item in evidence:
        matching = index.get((item.experiment_id, item.outcome), frozenset())
        if compatible is None:
            compatible = set(matching)
        else:
            compatible.intersection_update(matching)
        if not compatible:
            return frozenset()
    return frozenset(compatible or ())


def separation_for_modes(
    mode_ids: tuple[str, ...],
    observed_experiment_ids: tuple[int, ...],
    *,
    target_indices: tuple[int, ...] = DEFAULT_PREDICTION_TARGET,
    mode_table: ModeTable | None = None,
) -> tuple[float, float, float]:
    summary = predictive_separation_for_modes(
        mode_ids,
        observed_experiment_ids,
        target_indices=target_indices,
        mode_table=mode_table,
    )
    return (summary.mean, summary.minimum, summary.maximum)


def family_bucket(
    mode_ids: tuple[str, ...], *, mode_table: ModeTable | None = None
) -> str:
    table = mode_table or build_mode_table()
    families = {table.modes_by_id[mode_id].family for mode_id in mode_ids}
    if len(families) <= 1:
        return "within_family"
    if len(families) == 2:
        return "mixed"
    return "cross_family"


def assign_separation_buckets(states: list[EvidenceState]) -> list[EvidenceState]:
    by_count: dict[int, list[EvidenceState]] = {}
    for state in states:
        by_count.setdefault(state.valid_mode_count, []).append(state)
    updated: list[EvidenceState] = []
    for group in by_count.values():
        values = [state.mean_separation for state in group]
        if len(values) >= 3:
            low_cut, high_cut = quantiles(values, n=10)[2], quantiles(values, n=10)[6]
        else:
            low_cut = high_cut = values[0] if values else 0.0
        for state in group:
            if state.mean_separation <= low_cut:
                bucket = "low"
            elif state.mean_separation <= high_cut:
                bucket = "medium"
            else:
                bucket = "high"
            updated.append(
                EvidenceState(
                    state_id=state.state_id,
                    hidden_mode_id=state.hidden_mode_id,
                    evidence=state.evidence,
                    valid_mode_ids=state.valid_mode_ids,
                    mean_separation=state.mean_separation,
                    minimum_separation=state.minimum_separation,
                    maximum_separation=state.maximum_separation,
                    separation_bucket=bucket,
                    family_bucket=state.family_bucket,
                )
            )
    return sorted(updated, key=lambda item: (item.valid_mode_count, item.state_id))


def _validated_absolute_bands(
    bands: tuple[tuple[str, float, float], ...],
    tolerance: float,
) -> tuple[tuple[str, float, float], ...]:
    if tolerance < 0.0:
        raise ValueError("separation-band tolerance must be nonnegative")
    ordered_bands = tuple(sorted(bands, key=lambda item: (item[1], item[2], item[0])))
    previous_high = -math.inf
    for label, low, high in ordered_bands:
        if not label:
            raise ValueError("separation band labels must be nonempty")
        if low < 0.0 or high > 1.0 or low > high:
            raise ValueError(f"invalid separation band {label!r}: [{low}, {high}]")
        if low <= previous_high:
            raise ValueError("absolute separation bands must not overlap")
        previous_high = high
    return ordered_bands


def absolute_separation_bucket(
    value: float,
    *,
    bands: tuple[tuple[str, float, float], ...] = DEFAULT_ABSOLUTE_SEPARATION_BANDS,
    outside_label: str = "out_of_band",
    tolerance: float = 1e-12,
) -> str:
    for label, low, high in _validated_absolute_bands(bands, tolerance):
        if low - tolerance <= value <= high + tolerance:
            return label
    return outside_label


def assign_absolute_separation_buckets(
    states: list[EvidenceState],
    *,
    bands: tuple[tuple[str, float, float], ...] = DEFAULT_ABSOLUTE_SEPARATION_BANDS,
    outside_label: str = "out_of_band",
    tolerance: float = 1e-12,
) -> list[EvidenceState]:
    ordered_bands = _validated_absolute_bands(bands, tolerance)

    updated = []
    for state in states:
        bucket = next(
            (
                label
                for label, low, high in ordered_bands
                if low - tolerance <= state.mean_separation <= high + tolerance
            ),
            outside_label,
        )
        updated.append(
            replace(
                state,
                separation_bucket=bucket,
            )
        )
    return sorted(updated, key=lambda item: (item.valid_mode_count, item.state_id))


def make_state(
    *,
    hidden_mode: ModeRecord,
    evidence_ids: tuple[int, ...],
    mode_table: ModeTable | None = None,
    compute_separation: bool = True,
) -> EvidenceState:
    table = mode_table or build_mode_table()
    evidence = tuple(
        EvidenceItem(
            experiment_id=experiment_id, outcome=hidden_mode.signature[experiment_id]
        )
        for experiment_id in sorted(evidence_ids)
    )
    valid_mode_ids = valid_modes_for_evidence(evidence, mode_table=table)
    if compute_separation:
        mean_sep, min_sep, max_sep = separation_for_modes(
            valid_mode_ids,
            tuple(item.experiment_id for item in evidence),
            mode_table=table,
        )
        bucket = family_bucket(valid_mode_ids, mode_table=table)
    else:
        mean_sep, min_sep, max_sep, bucket = 0.0, 0.0, 0.0, "unknown"
    return EvidenceState(
        state_id=_state_id(
            hidden_mode.mode_id, tuple(item.experiment_id for item in evidence)
        ),
        hidden_mode_id=hidden_mode.mode_id,
        evidence=evidence,
        valid_mode_ids=valid_mode_ids,
        mean_separation=mean_sep,
        minimum_separation=min_sep,
        maximum_separation=max_sep,
        separation_bucket="unassigned",
        family_bucket=bucket,
    )


def _make_state_from_indices(
    *,
    hidden_mode: ModeRecord,
    evidence_ids: tuple[int, ...],
    valid_mode_indices: frozenset[int],
    mode_table: ModeTable,
    compute_separation: bool = True,
) -> EvidenceState:
    evidence = tuple(
        EvidenceItem(
            experiment_id=experiment_id, outcome=hidden_mode.signature[experiment_id]
        )
        for experiment_id in sorted(evidence_ids)
    )
    valid_mode_ids = tuple(
        mode_table.modes[index].mode_id for index in sorted(valid_mode_indices)
    )
    if compute_separation:
        mean_sep, min_sep, max_sep = separation_for_modes(
            valid_mode_ids,
            tuple(item.experiment_id for item in evidence),
            mode_table=mode_table,
        )
        bucket = family_bucket(valid_mode_ids, mode_table=mode_table)
    else:
        mean_sep, min_sep, max_sep, bucket = 0.0, 0.0, 0.0, "unknown"
    return EvidenceState(
        state_id=_state_id(
            hidden_mode.mode_id, tuple(item.experiment_id for item in evidence)
        ),
        hidden_mode_id=hidden_mode.mode_id,
        evidence=evidence,
        valid_mode_ids=valid_mode_ids,
        mean_separation=mean_sep,
        minimum_separation=min_sep,
        maximum_separation=max_sep,
        separation_bucket="unassigned",
        family_bucket=bucket,
    )


def find_states(
    hidden_mode: int | str,
    target_mode_count: int,
    max_evidence: int = 8,
    beam_width: int = 256,
    *,
    mode_table: ModeTable | None = None,
    max_results: int | None = None,
) -> list[EvidenceState]:
    table = mode_table or build_mode_table()
    if isinstance(hidden_mode, int):
        hidden = table.modes[hidden_mode]
    else:
        hidden = table.modes_by_id[hidden_mode]

    outcome_index = _outcome_index(table)
    all_mode_indices = frozenset(range(len(table.modes)))
    beams: list[tuple[tuple[int, ...], frozenset[int]]] = [((), all_mode_indices)]
    saved: dict[str, EvidenceState] = {}
    for _depth in range(max_evidence):
        candidates: list[tuple[float, tuple[int, ...], frozenset[int]]] = []
        for evidence_ids, valid_indices in beams:
            used = set(evidence_ids)
            for experiment in table.experiments:
                if experiment.experiment_id in used:
                    continue
                next_ids = tuple(sorted((*evidence_ids, experiment.experiment_id)))
                matching_indices = outcome_index.get(
                    (
                        experiment.experiment_id,
                        hidden.signature[experiment.experiment_id],
                    ),
                    frozenset(),
                )
                next_valid_indices = valid_indices.intersection(matching_indices)
                count = len(next_valid_indices)
                if count < target_mode_count:
                    continue
                if count == target_mode_count:
                    state = _make_state_from_indices(
                        hidden_mode=hidden,
                        evidence_ids=next_ids,
                        valid_mode_indices=next_valid_indices,
                        mode_table=table,
                        compute_separation=True,
                    )
                    saved[state.state_id] = state
                    if max_results is not None and len(saved) >= max_results:
                        return assign_separation_buckets(list(saved.values()))
                score = abs(math.log2(count) - math.log2(target_mode_count)) + (
                    0.05 * len(next_ids)
                )
                candidates.append((score, next_ids, next_valid_indices))
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], item[1]))
        beams = [(ids, valid) for _score, ids, valid in candidates[:beam_width]]
    return assign_separation_buckets(list(saved.values()))
