from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections.abc import Iterable
from dataclasses import dataclass

from scattered_discovery.envs.causal_micro_lab.planner import (
    experiment_entropy,
    oracle_disagreement_experiment,
    select_disagreement_experiment,
    select_seeded_random_experiment,
)
from scattered_discovery.envs.causal_micro_lab.signatures import (
    ModeTable,
    build_mode_table,
)
from scattered_discovery.envs.causal_micro_lab.state_generator import (
    EvidenceState,
    make_state,
)


REFERENCE_POLICIES = (
    "oracle_planner",
    "greedy_representative",
    "uniform_distinct",
    "uniform_with_replacement",
    "collapsed",
    "random_experiment",
)


@dataclass(frozen=True)
class ReferenceTrajectory:
    policy: str
    initial_state_id: str
    replicate: int
    initial_mode_count: int
    steps: tuple[dict[str, float | int | str | bool], ...]

    @property
    def identified(self) -> bool:
        return bool(self.steps and self.steps[-1]["version_space_size_after"] == 1)


def _query_ids(state: EvidenceState, table: ModeTable) -> tuple[int, ...]:
    observed = set(state.observed_experiment_ids())
    return tuple(
        experiment.experiment_id
        for experiment in table.experiments
        if experiment.experiment_id not in observed
    )


def _prediction_bits(
    mode_id: str,
    query_ids: Iterable[int],
    table: ModeTable,
) -> int:
    """Pack each categorical (Z1, Z2, Y) outcome as a one-hot byte."""

    value = 0
    signature = table.modes_by_id[mode_id].signature
    for experiment_id in query_ids:
        z1, z2, y = signature[experiment_id]
        category = (z1 << 2) | (z2 << 1) | y
        value = (value << 8) | (1 << category)
    return value


def _categorical_disagreements(left: int, right: int) -> int:
    # Differing one-hot bytes contribute exactly two set bits to XOR.
    return (left ^ right).bit_count() // 2


def greedy_representative_modes(
    state: EvidenceState,
    budget: int,
    *,
    mode_table: ModeTable | None = None,
) -> tuple[str, ...]:
    """Greedy facility-location representatives over unobserved predictions."""

    table = mode_table or build_mode_table()
    target = min(max(0, int(budget)), state.valid_mode_count)
    if target == 0:
        return ()
    mode_ids = tuple(sorted(state.valid_mode_ids))
    if target == len(mode_ids):
        return mode_ids
    queries = _query_ids(state, table)
    encoded = tuple(_prediction_bits(mode_id, queries, table) for mode_id in mode_ids)

    # The first representative is the medoid. Later representatives greedily
    # reduce mean distance to the nearest selected prediction signature.
    first = min(
        range(len(mode_ids)),
        key=lambda index: (
            sum(_categorical_disagreements(encoded[index], other) for other in encoded),
            mode_ids[index],
        ),
    )
    selected = [first]
    nearest = [_categorical_disagreements(encoded[first], other) for other in encoded]
    while len(selected) < target:
        selected_set = set(selected)
        best = min(
            (index for index in range(len(mode_ids)) if index not in selected_set),
            key=lambda index: (
                sum(
                    min(
                        distance,
                        _categorical_disagreements(encoded[index], encoded[other]),
                    )
                    for other, distance in enumerate(nearest)
                ),
                mode_ids[index],
            ),
        )
        selected.append(best)
        nearest = [
            min(
                distance,
                _categorical_disagreements(encoded[best], encoded[index]),
            )
            for index, distance in enumerate(nearest)
        ]
    return tuple(mode_ids[index] for index in selected)


def representative_coverage_score(
    state: EvidenceState,
    representatives: Iterable[str],
    *,
    mode_table: ModeTable | None = None,
) -> float:
    table = mode_table or build_mode_table()
    selected = tuple(
        dict.fromkeys(
            mode_id
            for mode_id in representatives
            if mode_id in set(state.valid_mode_ids)
        )
    )
    if not selected:
        return 0.0
    queries = _query_ids(state, table)
    dimensions = len(queries)
    if dimensions == 0:
        return 1.0
    representative_bits = tuple(
        _prediction_bits(mode_id, queries, table) for mode_id in selected
    )
    error = statistics.fmean(
        min(
            _categorical_disagreements(
                _prediction_bits(mode_id, queries, table),
                representative,
            )
            for representative in representative_bits
        )
        / dimensions
        for mode_id in state.valid_mode_ids
    )
    return 1.0 - error


def planner_headroom(
    state: EvidenceState,
    *,
    mode_table: ModeTable | None = None,
) -> float:
    """Entropy advantage of the best experiment over an average experiment."""

    table = mode_table or build_mode_table()
    queries = _query_ids(state, table)
    if not queries:
        return 0.0
    entropies = [
        experiment_entropy(state.valid_mode_ids, experiment_id, mode_table=table)
        for experiment_id in queries
    ]
    return max(entropies) - statistics.fmean(entropies)


def _stable_seed(
    seed: int,
    policy: str,
    state: EvidenceState,
    step: int,
    replicate: int,
) -> int:
    payload = f"{seed}:{policy}:{state.hidden_mode_id}:{step}:{replicate}:" + ",".join(
        str(item) for item in state.observed_experiment_ids()
    )
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big")


def _policy_bank(
    state: EvidenceState,
    *,
    policy: str,
    k: int,
    seed: int,
    step: int,
    replicate: int,
    mode_table: ModeTable,
) -> tuple[str, ...]:
    if policy == "greedy_representative":
        return greedy_representative_modes(state, k, mode_table=mode_table)
    rng = random.Random(_stable_seed(seed, policy, state, step, replicate))
    modes = state.valid_mode_ids
    if policy == "uniform_distinct":
        return tuple(rng.sample(modes, min(k, len(modes))))
    if policy == "uniform_with_replacement":
        return tuple(rng.choice(modes) for _ in range(k))
    if policy == "collapsed":
        return (rng.choice(modes),) * k
    return ()


def run_reference_trajectory(
    state: EvidenceState,
    *,
    policy: str,
    k: int,
    max_steps: int,
    seed: int,
    replicate: int = 0,
    mode_table: ModeTable | None = None,
) -> ReferenceTrajectory:
    if policy not in REFERENCE_POLICIES:
        raise ValueError(f"unsupported reference policy: {policy}")
    table = mode_table or build_mode_table()
    hidden = table.modes_by_id[state.hidden_mode_id]
    current = state
    records: list[dict[str, float | int | str | bool]] = []
    for step_index in range(1, max_steps + 1):
        if current.valid_mode_count <= 1:
            break
        oracle_experiment = oracle_disagreement_experiment(
            current,
            mode_table=table,
        )
        if oracle_experiment is None:
            break
        oracle_entropy = experiment_entropy(
            current.valid_mode_ids,
            oracle_experiment,
            mode_table=table,
        )
        bank = _policy_bank(
            current,
            policy=policy,
            k=k,
            seed=seed,
            step=step_index,
            replicate=replicate,
            mode_table=table,
        )
        unique_bank = tuple(dict.fromkeys(bank))
        if policy == "oracle_planner":
            experiment_id = oracle_experiment
            selection_reason = "oracle"
            coverage = 1.0
        elif policy == "random_experiment" or len(unique_bank) < 2:
            experiment_id = select_seeded_random_experiment(
                current,
                seed=_stable_seed(seed, policy, current, step_index, replicate),
                mode_table=table,
            )
            selection_reason = "seeded_random"
            coverage = representative_coverage_score(
                current,
                unique_bank,
                mode_table=table,
            )
        else:
            experiment_id = select_disagreement_experiment(
                list(bank),
                current,
                mode_table=table,
            )
            selection_reason = "bank_entropy"
            coverage = representative_coverage_score(
                current,
                unique_bank,
                mode_table=table,
            )
        if experiment_id is None:
            break
        selected_true_entropy = experiment_entropy(
            current.valid_mode_ids,
            experiment_id,
            mode_table=table,
        )
        before = current.valid_mode_count
        current = make_state(
            hidden_mode=hidden,
            evidence_ids=tuple(
                sorted((*current.observed_experiment_ids(), experiment_id))
            ),
            mode_table=table,
            compute_separation=False,
        )
        after = current.valid_mode_count
        records.append(
            {
                "step": step_index,
                "experiment_id": experiment_id,
                "selection_reason": selection_reason,
                "bank_unique_modes": len(unique_bank),
                "bank_coverage_score": coverage,
                "version_space_size_before": before,
                "version_space_size_after": after,
                "information_gain_bits": math.log2(before) - math.log2(after),
                "oracle_entropy": oracle_entropy,
                "selected_true_entropy": selected_true_entropy,
                "entropy_regret": max(0.0, oracle_entropy - selected_true_entropy),
                "identified": after == 1,
            }
        )
    return ReferenceTrajectory(
        policy=policy,
        initial_state_id=state.state_id,
        replicate=replicate,
        initial_mode_count=state.valid_mode_count,
        steps=tuple(records),
    )


def summarize_reference_trajectories(
    trajectories: Iterable[ReferenceTrajectory],
    *,
    max_steps: int,
) -> dict[str, object]:
    items = list(trajectories)
    if not items:
        raise ValueError("at least one trajectory is required")
    endpoints = []
    carried_curves = []
    for trajectory in items:
        sizes = [float(trajectory.initial_mode_count)]
        sizes.extend(
            float(step["version_space_size_after"]) for step in trajectory.steps
        )
        sizes.extend([sizes[-1]] * (max_steps + 1 - len(sizes)))
        carried_curves.append(sizes)
        endpoints.append(trajectory_metrics(trajectory, max_steps=max_steps))
    curves = []
    for step in range(max_steps + 1):
        sizes = [curve[step] for curve in carried_curves]
        curves.append(
            {
                "step": step,
                "identification_success": statistics.fmean(
                    float(size == 1) for size in sizes
                ),
                "mean_log2_version_space_size": statistics.fmean(
                    math.log2(size) for size in sizes
                ),
            }
        )
    return {
        "trajectories": len(items),
        "identification_at_budget": statistics.fmean(
            float(row["identified"]) for row in endpoints
        ),
        "mean_experiments_used": statistics.fmean(
            float(row["experiments_used"]) for row in endpoints
        ),
        "mean_final_log2_version_space_size": statistics.fmean(
            float(row["final_log2_version_space_size"]) for row in endpoints
        ),
        "mean_cumulative_information_gain_bits": statistics.fmean(
            float(row["cumulative_information_gain_bits"]) for row in endpoints
        ),
        "mean_normalized_log_version_space_auc": statistics.fmean(
            float(row["normalized_log_version_space_auc"]) for row in endpoints
        ),
        "mean_entropy_regret": statistics.fmean(
            float(row["mean_entropy_regret"]) for row in endpoints
        ),
        "mean_bank_coverage_score": statistics.fmean(
            float(row["mean_bank_coverage_score"]) for row in endpoints
        ),
        "curves": curves,
    }


def trajectory_metrics(
    trajectory: ReferenceTrajectory,
    *,
    max_steps: int,
) -> dict[str, float | bool]:
    sizes = [float(trajectory.initial_mode_count)]
    sizes.extend(float(step["version_space_size_after"]) for step in trajectory.steps)
    sizes.extend([sizes[-1]] * (max_steps + 1 - len(sizes)))
    initial_log = math.log2(trajectory.initial_mode_count)
    identified_step = next(
        (int(step["step"]) for step in trajectory.steps if bool(step["identified"])),
        None,
    )
    step_coverages = [
        float(step["bank_coverage_score"])
        for step in trajectory.steps
        if str(step["selection_reason"]) != "oracle"
    ]
    return {
        "identified": sizes[-1] == 1,
        "experiments_used": float(identified_step or max_steps),
        "final_version_space_size": sizes[-1],
        "final_log2_version_space_size": math.log2(sizes[-1]),
        "cumulative_information_gain_bits": initial_log - math.log2(sizes[-1]),
        "normalized_log_version_space_auc": statistics.fmean(
            math.log2(size) / initial_log for size in sizes
        ),
        "mean_entropy_regret": statistics.fmean(
            float(step["entropy_regret"]) for step in trajectory.steps
        )
        if trajectory.steps
        else 0.0,
        "mean_bank_coverage_score": statistics.fmean(step_coverages)
        if step_coverages
        else 1.0,
    }
