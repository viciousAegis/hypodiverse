from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from scattered_discovery.envs.causal_micro_lab.signatures import (
    ModeTable,
    build_mode_table,
)
from scattered_discovery.envs.causal_micro_lab.state_generator import (
    EvidenceItem,
    EvidenceState,
    make_state,
)
from scattered_discovery.envs.causal_micro_lab.simulator import Outcome


@dataclass(frozen=True)
class ClosedLoopTrace:
    initial_state_id: str
    hidden_mode_id: str
    steps: tuple[dict[str, float | int | str | bool | None], ...]

    def final_version_space_size(self) -> int:
        if not self.steps:
            return 0
        return int(self.steps[-1]["remaining_version_space_size"])

    def identified(self) -> bool:
        return self.final_version_space_size() == 1


def outcome_entropy(outcomes: list[tuple[int, int, int]]) -> float:
    if not outcomes:
        return 0.0
    counts = Counter(outcomes)
    total = len(outcomes)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def available_experiment_ids(
    state: EvidenceState,
    *,
    mode_table: ModeTable | None = None,
) -> tuple[int, ...]:
    table = mode_table or build_mode_table()
    observed = set(state.observed_experiment_ids())
    return tuple(
        experiment.experiment_id
        for experiment in table.experiments
        if experiment.experiment_id not in observed
    )


def experiment_outcome_counts(
    mode_ids: Iterable[str],
    experiment_id: int,
    *,
    mode_table: ModeTable | None = None,
) -> Counter[Outcome]:
    table = mode_table or build_mode_table()
    return Counter(
        table.modes_by_id[mode_id].signature[experiment_id]
        for mode_id in mode_ids
        if mode_id in table.modes_by_id
    )


def experiment_entropy(
    mode_ids: Iterable[str],
    experiment_id: int,
    *,
    mode_table: ModeTable | None = None,
) -> float:
    counts = experiment_outcome_counts(
        mode_ids,
        experiment_id,
        mode_table=mode_table,
    )
    return outcome_entropy(
        [outcome for outcome, count in counts.items() for _ in range(count)]
    )


def prediction_distributions(
    mode_ids: list[str],
    state: EvidenceState,
    *,
    mode_table: ModeTable | None = None,
) -> dict[int, dict[str, object]]:
    table = mode_table or build_mode_table()
    distributions: dict[int, dict[str, object]] = {}
    for experiment_id in available_experiment_ids(state, mode_table=table):
        counts = experiment_outcome_counts(
            mode_ids,
            experiment_id,
            mode_table=table,
        )
        total = sum(counts.values())
        outcomes = {
            "".join(str(bit) for bit in outcome): {
                "count": count,
                "probability": count / total if total else 0.0,
            }
            for outcome, count in sorted(counts.items())
        }
        distributions[experiment_id] = {
            "entropy": outcome_entropy(
                [outcome for outcome, count in counts.items() for _ in range(count)]
            ),
            "outcomes": outcomes,
            "sample_count": total,
        }
    return distributions


def select_disagreement_experiment(
    generated_mode_ids: list[str],
    state: EvidenceState,
    *,
    mode_table: ModeTable | None = None,
) -> int | None:
    table = mode_table or build_mode_table()
    observed = set(state.observed_experiment_ids())
    modes = [table.modes_by_id[mode_id] for mode_id in generated_mode_ids if mode_id in table.modes_by_id]
    if not modes:
        return None
    best: tuple[float, int] | None = None
    for experiment in table.experiments:
        if experiment.experiment_id in observed:
            continue
        entropy = outcome_entropy(
            [mode.signature[experiment.experiment_id] for mode in modes]
        )
        candidate = (entropy, -experiment.experiment_id)
        if best is None or candidate > best:
            best = candidate
    return -best[1] if best is not None else None


def select_seeded_random_experiment(
    state: EvidenceState,
    *,
    seed: int,
    mode_table: ModeTable | None = None,
) -> int | None:
    available = available_experiment_ids(state, mode_table=mode_table)
    if not available:
        return None
    return random.Random(seed).choice(available)


def oracle_disagreement_experiment(
    state: EvidenceState, *, mode_table: ModeTable | None = None
) -> int | None:
    return select_disagreement_experiment(
        list(state.valid_mode_ids),
        state,
        mode_table=mode_table,
    )


def run_oracle_closed_loop(
    state: EvidenceState,
    *,
    max_steps: int = 8,
    mode_table: ModeTable | None = None,
) -> ClosedLoopTrace:
    table = mode_table or build_mode_table()
    if not state.hidden_mode_id:
        raise ValueError("closed-loop evaluation requires state.hidden_mode_id")
    hidden = table.modes_by_id[state.hidden_mode_id]
    current = state
    trace: list[dict[str, float | int | str | bool | None]] = []
    previous_count = current.valid_mode_count
    for step_index in range(max_steps):
        if current.valid_mode_count <= 1:
            break
        experiment_id = oracle_disagreement_experiment(current, mode_table=table)
        if experiment_id is None:
            break
        evidence = tuple(
            sorted(
                (
                    *current.evidence,
                    EvidenceItem(
                        experiment_id=experiment_id,
                        outcome=hidden.signature[experiment_id],
                    ),
                ),
                key=lambda item: item.experiment_id,
            )
        )
        current = make_state(
            hidden_mode=hidden,
            evidence_ids=tuple(item.experiment_id for item in evidence),
            mode_table=table,
            compute_separation=True,
        )
        remaining = current.valid_mode_count
        trace.append(
            {
                "step": step_index + 1,
                "experiment_id": experiment_id,
                "remaining_version_space_size": remaining,
                "information_gain_modes": previous_count - remaining,
                "identified": remaining == 1,
            }
        )
        previous_count = remaining
    return ClosedLoopTrace(
        initial_state_id=state.state_id,
        hidden_mode_id=state.hidden_mode_id,
        steps=tuple(trace),
    )
