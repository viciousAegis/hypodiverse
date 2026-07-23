from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from random import Random
from typing import Any

from scattered_discovery.envs.causal_micro_lab.interventions import (
    Experiment,
    enumerate_experiments,
)
from scattered_discovery.envs.causal_micro_lab.parser import (
    HypothesisParseError,
    parse_hypothesis,
)
from scattered_discovery.envs.causal_micro_lab.simulator import (
    pack_signature_bits,
    prediction_signature,
    run_experiment,
)


class CandidateStatus(str, Enum):
    TRUNCATED = "truncated"
    PARSE_FAIL = "parse_fail"
    INVALID = "invalid"
    VALID = "valid"


@dataclass(frozen=True)
class VisibleEvidence:
    state_id: str
    observed: tuple[tuple[int, tuple[int, int, int]], ...]
    available_experiment_ids: tuple[int, ...]


@dataclass(frozen=True)
class ConsequenceResult:
    status: CandidateStatus
    state_id: str
    evidence_consistent: bool
    probe_experiment_ids: tuple[int, ...]
    consequence_signature: str | None
    behavior_key: str | None
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.status is CandidateStatus.VALID

    def reward_fields(self) -> dict[str, object]:
        return {
            "cd_status": self.status.value,
            "cd_state_id": self.state_id,
            "cd_evidence_consistent": float(self.evidence_consistent),
            "cd_probe_count": float(len(self.probe_experiment_ids)),
            "cd_consequence_signature": self.consequence_signature or "",
            "cd_behavior_key": self.behavior_key or "",
        }


def parse_visible_evidence(record: dict[str, Any]) -> VisibleEvidence:
    state_id = str(record["state_id"])
    experiments = enumerate_experiments()
    observed: list[tuple[int, tuple[int, int, int]]] = []
    seen: set[int] = set()
    for item in record.get("visible_experiments") or ():
        experiment_id = int(item["experiment_id"])
        if experiment_id < 0 or experiment_id >= len(experiments):
            raise ValueError(f"invalid experiment_id {experiment_id}")
        if experiment_id in seen:
            raise ValueError(f"duplicate experiment_id {experiment_id}")
        seen.add(experiment_id)
        outcome = item["observation"]
        observed.append(
            (
                experiment_id,
                (
                    int(outcome["Z1"]),
                    int(outcome["Z2"]),
                    int(outcome["Y"]),
                ),
            )
        )

    advertised = record.get("available_experiment_ids")
    if advertised is None:
        available = tuple(
            experiment.experiment_id
            for experiment in experiments
            if experiment.experiment_id not in seen
        )
    else:
        available = tuple(sorted(int(item) for item in advertised))
        expected = {
            experiment.experiment_id
            for experiment in experiments
            if experiment.experiment_id not in seen
        }
        if set(available) != expected:
            raise ValueError("available experiments do not complement visible evidence")

    return VisibleEvidence(
        state_id=state_id,
        observed=tuple(sorted(observed)),
        available_experiment_ids=available,
    )


def deterministic_probe_ids(
    evidence: VisibleEvidence,
    *,
    probe_fraction: float = 1.0,
) -> tuple[int, ...]:
    if not 0.0 < probe_fraction <= 1.0:
        raise ValueError("probe_fraction must be in (0, 1]")
    available = evidence.available_experiment_ids
    if probe_fraction == 1.0 or not available:
        return available
    size = max(1, math.ceil(probe_fraction * len(available)))
    seed_bytes = hashlib.sha256(
        f"{evidence.state_id}:probe".encode("ascii")
    ).digest()[:8]
    rng = Random(int.from_bytes(seed_bytes, "big"))
    return tuple(sorted(rng.sample(list(available), size)))


def _experiments_by_id() -> dict[int, Experiment]:
    return {
        experiment.experiment_id: experiment
        for experiment in enumerate_experiments()
    }


def evaluate_consequences(
    text: str,
    state_record: dict[str, Any],
    *,
    truncated: bool = False,
    probe_fraction: float = 1.0,
) -> ConsequenceResult:
    evidence = parse_visible_evidence(state_record)
    if truncated:
        return ConsequenceResult(
            status=CandidateStatus.TRUNCATED,
            state_id=evidence.state_id,
            evidence_consistent=False,
            probe_experiment_ids=(),
            consequence_signature=None,
            behavior_key=None,
        )

    try:
        hypothesis = parse_hypothesis(text, strict=True)
    except HypothesisParseError as exc:
        return ConsequenceResult(
            status=CandidateStatus.PARSE_FAIL,
            state_id=evidence.state_id,
            evidence_consistent=False,
            probe_experiment_ids=(),
            consequence_signature=None,
            behavior_key=None,
            error=str(exc),
        )

    experiments = _experiments_by_id()
    consistent = all(
        run_experiment(hypothesis, experiments[experiment_id]) == outcome
        for experiment_id, outcome in evidence.observed
    )
    if not consistent:
        return ConsequenceResult(
            status=CandidateStatus.INVALID,
            state_id=evidence.state_id,
            evidence_consistent=False,
            probe_experiment_ids=(),
            consequence_signature=None,
            behavior_key=None,
        )

    probe_ids = deterministic_probe_ids(
        evidence,
        probe_fraction=probe_fraction,
    )
    probes = tuple(experiments[experiment_id] for experiment_id in probe_ids)
    packed = pack_signature_bits(prediction_signature(hypothesis, probes))
    behavior_key = hashlib.sha256(packed.encode("ascii")).hexdigest()
    return ConsequenceResult(
        status=CandidateStatus.VALID,
        state_id=evidence.state_id,
        evidence_consistent=True,
        probe_experiment_ids=probe_ids,
        consequence_signature=packed,
        behavior_key=behavior_key,
    )
