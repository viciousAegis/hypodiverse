from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib

from scattered_discovery.envs.causal_micro_lab.dsl import Hypothesis
from scattered_discovery.envs.causal_micro_lab.enumerate_hypotheses import (
    enumerate_hypotheses,
)
from scattered_discovery.envs.causal_micro_lab.interventions import (
    Experiment,
    enumerate_experiments,
)
from scattered_discovery.envs.causal_micro_lab.simulator import (
    Outcome,
    Signature,
    pack_signature_bits,
    prediction_signature,
)


@dataclass(frozen=True)
class ModeRecord:
    mode_id: str
    signature: Signature
    canonical: Hypothesis
    syntactic_count: int
    family: tuple[str, str]


@dataclass(frozen=True)
class ModeTable:
    experiments: tuple[Experiment, ...]
    modes: tuple[ModeRecord, ...]
    modes_by_id: dict[str, ModeRecord]
    mode_index_by_id: dict[str, int]
    all_hypotheses_count: int


def mode_id_for_signature(signature: Signature) -> str:
    digest = hashlib.sha256(pack_signature_bits(signature).encode("ascii")).hexdigest()
    return digest


def mechanism_family(hypothesis: Hypothesis) -> tuple[str, str]:
    rule = hypothesis.y_rule
    inputs = set(rule.inputs)
    if rule.operator in {"COPY", "NOT"}:
        source = "UNARY"
    elif all(name.startswith("X") for name in inputs):
        source = "DIRECT"
    elif all(name.startswith("Z") for name in inputs):
        source = "MEDIATED"
    else:
        source = "MIXED"
    operator = "COPY_NOT" if rule.operator in {"COPY", "NOT"} else rule.operator
    return (source, operator)


@lru_cache(maxsize=1)
def build_mode_table() -> ModeTable:
    experiments = enumerate_experiments()
    grouped: dict[str, list[Hypothesis]] = {}
    signatures: dict[str, Signature] = {}
    hypotheses = enumerate_hypotheses()
    for hypothesis in hypotheses:
        signature = prediction_signature(hypothesis, experiments)
        mode_id = mode_id_for_signature(signature)
        grouped.setdefault(mode_id, []).append(hypothesis)
        signatures.setdefault(mode_id, signature)

    records: list[ModeRecord] = []
    for mode_id, programs in grouped.items():
        canonical = min(programs, key=lambda item: item.canonical_sort_key())
        records.append(
            ModeRecord(
                mode_id=mode_id,
                signature=signatures[mode_id],
                canonical=canonical,
                syntactic_count=len(programs),
                family=mechanism_family(canonical),
            )
        )
    modes = tuple(sorted(records, key=lambda record: record.mode_id))
    return ModeTable(
        experiments=experiments,
        modes=modes,
        modes_by_id={record.mode_id: record for record in modes},
        mode_index_by_id={record.mode_id: index for index, record in enumerate(modes)},
        all_hypotheses_count=len(hypotheses),
    )


def outcome_for_mode(mode: ModeRecord, experiment_id: int) -> Outcome:
    return mode.signature[experiment_id]
