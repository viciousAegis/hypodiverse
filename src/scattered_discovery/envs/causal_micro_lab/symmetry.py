from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import permutations

from scattered_discovery.envs.causal_micro_lab.dsl import (
    BINARY_OPERATORS,
    EXOGENOUS,
    PARENTS,
    Hypothesis,
    Rule,
    make_hypothesis,
)

ExogenousPermutation = tuple[str, str, str]


def _canonical_inputs(
    target: str,
    operator: str,
    inputs: tuple[str, ...],
) -> tuple[str, ...]:
    if operator not in BINARY_OPERATORS:
        return inputs
    allowed = PARENTS[target]
    return tuple(sorted(inputs, key=allowed.index))


@dataclass(frozen=True)
class PromptSymmetry:
    """An invertible nuisance transformation of a micro-lab prompt."""

    exogenous_permutation: ExogenousPermutation = EXOGENOUS
    evidence_order: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if set(self.exogenous_permutation) != set(EXOGENOUS):
            raise ValueError("exogenous_permutation must be a permutation of X1,X2,X3")
        if len(self.exogenous_permutation) != len(EXOGENOUS):
            raise ValueError("exogenous_permutation must contain exactly three names")
        if self.evidence_order is not None:
            expected = tuple(range(len(self.evidence_order)))
            if tuple(sorted(self.evidence_order)) != expected:
                raise ValueError("evidence_order must be a zero-based permutation")

    @property
    def name_map(self) -> dict[str, str]:
        return dict(zip(EXOGENOUS, self.exogenous_permutation, strict=True))

    @property
    def inverse_name_map(self) -> dict[str, str]:
        return {target: source for source, target in self.name_map.items()}

    @property
    def transform_id(self) -> str:
        names = "".join(name[-1] for name in self.exogenous_permutation)
        if self.evidence_order is None:
            order = "natural"
        else:
            order = "-".join(str(index) for index in self.evidence_order)
        return f"x{names}:e{order}"

    def order_evidence(self, evidence: tuple[object, ...]) -> tuple[object, ...]:
        if self.evidence_order is None:
            return evidence
        if len(self.evidence_order) != len(evidence):
            raise ValueError(
                "evidence_order length must equal the state's evidence count"
            )
        return tuple(evidence[index] for index in self.evidence_order)

    def transform_inputs(self, inputs: tuple[int, int, int]) -> tuple[int, int, int]:
        canonical_values = dict(zip(EXOGENOUS, inputs, strict=True))
        transformed_values = {
            self.name_map[name]: value for name, value in canonical_values.items()
        }
        return tuple(transformed_values[name] for name in EXOGENOUS)

    def transform_hypothesis(self, hypothesis: Hypothesis) -> Hypothesis:
        return self._map_hypothesis(hypothesis, self.name_map)

    def inverse_hypothesis(self, hypothesis: Hypothesis) -> Hypothesis:
        return self._map_hypothesis(hypothesis, self.inverse_name_map)

    @staticmethod
    def _map_hypothesis(
        hypothesis: Hypothesis,
        mapping: dict[str, str],
    ) -> Hypothesis:
        rules: list[Rule] = []
        for rule in hypothesis.rules:
            inputs = tuple(mapping.get(name, name) for name in rule.inputs)
            rules.append(
                Rule(
                    target=rule.target,
                    operator=rule.operator,
                    inputs=_canonical_inputs(rule.target, rule.operator, inputs),
                )
            )
        return make_hypothesis(rules)


def _evidence_permutation(size: int, variant: int) -> tuple[int, ...] | None:
    if size <= 1 or variant == 0:
        return None
    indices = tuple(range(size))
    if variant % 2:
        return tuple(reversed(indices))
    shift = (variant // 2) % size
    return indices[shift:] + indices[:shift]


def symmetry_schedule(
    *,
    state_id: str,
    evidence_count: int,
    group_size: int,
    seed: int = 1,
) -> tuple[PromptSymmetry, ...]:
    """Return a deterministic, balanced transform schedule for one state."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if evidence_count < 0:
        raise ValueError("evidence_count must be non-negative")

    exogenous = list(permutations(EXOGENOUS))
    identity = tuple(EXOGENOUS)
    exogenous.remove(identity)
    digest = hashlib.sha256(f"{seed}:{state_id}".encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big") % len(exogenous)
    exogenous = exogenous[offset:] + exogenous[:offset]
    ordered: list[ExogenousPermutation] = [
        identity,
        *(tuple(item) for item in exogenous),
    ]

    schedule: list[PromptSymmetry] = []
    for rollout_index in range(group_size):
        cycle, permutation_index = divmod(rollout_index, len(ordered))
        schedule.append(
            PromptSymmetry(
                exogenous_permutation=ordered[permutation_index],
                evidence_order=_evidence_permutation(evidence_count, cycle),
            )
        )
    return tuple(schedule)
