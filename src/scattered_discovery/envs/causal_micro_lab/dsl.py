from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Operator = Literal["COPY", "NOT", "AND", "OR", "XOR"]

UNARY_OPERATORS = ("COPY", "NOT")
BINARY_OPERATORS = ("AND", "OR", "XOR")
OPERATORS = UNARY_OPERATORS + BINARY_OPERATORS
TARGETS = ("Z1", "Z2", "Y")
EXOGENOUS = ("X1", "X2", "X3")
PARENTS: dict[str, tuple[str, ...]] = {
    "Z1": ("X1", "X2", "X3"),
    "Z2": ("X1", "X2", "X3", "Z1"),
    "Y": ("X1", "X2", "X3", "Z1", "Z2"),
}


class CausalMicroLabError(ValueError):
    pass


@dataclass(frozen=True)
class Rule:
    target: str
    operator: Operator
    inputs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "operator", self.operator.upper())
        object.__setattr__(self, "inputs", tuple(self.inputs))
        validate_rule(self)

    def to_json(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "operator": self.operator,
            "inputs": list(self.inputs),
        }

    def render(self) -> str:
        return f"{self.target} = {self.operator}({', '.join(self.inputs)})"


@dataclass(frozen=True)
class Hypothesis:
    z1_rule: Rule
    z2_rule: Rule
    y_rule: Rule

    def __post_init__(self) -> None:
        targets = [rule.target for rule in self.rules]
        if targets != list(TARGETS):
            raise CausalMicroLabError(
                "hypothesis rules must be ordered as Z1, Z2, Y with no duplicates"
            )

    @property
    def rules(self) -> tuple[Rule, Rule, Rule]:
        return (self.z1_rule, self.z2_rule, self.y_rule)

    def rule_for(self, target: str) -> Rule:
        if target == "Z1":
            return self.z1_rule
        if target == "Z2":
            return self.z2_rule
        if target == "Y":
            return self.y_rule
        raise CausalMicroLabError(f"unknown target {target!r}")

    def to_json(self) -> dict[str, Any]:
        return {"rules": [rule.to_json() for rule in self.rules]}

    def render_json(self) -> str:
        import json

        return json.dumps(self.to_json(), sort_keys=False, indent=2)

    def render_rules(self) -> str:
        return "\n".join(rule.render() for rule in self.rules)

    def total_inputs(self) -> int:
        return sum(len(rule.inputs) for rule in self.rules)

    def canonical_sort_key(self) -> tuple[int, tuple[tuple[str, str, tuple[str, ...]], ...]]:
        return (
            self.total_inputs(),
            tuple((rule.target, rule.operator, rule.inputs) for rule in self.rules),
        )


def validate_rule(rule: Rule) -> None:
    if rule.target not in TARGETS:
        raise CausalMicroLabError(f"invalid target {rule.target!r}")
    if rule.operator not in OPERATORS:
        raise CausalMicroLabError(f"invalid operator {rule.operator!r}")
    allowed = PARENTS[rule.target]
    if rule.operator in UNARY_OPERATORS:
        if len(rule.inputs) != 1:
            raise CausalMicroLabError(f"{rule.operator} requires exactly one input")
    elif len(rule.inputs) != 2:
        raise CausalMicroLabError(f"{rule.operator} requires exactly two inputs")
    if any(input_name not in allowed for input_name in rule.inputs):
        bad = sorted(set(rule.inputs) - set(allowed))
        raise CausalMicroLabError(
            f"{rule.target} cannot use parent(s): {', '.join(bad)}"
        )
    if rule.operator in BINARY_OPERATORS:
        if len(set(rule.inputs)) != 2:
            raise CausalMicroLabError("binary inputs must be distinct")
        if tuple(rule.inputs) != tuple(sorted(rule.inputs, key=allowed.index)):
            raise CausalMicroLabError("binary inputs must be in canonical parent order")


def make_hypothesis(rules: list[Rule] | tuple[Rule, ...]) -> Hypothesis:
    if len(rules) != 3:
        raise CausalMicroLabError("hypothesis must contain exactly three rules")
    by_target: dict[str, Rule] = {}
    for rule in rules:
        if rule.target in by_target:
            raise CausalMicroLabError(f"duplicate target {rule.target!r}")
        by_target[rule.target] = rule
    missing = [target for target in TARGETS if target not in by_target]
    if missing:
        raise CausalMicroLabError(f"missing target(s): {', '.join(missing)}")
    return Hypothesis(by_target["Z1"], by_target["Z2"], by_target["Y"])
