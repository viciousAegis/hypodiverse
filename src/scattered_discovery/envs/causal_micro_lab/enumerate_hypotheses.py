from __future__ import annotations

from itertools import combinations

from scattered_discovery.envs.causal_micro_lab.dsl import (
    BINARY_OPERATORS,
    PARENTS,
    TARGETS,
    UNARY_OPERATORS,
    Hypothesis,
    Rule,
)


def enumerate_rules(target: str) -> tuple[Rule, ...]:
    parents = PARENTS[target]
    rules: list[Rule] = []
    for operator in UNARY_OPERATORS:
        for parent in parents:
            rules.append(Rule(target=target, operator=operator, inputs=(parent,)))
    for operator in BINARY_OPERATORS:
        for left, right in combinations(parents, 2):
            rules.append(Rule(target=target, operator=operator, inputs=(left, right)))
    return tuple(rules)


def enumerate_hypotheses() -> tuple[Hypothesis, ...]:
    z1_rules = enumerate_rules(TARGETS[0])
    z2_rules = enumerate_rules(TARGETS[1])
    y_rules = enumerate_rules(TARGETS[2])
    return tuple(
        Hypothesis(z1_rule, z2_rule, y_rule)
        for z1_rule in z1_rules
        for z2_rule in z2_rules
        for y_rule in y_rules
    )
