from __future__ import annotations

from scattered_discovery.envs.causal_micro_lab.dsl import Hypothesis, Rule
from scattered_discovery.envs.causal_micro_lab.interventions import Experiment

Outcome = tuple[int, int, int]
Signature = tuple[Outcome, ...]


def evaluate_rule(rule: Rule, values: dict[str, int]) -> int:
    inputs = [int(values[name]) for name in rule.inputs]
    if rule.operator == "COPY":
        return inputs[0]
    if rule.operator == "NOT":
        return 1 - inputs[0]
    if rule.operator == "AND":
        return int(inputs[0] and inputs[1])
    if rule.operator == "OR":
        return int(inputs[0] or inputs[1])
    if rule.operator == "XOR":
        return inputs[0] ^ inputs[1]
    raise ValueError(f"unsupported operator {rule.operator!r}")


def run_experiment(hypothesis: Hypothesis, experiment: Experiment) -> Outcome:
    values = experiment.inputs_dict()
    if experiment.intervention == "DO_Z1_0":
        values["Z1"] = 0
    elif experiment.intervention == "DO_Z1_1":
        values["Z1"] = 1
    else:
        values["Z1"] = evaluate_rule(hypothesis.z1_rule, values)

    if experiment.intervention == "DO_Z2_0":
        values["Z2"] = 0
    elif experiment.intervention == "DO_Z2_1":
        values["Z2"] = 1
    else:
        values["Z2"] = evaluate_rule(hypothesis.z2_rule, values)
    values["Y"] = evaluate_rule(hypothesis.y_rule, values)
    return (values["Z1"], values["Z2"], values["Y"])


def prediction_signature(
    hypothesis: Hypothesis, experiments: tuple[Experiment, ...]
) -> Signature:
    return tuple(run_experiment(hypothesis, experiment) for experiment in experiments)


def pack_signature_bits(signature: Signature) -> str:
    return "".join(str(bit) for outcome in signature for bit in outcome)
