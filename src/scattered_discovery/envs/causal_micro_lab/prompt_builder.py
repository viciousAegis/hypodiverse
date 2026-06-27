from __future__ import annotations

from scattered_discovery.envs.causal_micro_lab.state_generator import EvidenceState


def build_prompt(state: EvidenceState) -> str:
    lines = [
        "Infer one Boolean causal program consistent with the evidence.",
        "",
        "Variables: inputs X1,X2,X3; intermediates Z1,Z2; output Y.",
        "Rules: exactly one shallow rule for each target Z1, Z2, Y.",
        "Operators: COPY(A), NOT(A), AND(A,B), OR(A,B), XOR(A,B).",
        "Available inputs:",
        "- Z1 rule can use: X1, X2, X3",
        "- Z2 rule can use: X1, X2, X3, Z1",
        "- Y rule can use: X1, X2, X3, Z1, Z2",
        "For AND/OR/XOR, use two different inputs in the listed order.",
        "",
        "Evidence:",
        "",
    ]
    from scattered_discovery.envs.causal_micro_lab.signatures import build_mode_table

    table = build_mode_table()
    for index, item in enumerate(state.evidence, start=1):
        experiment = table.experiments[item.experiment_id]
        intervention = experiment.intervention
        if intervention == "OBSERVE":
            intervention_text = "none"
        elif intervention.startswith("DO_Z1_"):
            intervention_text = f"set Z1={intervention[-1]}"
        else:
            intervention_text = f"set Z2={intervention[-1]}"
        lines.extend(
            [
                f"Experiment {index}:",
                f"  inputs: X1={experiment.inputs[0]}, X2={experiment.inputs[1]}, X3={experiment.inputs[2]}",
                f"  intervention: {intervention_text}",
                f"  observed: Z1={item.outcome[0]}, Z2={item.outcome[1]}, Y={item.outcome[2]}",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "Return exactly three lines, no extra text:",
            "Z1 = OP(input[, input])",
            "Z2 = OP(input[, input])",
            "Y = OP(input[, input])",
        ]
    )
    return "\n".join(lines)
