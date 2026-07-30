from __future__ import annotations

from typing import Literal

from scattered_discovery.envs.causal_micro_lab.state_generator import EvidenceState


def build_latent_prompt(prompt: str, latent_id: int) -> str:
    """Condition generation on a semantically neutral strategy identifier."""
    if latent_id < 1:
        raise ValueError("latent_id must be positive")
    return f"Strategy {latent_id} | {prompt}"


def build_prompt(
    state: EvidenceState,
    *,
    output_mode: Literal[
        "single",
        "multi_answer_rlvr",
        "verbalized_sampling",
    ] = "single",
    answer_count: int = 1,
) -> str:
    uniform_probability = (
        f"{1.0 / answer_count:.6f}".rstrip("0").rstrip(".")
        if answer_count > 0
        else "0"
    )
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
        "For AND/OR/XOR, use two different inputs. Do not nest operators.",
        "",
        "Evidence:",
        "",
    ]
    from scattered_discovery.envs.causal_micro_lab.signatures import build_mode_table

    table = build_mode_table()
    for index, item in enumerate(state.evidence, start=1):
        experiment = table.experiments[item.experiment_id]
        inputs = experiment.inputs
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
                f"  inputs: X1={inputs[0]}, X2={inputs[1]}, X3={inputs[2]}",
                f"  intervention: {intervention_text}",
                f"  observed: Z1={item.outcome[0]}, Z2={item.outcome[1]}, Y={item.outcome[2]}",
                "",
            ]
        )
    if output_mode == "verbalized_sampling":
        lines.extend(
            [
                "",
                f"Return exactly {answer_count} different candidate hypotheses.",
                "Assign each candidate a probability between 0 and 1.",
                "The probabilities must sum to 1.",
                "Every rule must begin with an operator.",
                "Use COPY for a direct input; use prefix order for binary operators.",
                "Use exactly this plain format, repeated in order:",
                "ANSWER 1",
                "Z1: AND X1 X2",
                "Z2: COPY Z1",
                "Y: XOR X3 Z2",
                f"PROBABILITY: {uniform_probability}",
                "",
                f"Continue through ANSWER {answer_count}.",
                "Return no other final-answer text.",
            ]
        )
    elif output_mode == "multi_answer_rlvr":
        lines.extend(
            [
                "",
                f"Return exactly {answer_count} candidate hypotheses.",
                f"Use exactly these tags: <answer1>...</answer1> through <answer{answer_count}>...</answer{answer_count}>.",
                "Inside each answer tag, return exactly three rule lines and no extra text.",
                "Use this flat format inside each tag:",
                "Z1: OP input [input]",
                "Z2: OP input [input]",
                "Y: OP input [input]",
                "Example:",
                "<answer1>",
                "Z1: AND X1 X2",
                "Z2: OR X2 Z1",
                "Y: XOR X3 Z1",
                "</answer1>",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Return exactly three lines, no extra text.",
                "Use this flat format:",
                "Z1: OP input [input]",
                "Z2: OP input [input]",
                "Y: OP input [input]",
                "Example:",
                "Z1: AND X1 X2",
                "Z2: OR X2 Z1",
                "Y: XOR X3 Z1",
            ]
        )
    return "\n".join(lines)
