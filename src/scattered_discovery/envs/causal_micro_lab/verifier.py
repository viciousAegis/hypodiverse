from __future__ import annotations

from dataclasses import dataclass

from scattered_discovery.envs.causal_micro_lab.parser import (
    HypothesisParseError,
    parse_hypothesis,
)
from scattered_discovery.envs.causal_micro_lab.signatures import (
    ModeTable,
    build_mode_table,
    mechanism_family,
    mode_id_for_signature,
)
from scattered_discovery.envs.causal_micro_lab.simulator import (
    pack_signature_bits,
    prediction_signature,
)
from scattered_discovery.envs.causal_micro_lab.state_generator import EvidenceState


@dataclass(frozen=True)
class VerificationResult:
    parse_valid: bool
    syntax_valid: bool
    evidence_consistent: bool
    semantic_mode_id: str | None
    is_currently_valid_mode: bool
    prediction_signature: str | None
    mechanism_family: tuple[str, str] | None
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "parse_valid": self.parse_valid,
            "syntax_valid": self.syntax_valid,
            "evidence_consistent": self.evidence_consistent,
            "semantic_mode_id": self.semantic_mode_id,
            "is_currently_valid_mode": self.is_currently_valid_mode,
            "prediction_signature": self.prediction_signature,
            "mechanism_family": list(self.mechanism_family)
            if self.mechanism_family
            else None,
            "error": self.error,
        }


def verify_output(
    text: str,
    state: EvidenceState,
    *,
    mode_table: ModeTable | None = None,
    strict: bool = True,
) -> VerificationResult:
    table = mode_table or build_mode_table()
    try:
        hypothesis = parse_hypothesis(text, strict=strict)
    except HypothesisParseError as exc:
        return VerificationResult(
            parse_valid=False,
            syntax_valid=False,
            evidence_consistent=False,
            semantic_mode_id=None,
            is_currently_valid_mode=False,
            prediction_signature=None,
            mechanism_family=None,
            error=str(exc),
        )
    signature = prediction_signature(hypothesis, table.experiments)
    mode_id = mode_id_for_signature(signature)
    consistent = all(
        signature[item.experiment_id] == item.outcome for item in state.evidence
    )
    return VerificationResult(
        parse_valid=True,
        syntax_valid=True,
        evidence_consistent=consistent,
        semantic_mode_id=mode_id,
        is_currently_valid_mode=mode_id in set(state.valid_mode_ids),
        prediction_signature=pack_signature_bits(signature),
        mechanism_family=mechanism_family(hypothesis),
    )
