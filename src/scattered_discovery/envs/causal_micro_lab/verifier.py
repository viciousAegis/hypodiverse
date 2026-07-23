from __future__ import annotations

from dataclasses import dataclass

from scattered_discovery.envs.causal_micro_lab.parser import (
    HypothesisParseError,
    parse_hypothesis,
    parse_hypothesis_set,
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


@dataclass(frozen=True)
class CandidateVerification:
    index: int
    text: str
    verification: VerificationResult

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "text": self.text,
            "verification": self.verification.as_dict(),
        }


@dataclass(frozen=True)
class SetVerificationResult:
    expected_count: int
    candidate_count: int
    format_valid: bool
    candidates: tuple[CandidateVerification, ...]
    valid_mode_ids: tuple[str, ...]
    unique_valid_mode_ids: tuple[str, ...]
    duplicate_valid_modes: int

    @property
    def any_valid(self) -> bool:
        return bool(self.unique_valid_mode_ids)

    @property
    def parse_valid_count(self) -> int:
        return sum(item.verification.parse_valid for item in self.candidates)

    @property
    def syntax_valid_count(self) -> int:
        return sum(item.verification.syntax_valid for item in self.candidates)

    @property
    def evidence_consistent_count(self) -> int:
        return sum(item.verification.evidence_consistent for item in self.candidates)

    @property
    def valid_count(self) -> int:
        return sum(item.verification.is_currently_valid_mode for item in self.candidates)

    def coverage_per_k(self) -> float:
        if self.expected_count <= 0:
            return 0.0
        return len(self.unique_valid_mode_ids) / self.expected_count

    def coverage_per_available(self, state: EvidenceState) -> float:
        denominator = min(self.expected_count, state.valid_mode_count)
        if denominator <= 0:
            return 0.0
        return len(self.unique_valid_mode_ids) / denominator

    def as_dict(self) -> dict[str, object]:
        return {
            "expected_count": self.expected_count,
            "candidate_count": self.candidate_count,
            "format_valid": self.format_valid,
            "parse_valid_count": self.parse_valid_count,
            "syntax_valid_count": self.syntax_valid_count,
            "evidence_consistent_count": self.evidence_consistent_count,
            "valid_count": self.valid_count,
            "valid_mode_ids": list(self.valid_mode_ids),
            "unique_valid_mode_ids": list(self.unique_valid_mode_ids),
            "duplicate_valid_modes": self.duplicate_valid_modes,
            "coverage_per_k": self.coverage_per_k(),
            "candidates": [item.as_dict() for item in self.candidates],
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


def verify_output_set(
    text: str,
    state: EvidenceState,
    *,
    expected_count: int,
    mode_table: ModeTable | None = None,
    strict: bool = True,
) -> SetVerificationResult:
    table = mode_table or build_mode_table()
    parsed_candidates = parse_hypothesis_set(text, expected_count=expected_count)
    candidate_indexes = {candidate.index for candidate in parsed_candidates}
    format_valid = (
        expected_count > 0
        and candidate_indexes == set(range(1, expected_count + 1))
        and all(candidate.text.strip() for candidate in parsed_candidates)
    )
    verified = tuple(
        CandidateVerification(
            index=candidate.index,
            text=candidate.text,
            verification=verify_output(
                candidate.text,
                state,
                mode_table=table,
                strict=strict,
            ),
        )
        for candidate in parsed_candidates
    )
    valid_mode_ids = tuple(
        item.verification.semantic_mode_id
        for item in verified
        if item.verification.is_currently_valid_mode
        and item.verification.semantic_mode_id is not None
    )
    unique_valid_mode_ids = tuple(dict.fromkeys(valid_mode_ids))
    return SetVerificationResult(
        expected_count=expected_count,
        candidate_count=len(parsed_candidates),
        format_valid=format_valid,
        candidates=verified,
        valid_mode_ids=valid_mode_ids,
        unique_valid_mode_ids=unique_valid_mode_ids,
        duplicate_valid_modes=len(valid_mode_ids) - len(unique_valid_mode_ids),
    )
