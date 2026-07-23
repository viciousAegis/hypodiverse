from __future__ import annotations

import re
from typing import Any, Literal

from scattered_discovery.envs.base import DiscoveryScore, DiscoveryStep, RewardBreakdown
from scattered_discovery.envs.causal_micro_lab.parser import parse_record_state
from scattered_discovery.envs.causal_micro_lab.prompt_builder import build_prompt
from scattered_discovery.envs.causal_micro_lab.signatures import build_mode_table
from scattered_discovery.envs.causal_micro_lab.state_generator import (
    EvidenceState,
    find_states,
)
from scattered_discovery.envs.causal_micro_lab.verifier import verify_output
from scattered_discovery.envs.causal_micro_lab.verifier import verify_output_set


def _default_state(seed: int = 0, target_mode_count: int = 4) -> EvidenceState:
    table = build_mode_table()
    for offset in range(len(table.modes)):
        states = find_states(
            (seed + offset) % len(table.modes),
            target_mode_count,
            max_evidence=8,
            beam_width=128,
            mode_table=table,
        )
        if states:
            return states[0]
    raise RuntimeError(f"could not find causal micro-lab state with M={target_mode_count}")


def _has_rule_markers(text: str) -> bool:
    return all(
        re.search(rf"^\s*{target}\s*(?::|=|:=)", text, re.IGNORECASE | re.MULTILINE)
        for target in ("Z1", "Z2", "Y")
    )


class CausalMicroLabEnv:
    protocol = "single"

    def __init__(
        self,
        *,
        state: EvidenceState | dict[str, Any] | None = None,
        seed: int = 0,
        target_mode_count: int = 4,
        nonempty_output_reward: float = 0.2,
        rule_marker_reward: float = 0.0,
        parse_valid_reward: float = 0.0,
        syntax_valid_reward: float = 0.0,
        evidence_consistent_reward: float = 0.0,
        valid_hypothesis_reward: float = 1.0,
        output_mode: Literal["single", "multi_answer_rlvr"] = "single",
        answer_count: int = 1,
        multi_answer_format_reward: float = 0.5,
        multi_answer_accuracy_reward: float = 0.5,
        multi_answer_accuracy_mode: Literal["any_valid", "unique_valid_per_k"] = "any_valid",
    ) -> None:
        if isinstance(state, EvidenceState):
            self.state = state
        elif isinstance(state, dict):
            self.state = parse_record_state(state)
        else:
            self.state = _default_state(seed=seed, target_mode_count=target_mode_count)
        self._done = False
        self._last_score: DiscoveryScore | None = None
        self._invalid_actions = 0
        self._parse_failures = 0
        self._nonempty_output_reward = float(nonempty_output_reward)
        self._rule_marker_reward = float(rule_marker_reward)
        self._parse_valid_reward = float(parse_valid_reward)
        self._syntax_valid_reward = float(syntax_valid_reward)
        self._evidence_consistent_reward = float(evidence_consistent_reward)
        self._valid_hypothesis_reward = float(valid_hypothesis_reward)
        self._output_mode = output_mode
        self._answer_count = max(1, int(answer_count))
        self._multi_answer_format_reward = float(multi_answer_format_reward)
        self._multi_answer_accuracy_reward = float(multi_answer_accuracy_reward)
        self._multi_answer_accuracy_mode = multi_answer_accuracy_mode

    @property
    def done(self) -> bool:
        return self._done

    def system_prompt(self, runtime: Literal["local", "verl"] = "local") -> str:
        del runtime
        return "You are solving a single-shot scientific hypothesis generation task."

    def reset(self) -> str:
        return build_prompt(
            self.state,
            output_mode=self._output_mode,
            answer_count=self._answer_count,
        )

    def observation_prompt(
        self,
        step: DiscoveryStep,
        runtime: Literal["local", "verl"] = "local",
    ) -> str:
        del runtime
        return step.observation

    def step(self, model_text_or_action: str) -> DiscoveryStep:
        if self._done:
            return DiscoveryStep(
                observation="Episode is already done.",
                done=True,
                parse_ok=False,
                score=self._last_score,
            )
        if self._output_mode == "multi_answer_rlvr":
            return self._step_multi_answer_rlvr(model_text_or_action)
        has_final_output = bool(model_text_or_action.strip())
        result = verify_output(model_text_or_action, self.state)
        if not result.parse_valid:
            self._parse_failures += 1
            self._invalid_actions += 1
        nonempty_bonus = (
            self._nonempty_output_reward
            if has_final_output and not result.is_currently_valid_mode
            else 0.0
        )
        rule_marker_bonus = (
            self._rule_marker_reward
            if _has_rule_markers(model_text_or_action)
            and not result.is_currently_valid_mode
            else 0.0
        )
        parse_bonus = (
            self._parse_valid_reward
            if result.parse_valid and not result.is_currently_valid_mode
            else 0.0
        )
        syntax_bonus = (
            self._syntax_valid_reward
            if result.syntax_valid and not result.is_currently_valid_mode
            else 0.0
        )
        evidence_bonus = (
            self._evidence_consistent_reward
            if result.evidence_consistent and not result.is_currently_valid_mode
            else 0.0
        )
        breakdown = RewardBreakdown(
            valid_hypothesis=self._valid_hypothesis_reward
            if result.is_currently_valid_mode
            else 0.0,
            nonempty_output=nonempty_bonus,
            format=rule_marker_bonus,
            admissible=parse_bonus,
            commit_format=syntax_bonus,
            clean_invalid_final=evidence_bonus,
        )
        score = DiscoveryScore(
            reward=breakdown.total,
            breakdown=breakdown,
            valid_keys=(result.semantic_mode_id,)
            if result.is_currently_valid_mode and result.semantic_mode_id
            else (),
            valid_committed_count=1 if result.is_currently_valid_mode else 0,
            valid_unique_count=1 if result.is_currently_valid_mode else 0,
            committed_count=1,
            false_count=0 if result.is_currently_valid_mode else 1,
            parse_failures=self._parse_failures,
            invalid_actions=self._invalid_actions,
            metrics={
                "parse_valid": float(result.parse_valid),
                "syntax_valid": float(result.syntax_valid),
                "evidence_consistent": float(result.evidence_consistent),
                "nonempty_output": float(has_final_output),
                "rule_markers": float(_has_rule_markers(model_text_or_action)),
                "final_version_space_size": self.state.valid_mode_count,
                "current_version_space_size": self.state.valid_mode_count,
                "recovery": 1.0 / self.state.valid_mode_count
                if result.is_currently_valid_mode and self.state.valid_mode_count
                else 0.0,
                "valid_mode_count": self.state.valid_mode_count,
                "evidence_size": self.state.evidence_size,
            },
            reward_vector=tuple(
                1.0 if mode_id == result.semantic_mode_id and result.is_currently_valid_mode else 0.0
                for mode_id in self.state.valid_mode_ids
            ),
        )
        self._done = True
        self._last_score = score
        return DiscoveryStep(
            observation=(
                "Episode complete. "
                f"parse_valid={result.parse_valid}; "
                f"evidence_consistent={result.evidence_consistent}; "
                f"valid_mode={result.is_currently_valid_mode}."
            ),
            done=True,
            parse_ok=result.parse_valid,
            action_text=model_text_or_action,
            reward=score.reward,
            score=score,
            metrics=score.metrics,
            debug={"verification": result.as_dict()},
        )

    def _step_multi_answer_rlvr(self, model_text_or_action: str) -> DiscoveryStep:
        has_final_output = bool(model_text_or_action.strip())
        result = verify_output_set(
            model_text_or_action,
            self.state,
            expected_count=self._answer_count,
        )
        parse_failures = self._answer_count - result.parse_valid_count
        self._parse_failures += max(0, parse_failures)
        self._invalid_actions += max(0, self._answer_count - result.valid_count)
        if self._multi_answer_accuracy_mode == "unique_valid_per_k":
            accuracy_fraction = result.coverage_per_k()
        else:
            accuracy_fraction = 1.0 if result.any_valid else 0.0
        format_reward = self._multi_answer_format_reward if result.format_valid else 0.0
        accuracy_reward = self._multi_answer_accuracy_reward * accuracy_fraction
        breakdown = RewardBreakdown(
            valid_hypothesis=accuracy_reward,
            format=format_reward,
        )
        coverage_per_available = result.coverage_per_available(self.state)
        valid_keys = result.unique_valid_mode_ids
        score = DiscoveryScore(
            reward=breakdown.total,
            breakdown=breakdown,
            valid_keys=valid_keys,
            valid_committed_count=result.valid_count,
            valid_unique_count=len(valid_keys),
            committed_count=result.candidate_count,
            false_count=max(0, result.candidate_count - result.valid_count),
            duplicate_count=result.duplicate_valid_modes,
            parse_failures=self._parse_failures,
            invalid_actions=self._invalid_actions,
            metrics={
                "parse_valid": float(result.parse_valid_count == self._answer_count),
                "syntax_valid": float(result.syntax_valid_count == self._answer_count),
                "evidence_consistent": float(result.evidence_consistent_count > 0),
                "nonempty_output": float(has_final_output),
                "rule_markers": float(_has_rule_markers(model_text_or_action)),
                "final_version_space_size": self.state.valid_mode_count,
                "current_version_space_size": self.state.valid_mode_count,
                "recovery": coverage_per_available,
                "valid_mode_count": self.state.valid_mode_count,
                "evidence_size": self.state.evidence_size,
                "multi_answer_expected_count": self._answer_count,
                "multi_answer_candidate_count": result.candidate_count,
                "multi_answer_format_valid": float(result.format_valid),
                "multi_answer_parse_valid_count": result.parse_valid_count,
                "multi_answer_syntax_valid_count": result.syntax_valid_count,
                "multi_answer_valid_count": result.valid_count,
                "multi_answer_any_valid": float(result.any_valid),
                "multi_answer_unique_valid_modes": len(valid_keys),
                "multi_answer_duplicate_valid_modes": result.duplicate_valid_modes,
                "multi_answer_coverage_per_k": result.coverage_per_k(),
                "multi_answer_coverage_per_available": coverage_per_available,
                "multi_answer_accuracy_fraction": accuracy_fraction,
            },
            reward_vector=tuple(
                1.0 if mode_id in set(valid_keys) else 0.0
                for mode_id in self.state.valid_mode_ids
            ),
        )
        self._done = True
        self._last_score = score
        return DiscoveryStep(
            observation=(
                "Episode complete. "
                f"format_valid={result.format_valid}; "
                f"valid_modes={len(valid_keys)}; "
                f"any_valid={result.any_valid}."
            ),
            done=True,
            parse_ok=result.format_valid,
            action_text=model_text_or_action,
            reward=score.reward,
            score=score,
            metrics=score.metrics,
            debug={"verification_set": result.as_dict()},
        )

    def force_finalize(self) -> DiscoveryScore:
        breakdown = RewardBreakdown()
        score = DiscoveryScore(
            reward=0.0,
            breakdown=breakdown,
            committed_count=0,
            parse_failures=self._parse_failures,
            invalid_actions=self._invalid_actions,
            metrics={
                "final_version_space_size": self.state.valid_mode_count,
                "current_version_space_size": self.state.valid_mode_count,
                "recovery": 0.0,
                "valid_mode_count": self.state.valid_mode_count,
                "evidence_size": self.state.evidence_size,
            },
        )
        self._done = True
        self._last_score = score
        return score

    def diagnostics(self) -> dict[str, Any]:
        return {
            "env_type": "causal_micro_lab",
            "state_id": self.state.state_id,
            "hidden_mode_id": self.state.hidden_mode_id,
            "valid_mode_ids": list(self.state.valid_mode_ids),
            "valid_mode_count": self.state.valid_mode_count,
            "evidence_size": self.state.evidence_size,
            "separation_bucket": self.state.separation_bucket,
            "family_bucket": self.state.family_bucket,
            "output_mode": self._output_mode,
            "answer_count": self._answer_count,
            "budget_used": 0,
            "parse_failures": self._parse_failures,
            "invalid_actions": self._invalid_actions,
        }
