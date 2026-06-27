from __future__ import annotations

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


class CausalMicroLabEnv:
    protocol = "single"

    def __init__(
        self,
        *,
        state: EvidenceState | dict[str, Any] | None = None,
        seed: int = 0,
        target_mode_count: int = 4,
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

    @property
    def done(self) -> bool:
        return self._done

    def system_prompt(self, runtime: Literal["local", "verl"] = "local") -> str:
        del runtime
        return "You are solving a single-shot scientific hypothesis generation task."

    def reset(self) -> str:
        return build_prompt(self.state)

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
        result = verify_output(model_text_or_action, self.state)
        if not result.parse_valid:
            self._parse_failures += 1
            self._invalid_actions += 1
        breakdown = RewardBreakdown(
            valid_hypothesis=1.0 if result.is_currently_valid_mode else 0.0
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
            "budget_used": 0,
            "parse_failures": self._parse_failures,
            "invalid_actions": self._invalid_actions,
        }
