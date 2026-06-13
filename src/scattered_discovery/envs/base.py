from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class RewardBreakdown:
    valid_hypothesis: float = 0.0
    format: float = 0.0
    admissible: float = 0.0
    commit_format: float = 0.0
    invalid_action: float = 0.0
    false_commit: float = 0.0
    non_final_commit: float = 0.0
    unsupported_commit: float = 0.0
    duplicate_commit: float = 0.0
    budget: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.valid_hypothesis
            + self.format
            + self.admissible
            + self.commit_format
            + self.invalid_action
            + self.false_commit
            + self.non_final_commit
            + self.unsupported_commit
            + self.duplicate_commit
            + self.budget
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "valid_hypothesis": self.valid_hypothesis,
            "format": self.format,
            "admissible": self.admissible,
            "commit_format": self.commit_format,
            "invalid_action": self.invalid_action,
            "false_commit": self.false_commit,
            "non_final_commit": self.non_final_commit,
            "unsupported_commit": self.unsupported_commit,
            "duplicate_commit": self.duplicate_commit,
            "budget": self.budget,
            "total": self.total,
        }

    def plus(self, **updates: float) -> "RewardBreakdown":
        values = self.as_dict()
        values.pop("total", None)
        for key, value in updates.items():
            values[key] = values.get(key, 0.0) + value
        return RewardBreakdown(**values)


@dataclass(frozen=True)
class DiscoveryScore:
    reward: float
    breakdown: RewardBreakdown
    valid_keys: tuple[str, ...] = ()
    valid_branch_ids: tuple[int, ...] = ()
    valid_committed_count: int = 0
    valid_unique_count: int = 0
    committed_count: int = 0
    false_count: int = 0
    non_final_count: int = 0
    unsupported_count: int = 0
    duplicate_count: int = 0
    parse_failures: int = 0
    invalid_actions: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    reward_vector: tuple[float, ...] = ()

    @property
    def validity(self) -> float:
        if self.committed_count == 0:
            return 0.0
        return self.valid_committed_count / self.committed_count

    @property
    def uniqueness(self) -> float:
        if self.committed_count == 0:
            return 0.0
        return (self.committed_count - self.duplicate_count) / self.committed_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "breakdown": self.breakdown.as_dict(),
            "valid_keys": list(self.valid_keys),
            "valid_branch_ids": list(self.valid_branch_ids),
            "valid_committed_count": self.valid_committed_count,
            "valid_unique_count": self.valid_unique_count,
            "committed_count": self.committed_count,
            "false_count": self.false_count,
            "non_final_count": self.non_final_count,
            "unsupported_count": self.unsupported_count,
            "duplicate_count": self.duplicate_count,
            "validity": self.validity,
            "uniqueness": self.uniqueness,
            "parse_failures": self.parse_failures,
            "invalid_actions": self.invalid_actions,
            "metrics": self.metrics,
            "reward_vector": list(self.reward_vector),
        }


@dataclass(frozen=True)
class DiscoveryStep:
    observation: str
    done: bool
    parse_ok: bool
    action_text: str | None = None
    reward: float | None = None
    score: DiscoveryScore | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvSpec:
    env_type: str
    task: dict[str, Any]
    protocol: str = "single"
    max_steps: int = 8
    max_commit: int = 1
    seed: int = 0


class DiscoveryEnv(Protocol):
    protocol: str

    def system_prompt(self, runtime: Literal["local", "verl"] = "local") -> str: ...

    def reset(self) -> str: ...

    def observation_prompt(
        self,
        step: DiscoveryStep,
        runtime: Literal["local", "verl"] = "local",
    ) -> str: ...

    def step(self, model_text_or_action: str) -> DiscoveryStep: ...

    def force_finalize(self) -> DiscoveryScore: ...

    def diagnostics(self) -> dict[str, Any]: ...

    @property
    def done(self) -> bool: ...
