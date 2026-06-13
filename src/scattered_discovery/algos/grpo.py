from __future__ import annotations

from dataclasses import dataclass

from scattered_discovery.algos.base import VerlAlgorithm


@dataclass(frozen=True)
class GRPOAlgorithm(VerlAlgorithm):
    name: str = "grpo"
    description: str = "Vanilla veRL GRPO over final scalar trajectory rewards."
    use_kl_in_reward: bool = False

    def verl_overrides(self) -> tuple[str, ...]:
        return (
            "algorithm.adv_estimator=grpo",
            f"algorithm.use_kl_in_reward={str(self.use_kl_in_reward)}",
        )

    def notes(self) -> tuple[str, ...]:
        return (
            "Compatible with single-answer protocol and pass@K via rollout.n=K.",
            "Set rewards are implemented by the environment protocol, not by changing GRPO.",
        )


@dataclass(frozen=True)
class SetRewardGRPOAlgorithm(GRPOAlgorithm):
    name: str = "set_reward_grpo"
    description: str = "GRPO with environment-level set COMMIT rewards. The veRL objective remains GRPO."

    def notes(self) -> tuple[str, ...]:
        return (
            "Use dataset EnvSpec protocol=set and max_commit=K.",
            "The trainer still receives one scalar reward per rollout.",
            "Use this for Puri-style single-rollout K-answer comparisons.",
        )
