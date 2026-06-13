from __future__ import annotations

from dataclasses import dataclass

from scattered_discovery.algos.grpo import GRPOAlgorithm


@dataclass(frozen=True)
class EchoGRPOAlgorithm(GRPOAlgorithm):
    name: str = "echo_grpo"
    description: str = (
        "GRPO plus auxiliary cross-entropy on environment observation tokens."
    )
    requires_custom_trainer: bool = True
    env_loss_coef: float = 0.05
    env_mask_field: str = "env_observation_mask"

    def verl_overrides(self) -> tuple[str, ...]:
        return (
            *super().verl_overrides(),
            "+discovery_algorithm.name=echo_grpo",
            "+discovery_algorithm.env_prediction.enabled=True",
            f"+discovery_algorithm.env_prediction.coef={self.env_loss_coef}",
            f"+discovery_algorithm.env_prediction.mask_field={self.env_mask_field}",
        )

    def notes(self) -> tuple[str, ...]:
        return (
            "Requires a trainer patch that consumes env-observation masks.",
            "The current DiscoveryAgentLoop must emit observation-token masks before this is active.",
            "Keep this as an explicit recipe so ECHO experiments are not confused with vanilla GRPO.",
        )
