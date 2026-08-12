"""Canonical LIFPO agent-loop registration."""

from __future__ import annotations

from scattered_discovery.verl.agent_loop import register
from scattered_discovery.verl.latent_agent_loop import (
    LATENT_REWARD_EXTRA_DEFAULTS,
    LatentGRPOAgentLoop,
    normalize_latent_reward_extra_info,
)


LIFPO_REWARD_EXTRA_DEFAULTS = LATENT_REWARD_EXTRA_DEFAULTS


def normalize_lifpo_reward_extra_info(
    value: object,
    *,
    reward_score: object = 0.0,
) -> dict[str, float]:
    """Return the fixed scalar schema expected by LIFPO batch assembly."""
    return normalize_latent_reward_extra_info(value, reward_score=reward_score)


@register("lifpo_agent_loop")
class LIFPOAgentLoop(LatentGRPOAgentLoop):  # type: ignore[misc]
    """LIFPO rollout loop; behavior is identical to the evaluated checkpoint."""
