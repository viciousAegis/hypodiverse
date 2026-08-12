"""Canonical LIFPO names for the historical trainer implementation.

The evaluated checkpoint predates the LIFPO name.  Keep the original module
importable for checkpoint and W&B provenance while exposing thesis-facing
names to new launchers and downstream users.
"""

from __future__ import annotations

from typing import Any

from scattered_discovery.verl.ips_grpo_trainer import (
    IPS_REWARD_EXTRA_DEFAULTS,
    IPSGRPOAgentLoopManager,
    IPSGRPOAgentLoopManagerTQ,
    IPSGRPOTrainerMixin,
    build_ips_task_runner,
    compute_latent_ips_grpo_advantages,
    normalize_ips_reward_extra_info,
)


LIFPOAgentLoopManager = IPSGRPOAgentLoopManager
LIFPOAgentLoopManagerTQ = IPSGRPOAgentLoopManagerTQ
LIFPOTrainerMixin = IPSGRPOTrainerMixin
LIFPO_REWARD_EXTRA_DEFAULTS = IPS_REWARD_EXTRA_DEFAULTS


def normalize_lifpo_reward_extra_info(
    value: Any,
    *,
    reward_score: Any = 0.0,
) -> dict[str, float]:
    """Return the fixed scalar rollout schema consumed by LIFPO."""
    return normalize_ips_reward_extra_info(value, reward_score=reward_score)


def compute_lifpo_advantages(**kwargs: Any) -> tuple[Any, Any, dict[str, float]]:
    """Compute the LIFPO advantages using the frozen evaluated objective."""
    kwargs.setdefault("metric_prefix", "lifpo")
    return compute_latent_ips_grpo_advantages(**kwargs)


def build_lifpo_task_runner() -> Any:
    """Build the veRL task runner used by LIFPO."""
    return build_ips_task_runner()
