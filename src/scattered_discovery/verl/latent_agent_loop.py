from __future__ import annotations

from typing import Any

from scattered_discovery.verl.agent_loop import CDGRPOAgentLoop, register


LATENT_REWARD_EXTRA_DEFAULTS = {
    "terminal_reward": 0.0,
    "base_terminal_reward": 0.0,
    "validity": 0.0,
    "parse_valid": 0.0,
    "evidence_consistent": 0.0,
    "response_length_raw": 0.0,
    "response_length_cap_hit": 0.0,
    "response_length_loss_masked": 0.0,
    "reward_length_cap": 0.0,
    "reward_syntax_valid": 0.0,
    "reward_valid_hypothesis": 0.0,
    "cd_probe_count": 0.0,
    "ips_behavior_hash_hi": -1.0,
    "ips_behavior_hash_lo": -1.0,
    "valid_mode_count": 0.0,
    "latent_enabled": 0.0,
    "latent_id": 0.0,
    "latent_negative_id": 0.0,
    "latent_answer_token_count": 0.0,
}


def normalize_latent_reward_extra_info(
    value: Any,
    *,
    reward_score: Any = 0.0,
) -> dict[str, float]:
    """Return the exact scalar schema expected by veRL batch assembly."""
    source = value if isinstance(value, dict) else {}
    defaults = dict(LATENT_REWARD_EXTRA_DEFAULTS)
    defaults["terminal_reward"] = float(reward_score or 0.0)
    defaults["base_terminal_reward"] = float(reward_score or 0.0)
    return {key: float(source.get(key, default)) for key, default in defaults.items()}


@register("latent_grpo_agent_loop")
class LatentGRPOAgentLoop(CDGRPOAgentLoop):  # type: ignore[misc]
    """Explicit latent rollout with a stable reward-extra contract."""

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> Any:
        output = await super().run(sampling_params, **kwargs)
        reward_extra_info = output.extra_fields.get("reward_extra_info", {})
        for key in (
            "cd_reward_payload",
            "cd_eval_payload",
            "cd_reward_payload_json",
            "cd_eval_payload_json",
        ):
            reward_extra_info.pop(key, None)
            output.extra_fields.pop(key, None)
        output.extra_fields["reward_extra_info"] = (
            normalize_latent_reward_extra_info(
                reward_extra_info,
                reward_score=output.reward_score,
            )
        )
        return output
