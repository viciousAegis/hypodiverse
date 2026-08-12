"""Single-shot LIFPO rollout with a stable veRL metadata contract."""

from __future__ import annotations

from typing import Any

from scattered_discovery.verl.agent_loop import VerifiableHypothesisAgentLoop, register
from scattered_discovery.verl.lifpo_trainer import normalize_lifpo_reward_extra_info


@register("lifpo_agent_loop")
class LIFPOAgentLoop(VerifiableHypothesisAgentLoop):  # type: ignore[misc]
    """Generate one latent-conditioned hypothesis and attach LIFPO metadata."""

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> Any:
        output = await super().run(sampling_params, **kwargs)
        reward_extra_info = output.extra_fields.get("reward_extra_info", {})
        for key in (
            "reward_payload",
            "eval_payload",
            "reward_payload_json",
            "eval_payload_json",
        ):
            reward_extra_info.pop(key, None)
            output.extra_fields.pop(key, None)
        output.extra_fields["reward_extra_info"] = normalize_lifpo_reward_extra_info(
            reward_extra_info,
            reward_score=output.reward_score,
        )
        output.extra_fields.update(output.extra_fields["reward_extra_info"])
        return output
