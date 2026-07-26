from __future__ import annotations

from pathlib import Path

import yaml

from scattered_discovery.verl.latent_agent_loop import (
    LATENT_REWARD_EXTRA_DEFAULTS,
    LatentGRPOAgentLoop,
    normalize_latent_reward_extra_info,
)


def main() -> None:
    config_path = Path("configs/verl/latent_grpo_agent_loop.yaml")
    entries = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_target = (
        "scattered_discovery.verl.latent_agent_loop.LatentGRPOAgentLoop"
    )
    assert entries == [
        {
            "name": "causal_micro_lab_agent_loop",
            "_target_": expected_target,
        },
        {
            "name": "latent_grpo_agent_loop",
            "_target_": expected_target,
        }
    ]
    assert LatentGRPOAgentLoop is not None

    success = normalize_latent_reward_extra_info(
        {
            "reward_syntax_valid": 0.2,
            "reward_valid_hypothesis": 1.0,
            "latent_id": 1,
        },
        reward_score=1.0,
    )
    sparse = normalize_latent_reward_extra_info({}, reward_score=0.0)
    assert set(success) == set(LATENT_REWARD_EXTRA_DEFAULTS)
    assert set(sparse) == set(LATENT_REWARD_EXTRA_DEFAULTS)
    assert sparse["reward_syntax_valid"] == 0.0

    print(
        "latent GRPO preflight OK: run-local agent routing and fixed "
        f"{len(LATENT_REWARD_EXTRA_DEFAULTS)}-field reward schema"
    )


if __name__ == "__main__":
    main()
