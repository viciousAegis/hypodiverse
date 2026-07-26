from __future__ import annotations


def main() -> None:
    import torch
    import transfer_queue  # noqa: F401
    from scattered_discovery.verl.agent_loop import IPSGRPOAgentLoop
    from scattered_discovery.verl.ips_grpo_trainer import (
        IPSGRPOTrainerMixin,
        build_ips_task_runner,
        compute_ips_grpo_advantages,
    )

    assert IPSGRPOAgentLoop is not None
    assert IPSGRPOTrainerMixin is not None
    assert callable(build_ips_task_runner)
    advantages, _, metrics = compute_ips_grpo_advantages(
        token_level_rewards=torch.tensor([[1.0], [1.0], [1.0], [0.2]]),
        response_mask=torch.ones(4, 1),
        index=["state"] * 4,
        metadata=[
            {"valid": True, "valid_reward": 1.0, "behavior": (1, 1)},
            {"valid": True, "valid_reward": 1.0, "behavior": (1, 1)},
            {"valid": True, "valid_reward": 1.0, "behavior": (2, 2)},
            {"valid": False, "valid_reward": 0.0, "behavior": None},
        ],
        epsilon=0.2,
    )
    assert advantages[2, 0] > advantages[0, 0] > advantages[3, 0]
    assert metrics["ips_grpo/weight_max"] == 4.0
    print("IPS-GRPO veRL imports ok")


if __name__ == "__main__":
    main()
