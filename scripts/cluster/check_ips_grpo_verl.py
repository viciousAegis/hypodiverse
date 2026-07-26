from __future__ import annotations

import argparse
import inspect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latent",
        action="store_true",
        help="also check the latent counterfactual-scoring path",
    )
    args = parser.parse_args()

    import torch
    import transfer_queue
    from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ

    from scattered_discovery.verl.agent_loop import (
        IPSGRPOAgentLoop,
        build_latent_prompt,
        latent_id_for_rollout,
        negative_latent_id,
    )
    from scattered_discovery.verl.ips_grpo_trainer import (
        IPS_REWARD_EXTRA_DEFAULTS,
        IPSGRPOAgentLoopManager,
        IPSGRPOAgentLoopManagerTQ,
        IPSGRPOAgentLoopWorker,
        IPSGRPOAgentLoopWorkerTQ,
        IPSGRPOTrainerMixin,
        build_ips_task_runner,
        compute_ips_grpo_advantages,
        compute_latent_ips_grpo_advantages,
        normalize_ips_reward_extra_info,
    )

    assert IPSGRPOAgentLoop is not None
    assert IPSGRPOTrainerMixin is not None
    assert IPSGRPOAgentLoopManager is not None
    assert IPSGRPOAgentLoopManagerTQ is not None
    assert (
        "_postprocess"
        in IPSGRPOAgentLoopWorker.__ray_metadata__.modified_class.__dict__
    )
    assert (
        "_agent_loop_postprocess"
        in IPSGRPOAgentLoopWorkerTQ.__ray_metadata__.modified_class.__dict__
    )
    assert callable(build_ips_task_runner)
    complete = normalize_ips_reward_extra_info(
        {
            "reward_syntax_valid": 0.2,
            "validity": 0.0,
            "ips_behavior_hash_hi": -1,
            "ips_behavior_hash_lo": -1,
        }
    )
    failure = normalize_ips_reward_extra_info({})
    assert set(complete) == set(failure) == set(IPS_REWARD_EXTRA_DEFAULTS)
    assert failure["reward_syntax_valid"] == 0.0
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
    run_prompt_source = inspect.getsource(AgentLoopWorkerTQ._run_prompt)
    assert "session_id=i" in run_prompt_source.replace(" ", ""), (
        "Installed veRL does not forward rollout session_id to agent loops; "
        "stratified rollout allocation is unavailable."
    )
    if args.latent:
        assert hasattr(transfer_queue, "KVBatchMeta")
        assert hasattr(torch, "nested") and hasattr(
            torch.nested,
            "as_nested_tensor",
        )
        assert "_compute_old_log_prob" in IPSGRPOTrainerMixin.__dict__
        scorer_source = inspect.getsource(IPSGRPOTrainerMixin._compute_old_log_prob)
        assert "latent_negative_log_probs" in scorer_source
        assert [latent_id_for_rollout(index, 8) for index in range(8)] == list(
            range(1, 9)
        )
        assert negative_latent_id(8, 8) == 1
        assert build_latent_prompt("prompt", 2).startswith("Strategy 2 |")
        latent_advantages, _, latent_metrics = compute_latent_ips_grpo_advantages(
            token_level_rewards=torch.tensor([[1.0], [1.0], [1.0], [0.2]]),
            response_mask=torch.ones(4, 1),
            assigned_log_probs=torch.tensor([[-1.0], [-1.0], [-0.1], [-0.1]]),
            negative_log_probs=torch.tensor([[-1.0], [-1.0], [-2.0], [-9.0]]),
            index=["state"] * 4,
            metadata=[
                {
                    "valid": True,
                    "valid_reward": 1.0,
                    "behavior": (1, 1),
                    "latent_enabled": True,
                    "latent_id": 1,
                    "answer_token_count": 1,
                },
                {
                    "valid": True,
                    "valid_reward": 1.0,
                    "behavior": (1, 1),
                    "latent_enabled": True,
                    "latent_id": 2,
                    "answer_token_count": 1,
                },
                {
                    "valid": True,
                    "valid_reward": 1.0,
                    "behavior": (2, 2),
                    "latent_enabled": True,
                    "latent_id": 3,
                    "answer_token_count": 1,
                },
                {
                    "valid": False,
                    "valid_reward": 0.0,
                    "behavior": None,
                    "latent_enabled": True,
                    "latent_id": 4,
                    "answer_token_count": 1,
                },
            ],
            epsilon=0.2,
            mi_alpha=0.1,
            mi_clip=1.0,
            use_ips=True,
            latent_count=4,
        )
        assert latent_advantages[2, 0] > latent_advantages[0, 0]
        assert latent_metrics["latent_ips/weight_max"] > 1.0
    print("IPS-GRPO veRL imports ok")


if __name__ == "__main__":
    main()
