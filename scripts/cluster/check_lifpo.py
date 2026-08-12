from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the LIFPO training path.")
    parser.add_argument("--run-config", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.run_config).read_text(encoding="utf-8"))
    assert config["trainer_entrypoint"] == "scattered_discovery.verl.lifpo_main"
    assert config["trainer_use_v1"] is True
    assert config["lifpo_latent_enabled"] is True
    assert int(config["lifpo_latent_count"]) >= 2
    assert config["lifpo_counterfactual_token_scope"] in {
        "answer",
        "full_response",
    }
    assert config["lifpo_counterfactual_reduction"] in {"mean", "sum"}
    alpha = float(config["lifpo_counterfactual_alpha"])
    clip = float(config["lifpo_counterfactual_clip"])
    frequency_credit = float(config["lifpo_frequency_credit_max"])
    valid_reward = float(config["causal_micro_lab_valid_hypothesis_reward"])
    assert alpha * clip + frequency_credit <= valid_reward

    import torch
    import transfer_queue
    from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ

    from scattered_discovery.verl.lifpo_agent_loop import (
        LIFPO_REWARD_EXTRA_DEFAULTS,
        LIFPOAgentLoop,
        normalize_lifpo_reward_extra_info,
    )
    from scattered_discovery.verl.lifpo_trainer import (
        LIFPOAgentLoopManager,
        LIFPOAgentLoopManagerTQ,
        LIFPOTrainerMixin,
        build_lifpo_task_runner,
        compute_lifpo_advantages,
    )

    assert LIFPOAgentLoop is not None
    assert LIFPOAgentLoopManager is not None
    assert LIFPOAgentLoopManagerTQ is not None
    assert LIFPOTrainerMixin is not None
    assert callable(build_lifpo_task_runner)
    assert hasattr(transfer_queue, "KVBatchMeta")
    assert hasattr(torch, "nested") and hasattr(torch.nested, "as_nested_tensor")
    assert set(normalize_lifpo_reward_extra_info({})) == set(
        LIFPO_REWARD_EXTRA_DEFAULTS
    )
    assert "session_id=i" in inspect.getsource(AgentLoopWorkerTQ._run_prompt).replace(
        " ", ""
    )

    advantages, _, metrics = compute_lifpo_advantages(
        token_level_rewards=torch.tensor([[1.0], [1.0], [0.2], [0.0]]),
        response_mask=torch.ones(4, 1),
        assigned_log_probs=torch.tensor([[-0.2], [-1.0], [-0.2], [-0.2]]),
        negative_log_probs=torch.tensor([[-1.0], [-0.2], [-1.0], [-1.0]]),
        index=["state"] * 4,
        metadata=[
            {
                "valid": valid,
                "valid_reward": 1.0 if valid else 0.0,
                "behavior": behavior,
                "latent_enabled": True,
                "latent_id": index + 1,
                "answer_token_count": 1,
            }
            for index, (valid, behavior) in enumerate(
                [(True, (1, 1)), (True, (2, 2)), (False, None), (False, None)]
            )
        ],
        epsilon=float(config["lifpo_frequency_epsilon"]),
        counterfactual_alpha=alpha,
        counterfactual_clip=clip,
        inverse_frequency_enabled=bool(config["lifpo_inverse_frequency_enabled"]),
        latent_count=4,
        frequency_credit_mode=str(config["lifpo_frequency_credit_mode"]),
        frequency_credit_max=frequency_credit,
        counterfactual_token_scope=str(config["lifpo_counterfactual_token_scope"]),
        counterfactual_reduction=str(config["lifpo_counterfactual_reduction"]),
        counterfactual_valid_only=bool(config["lifpo_counterfactual_valid_only"]),
    )
    assert torch.isfinite(advantages).all()
    assert metrics["lifpo/validity_rate"] == 0.5
    print("LIFPO preflight OK")


if __name__ == "__main__":
    main()
