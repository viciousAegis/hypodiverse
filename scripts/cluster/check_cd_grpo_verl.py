from __future__ import annotations

import inspect


def main() -> None:
    import torch
    import transfer_queue  # noqa: F401
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
    from verl.trainer import main_ppo
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer
    from verl.workers.utils.padding import response_to_nested  # noqa: F401

    from scattered_discovery.envs.causal_micro_lab.consequence_diversity import (
        BehaviorArchive,
    )
    from scattered_discovery.verl.agent_loop import CDGRPOAgentLoop
    from scattered_discovery.verl.cd_grpo_trainer import (
        CDGRPOTrainerMixin,
        build_cd_task_runner,
        compute_cd_grpo_advantages,
    )

    failures = []
    if not hasattr(main_ppo, "TaskRunnerV1"):
        failures.append("verl.trainer.main_ppo.TaskRunnerV1 is missing")
    if not hasattr(AgentLoopOutput, "as_dict"):
        failures.append("AgentLoopOutput.as_dict is missing")
    model_fields = getattr(
        AgentLoopOutput,
        "model_fields",
        getattr(AgentLoopOutput, "__fields__", {}),
    )
    if "extra_fields" not in model_fields:
        failures.append("AgentLoopOutput.extra_fields is missing")
    output_source = inspect.getsource(AgentLoopOutput.as_dict)
    if (
        'output["rollout_log_probs"]' not in output_source
        or 'output.pop("response_logprobs"' not in output_source
    ):
        failures.append(
            "AgentLoopOutput no longer maps response_logprobs to rollout_log_probs"
        )
    if not hasattr(PPOTrainer, "_compute_advantage"):
        failures.append("veRL v1 PPOTrainer._compute_advantage is missing")
    agent_source = inspect.getsource(CDGRPOAgentLoop.run)
    if "response_logprobs=response_logprobs" not in agent_source:
        failures.append(
            "CDGRPOAgentLoop does not propagate generation log probabilities"
        )
    source = inspect.getsource(PPOTrainer._compute_advantage)
    for required in ("rm_scores", "response_mask", "kv_batch_get"):
        if required not in source:
            failures.append(f"veRL v1 advantage hook no longer references {required!r}")
    if CDGRPOAgentLoop is None or CDGRPOTrainerMixin is None:
        failures.append("project CD-GRPO modules did not import")
    if failures:
        raise SystemExit(
            "CD-GRPO/veRL compatibility check failed:\n- " + "\n- ".join(failures)
        )

    runner = build_cd_task_runner()
    if runner is None:
        raise SystemExit("could not construct CD-GRPO TaskRunner")

    payloads = [
        {
            "state_id": "preflight-state",
            "status": "valid",
            "consequence_signature": signature,
            "behavior_key": key,
        }
        for signature, key in (("0000", "a"), ("0000", "a"), ("1111", "b"))
    ]
    advantages, _, metrics = compute_cd_grpo_advantages(
        token_level_rewards=torch.ones((3, 1)),
        response_mask=torch.ones((3, 1)),
        index=["preflight-group"] * 3,
        payloads=payloads,
        eval_payloads=None,
        config={"beta": 0.3, "archive": True},
        archive=BehaviorArchive(),
    )
    required_metrics = {
        "cd_grpo/groups_with_2plus_unique_valid_rate",
        "cd_grpo/diversity_signal_active_rate",
        "cd_grpo/pairwise_consequence_distance_mean",
        "cd_grpo/diversity_contribution_abs_mean",
        "cd_grpo/archive_new_unique_behavior_rate",
        "cd_grpo/all_truncated_group_rate",
    }
    missing_metrics = required_metrics - metrics.keys()
    if missing_metrics:
        raise SystemExit(
            "CD-GRPO metric preflight is missing: " + ", ".join(sorted(missing_metrics))
        )
    if advantages.shape != (3, 1):
        raise SystemExit(
            f"CD-GRPO advantage preflight returned shape {advantages.shape}"
        )
    if metrics["cd_grpo/diversity_signal_active_rate"] != 1.0:
        raise SystemExit("CD-GRPO diversity signal did not activate in preflight")

    print("CD-GRPO veRL compatibility check passed.")


if __name__ == "__main__":
    main()
