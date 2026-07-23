from __future__ import annotations

import inspect


def main() -> None:
    import transfer_queue  # noqa: F401
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
    from verl.trainer import main_ppo
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer
    from verl.workers.utils.padding import response_to_nested  # noqa: F401

    from scattered_discovery.verl.agent_loop import CDGRPOAgentLoop
    from scattered_discovery.verl.cd_grpo_trainer import (
        CDGRPOTrainerMixin,
        build_cd_task_runner,
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
    if not hasattr(PPOTrainer, "_compute_advantage"):
        failures.append("veRL v1 PPOTrainer._compute_advantage is missing")
    source = inspect.getsource(PPOTrainer._compute_advantage)
    for required in ("rm_scores", "response_mask", "kv_batch_get"):
        if required not in source:
            failures.append(
                f"veRL v1 advantage hook no longer references {required!r}"
            )
    if CDGRPOAgentLoop is None or CDGRPOTrainerMixin is None:
        failures.append("project CD-GRPO modules did not import")
    if failures:
        raise SystemExit(
            "CD-GRPO/veRL compatibility check failed:\n- "
            + "\n- ".join(failures)
        )

    runner = build_cd_task_runner()
    if runner is None:
        raise SystemExit("could not construct CD-GRPO TaskRunner")
    print("CD-GRPO veRL compatibility check passed.")


if __name__ == "__main__":
    main()
