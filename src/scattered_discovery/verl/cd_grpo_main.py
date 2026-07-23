from __future__ import annotations


def main() -> None:
    from verl.trainer import main_ppo

    from scattered_discovery.verl.cd_grpo_trainer import build_cd_task_runner

    main_ppo.TaskRunnerV1 = build_cd_task_runner()
    main_ppo.main()


if __name__ == "__main__":
    main()
