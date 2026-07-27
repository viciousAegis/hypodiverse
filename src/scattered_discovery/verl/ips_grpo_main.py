from __future__ import annotations

import sys


def require_v1_cli(argv: list[str]) -> None:
    """Refuse to run when Hydra would select veRL's legacy task runner."""
    values = []
    for argument in argv:
        key, separator, value = argument.partition("=")
        if separator and key.lstrip("+") == "trainer.use_v1":
            values.append(value.strip().lower())
    if not values or values[-1] not in {"1", "true", "yes", "on"}:
        raise SystemExit(
            "IPS/latent GRPO requires trainer.use_v1=True. "
            "Use the corrected *_v1 Slurm launcher; refusing to run the "
            "standard GRPO task runner."
        )


def main() -> None:
    require_v1_cli(sys.argv[1:])

    from verl.trainer import main_ppo

    from scattered_discovery.verl.ips_grpo_trainer import build_ips_task_runner

    main_ppo.TaskRunnerV1 = build_ips_task_runner()
    main_ppo.main()


if __name__ == "__main__":
    main()
