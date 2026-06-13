from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VerlAlgorithm:
    """Base recipe for objectives launched through veRL.

    The environment layer owns prompts, interaction, and reward construction.
    Algorithm recipes own the veRL trainer overrides needed to optimize those
    rollouts. Objectives that require a trainer loss patch should mark
    `requires_custom_trainer=True` so launch scripts fail loudly until that
    patch is installed.
    """

    name: str
    description: str
    requires_custom_trainer: bool = False
    trainer_entrypoint: str = "verl.trainer.main_ppo"

    def verl_overrides(self) -> tuple[str, ...]:
        raise NotImplementedError

    def notes(self) -> tuple[str, ...]:
        return ()

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "requires_custom_trainer": self.requires_custom_trainer,
            "trainer_entrypoint": self.trainer_entrypoint,
            "verl_overrides": list(self.verl_overrides()),
            "notes": list(self.notes()),
        }
