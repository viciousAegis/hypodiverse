from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

Intervention = Literal["OBSERVE", "DO_Z1_0", "DO_Z1_1", "DO_Z2_0", "DO_Z2_1"]
INTERVENTIONS: tuple[Intervention, ...] = (
    "OBSERVE",
    "DO_Z1_0",
    "DO_Z1_1",
    "DO_Z2_0",
    "DO_Z2_1",
)


@dataclass(frozen=True)
class Experiment:
    experiment_id: int
    inputs: tuple[int, int, int]
    intervention: Intervention

    def inputs_dict(self) -> dict[str, int]:
        return {"X1": self.inputs[0], "X2": self.inputs[1], "X3": self.inputs[2]}

    def to_json(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "inputs": self.inputs_dict(),
            "intervention": self.intervention,
        }


def enumerate_experiments() -> tuple[Experiment, ...]:
    experiments: list[Experiment] = []
    experiment_id = 0
    for values in product((0, 1), repeat=3):
        for intervention in INTERVENTIONS:
            experiments.append(
                Experiment(
                    experiment_id=experiment_id,
                    inputs=tuple(values),
                    intervention=intervention,
                )
            )
            experiment_id += 1
    return tuple(experiments)
