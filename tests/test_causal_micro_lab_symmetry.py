from __future__ import annotations

import unittest

from scattered_discovery.envs.causal_micro_lab.dsl import Hypothesis, Rule
from scattered_discovery.envs.causal_micro_lab.enumerate_hypotheses import (
    enumerate_hypotheses,
)
from scattered_discovery.envs.causal_micro_lab.interventions import (
    Experiment,
    enumerate_experiments,
)
from scattered_discovery.envs.causal_micro_lab.prompt_builder import build_prompt
from scattered_discovery.envs.causal_micro_lab.simulator import run_experiment
from scattered_discovery.envs.causal_micro_lab.state_generator import (
    EvidenceItem,
    EvidenceState,
)
from scattered_discovery.envs.causal_micro_lab.symmetry import (
    PromptSymmetry,
    symmetry_schedule,
)


def _state() -> EvidenceState:
    hypothesis = Hypothesis(
        Rule("Z1", "AND", ("X1", "X3")),
        Rule("Z2", "OR", ("X2", "Z1")),
        Rule("Y", "XOR", ("X3", "Z2")),
    )
    experiments = enumerate_experiments()
    evidence_ids = (0, 17, 34)
    return EvidenceState(
        state_id="symmetry-test-state",
        hidden_mode_id="hidden",
        evidence=tuple(
            EvidenceItem(
                experiment_id=experiment_id,
                outcome=run_experiment(hypothesis, experiments[experiment_id]),
            )
            for experiment_id in evidence_ids
        ),
        valid_mode_ids=("mode-a",),
        mean_separation=0.0,
        minimum_separation=0.0,
        maximum_separation=0.0,
        separation_bucket="low",
        family_bucket="within_family",
    )


class PromptSymmetryTests(unittest.TestCase):
    def test_hypothesis_transform_is_invertible(self) -> None:
        transforms = symmetry_schedule(
            state_id="state",
            evidence_count=3,
            group_size=8,
        )
        hypotheses = enumerate_hypotheses()
        sample = (
            hypotheses[0],
            hypotheses[137],
            hypotheses[7_981],
            hypotheses[-1],
        )
        for transform in transforms:
            for hypothesis in sample:
                self.assertEqual(
                    transform.inverse_hypothesis(
                        transform.transform_hypothesis(hypothesis)
                    ),
                    hypothesis,
                )

    def test_transformed_program_preserves_experiment_outcomes(self) -> None:
        hypothesis = Hypothesis(
            Rule("Z1", "NOT", ("X2",)),
            Rule("Z2", "AND", ("X1", "Z1")),
            Rule("Y", "OR", ("X3", "Z2")),
        )
        transform = PromptSymmetry(("X3", "X1", "X2"))
        transformed = transform.transform_hypothesis(hypothesis)
        for experiment in enumerate_experiments():
            transformed_experiment = Experiment(
                experiment_id=experiment.experiment_id,
                inputs=transform.transform_inputs(experiment.inputs),
                intervention=experiment.intervention,
            )
            self.assertEqual(
                run_experiment(hypothesis, experiment),
                run_experiment(transformed, transformed_experiment),
            )

    def test_schedule_is_deterministic_and_covers_all_variable_permutations(
        self,
    ) -> None:
        first = symmetry_schedule(
            state_id="state-a",
            evidence_count=5,
            group_size=8,
            seed=9,
        )
        second = symmetry_schedule(
            state_id="state-a",
            evidence_count=5,
            group_size=8,
            seed=9,
        )
        self.assertEqual(first, second)
        self.assertEqual(first[0].exogenous_permutation, ("X1", "X2", "X3"))
        self.assertEqual(
            len({item.exogenous_permutation for item in first[:6]}),
            6,
        )
        self.assertIsNotNone(first[6].evidence_order)
        self.assertIsNotNone(first[7].evidence_order)

    def test_transformed_prompt_reorders_evidence_without_private_fields(self) -> None:
        state = _state()
        transform = PromptSymmetry(
            ("X2", "X3", "X1"),
            evidence_order=(2, 0, 1),
        )
        prompt = build_prompt(state, symmetry=transform)
        self.assertNotIn("hidden", prompt)
        self.assertNotIn("mode-a", prompt)
        self.assertLess(
            prompt.index("intervention: set Z2=1"),
            prompt.index("intervention: none"),
        )
        self.assertIn("inputs: X1=0, X2=1, X3=1", prompt)

    def test_invalid_transform_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PromptSymmetry(("X1", "X1", "X3"))
        with self.assertRaises(ValueError):
            PromptSymmetry(evidence_order=(0, 2))


if __name__ == "__main__":
    unittest.main()
