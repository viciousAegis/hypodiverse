from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
import unittest

from scattered_discovery.envs.causal_micro_lab.consequence_diversity import (
    BehaviorArchive,
    DiversityCandidate,
    diversity_rewards,
)
from scattered_discovery.envs.causal_micro_lab.consequence_reward import (
    CandidateStatus,
    deterministic_probe_ids,
    evaluate_consequences,
    parse_visible_evidence,
)
from scattered_discovery.envs.causal_micro_lab.interventions import (
    enumerate_experiments,
)
from scattered_discovery.envs.causal_micro_lab.parser import parse_hypothesis
from scattered_discovery.envs.causal_micro_lab.simulator import run_experiment


VALID_PROGRAM = """\
Z1: COPY X1
Z2: COPY X2
Y: XOR Z1 Z2
"""

INVALID_PROGRAM = """\
Z1: COPY X1
Z2: COPY X2
Y: NOT X3
"""


def visible_state_record() -> dict[str, object]:
    experiments = enumerate_experiments()
    hypothesis = parse_hypothesis(VALID_PROGRAM)
    observed_id = 0
    outcome = run_experiment(hypothesis, experiments[observed_id])
    return {
        "state_id": "test-state",
        "visible_experiments": [
            {
                "experiment_id": observed_id,
                "inputs": experiments[observed_id].inputs_dict(),
                "intervention": experiments[observed_id].intervention,
                "observation": {
                    "Z1": outcome[0],
                    "Z2": outcome[1],
                    "Y": outcome[2],
                },
            }
        ],
        "available_experiment_ids": list(range(1, 40)),
        "private": {
            "valid_mode_ids": ["must-not-be-read"],
            "hidden_mode_id": "must-not-be-read",
        },
    }


class ConsequenceRewardTests(unittest.TestCase):
    def test_visible_candidate_pipeline(self):
        state = visible_state_record()
        valid = evaluate_consequences(VALID_PROGRAM, state)
        invalid = evaluate_consequences(INVALID_PROGRAM, state)
        malformed = evaluate_consequences("not a program", state)
        truncated = evaluate_consequences(VALID_PROGRAM, state, truncated=True)

        self.assertEqual(valid.status, CandidateStatus.VALID)
        self.assertEqual(len(valid.consequence_signature or ""), 39 * 3)
        self.assertEqual(invalid.status, CandidateStatus.INVALID)
        self.assertEqual(malformed.status, CandidateStatus.PARSE_FAIL)
        self.assertEqual(truncated.status, CandidateStatus.TRUNCATED)
        self.assertIsNone(truncated.behavior_key)

    def test_probe_subset_is_state_deterministic(self):
        evidence = parse_visible_evidence(visible_state_record())
        first = deterministic_probe_ids(evidence, probe_fraction=0.25)
        second = deterministic_probe_ids(evidence, probe_fraction=0.25)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)

    def test_reward_module_has_no_oracle_imports(self):
        path = (
            Path(__file__).parents[1]
            / "src/scattered_discovery/envs/causal_micro_lab/consequence_reward.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        forbidden = {"verifier", "signatures", "state_generator", "tables"}
        self.assertFalse(
            any(any(part in name for part in forbidden) for name in imported)
        )


class ConsequenceDiversityTests(unittest.TestCase):
    def _candidate(self, key: str, signature: str) -> DiversityCandidate:
        return DiversityCandidate(
            state_id="state",
            status=CandidateStatus.VALID,
            consequence_signature=signature,
            behavior_key=key,
        )

    def test_duplicate_tax_and_distance_credit(self):
        candidates = [
            self._candidate("a", "000000"),
            self._candidate("a", "000000"),
            self._candidate("a", "000000"),
            self._candidate("a", "000000"),
            self._candidate("b", "111111"),
        ]
        rewards, diagnostics = diversity_rewards(candidates, variant="logdet")
        self.assertLess(rewards[0], rewards[4])
        self.assertEqual(rewards[:4], [rewards[0]] * 4)
        self.assertEqual(diagnostics.unique_behaviors, 2)
        self.assertEqual(diagnostics.duplicate_valid_completions, 3)

    def test_archive_scaling_decay_and_roundtrip(self):
        archive = BehaviorArchive()
        candidates = [
            self._candidate("a", "000000"),
            self._candidate("b", "111111"),
        ]
        first, _ = diversity_rewards(candidates, archive=archive)
        second, _ = diversity_rewards(candidates, archive=archive)
        self.assertLess(second[0], first[0])
        self.assertEqual(archive.count("state", "a"), 2.0)
        archive.decay(0.5)
        self.assertEqual(archive.count("state", "a"), 1.0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.json"
            archive.save(path)
            restored = BehaviorArchive.load(path)
        self.assertEqual(restored.to_dict(), archive.to_dict())

    def test_invalid_candidates_never_enter_archive(self):
        archive = BehaviorArchive()
        candidates = [
            DiversityCandidate(
                state_id="state",
                status=CandidateStatus.INVALID,
                consequence_signature=None,
                behavior_key=None,
            ),
            self._candidate("a", "000"),
        ]
        rewards, diagnostics = diversity_rewards(candidates, archive=archive)
        self.assertEqual(rewards, [0.0, 0.0])
        self.assertTrue(diagnostics.skipped)
        self.assertEqual(archive.counts, {("state", "a"): 1.0})


class CDGRPOConfigTests(unittest.TestCase):
    def test_cluster_yaml_selects_cd_agent_and_entrypoint(self):
        import yaml

        path = (
            Path(__file__).parents[1]
            / "configs/verl/runs/causal_micro_lab_blackwell_cd_grpo_smoke.yaml"
        )
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(config["causal_micro_lab_agent_name"], "cd_grpo_agent_loop")
        self.assertEqual(config["default_agent_loop"], "cd_grpo_agent_loop")
        self.assertEqual(
            config["trainer_entrypoint"],
            "scattered_discovery.verl.cd_grpo_main",
        )
        self.assertEqual(config["rollout_n"], 16)
        self.assertEqual(config["total_training_steps"], 2)

    def test_archive_json_is_plain_and_stable(self):
        archive = BehaviorArchive(counts={("s", "b"): 2.0})
        encoded = json.dumps(archive.to_dict(), sort_keys=True)
        self.assertIn('"counts": [["s", "b", 2.0]]', encoded)


if __name__ == "__main__":
    unittest.main()
