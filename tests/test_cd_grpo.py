from __future__ import annotations

import ast
from dataclasses import dataclass
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
    base_candidate_reward,
    deterministic_probe_ids,
    evaluate_consequences,
    parse_visible_evidence,
)
from scattered_discovery.verl.cd_grpo_trainer import (
    compute_cd_grpo_advantages,
    sanitize_validation_reward_extras,
    scatter_real_rows,
    select_cd_payloads,
)
from scattered_discovery.verl.agent_loop import (
    CDGRPOAgentLoop,
    _generation_log_probs,
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

        self.assertEqual(base_candidate_reward(valid), (1.0, 0.0, 1.0))
        self.assertEqual(base_candidate_reward(invalid), (0.2, 0.2, 0.0))
        self.assertEqual(base_candidate_reward(malformed), (0.0, 0.0, 0.0))
        self.assertEqual(base_candidate_reward(truncated), (0.0, 0.0, 0.0))

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
    def test_cd_run_yamls_recompute_old_log_probs(self):
        import yaml

        paths = sorted(
            (Path(__file__).parents[1] / "configs/verl/runs").glob(
                "causal_micro_lab_*_cd_grpo*.yaml"
            )
        )
        self.assertEqual(len(paths), 5)
        for path in paths:
            with self.subTest(path=path.name):
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    config["trainer_entrypoint"],
                    "scattered_discovery.verl.cd_grpo_main",
                )
                self.assertEqual(config["rollout_n"], 16)
                self.assertEqual(config["rollout_calculate_log_probs"], "False")

        launcher = (
            Path(__file__).parents[1]
            / "scripts/cluster/run_verl_discovery_grpo.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "agent_loop_manager_class="
            "scattered_discovery.verl.cd_grpo_trainer."
            "CDGRPOAgentLoopManagerTQ",
            launcher,
        )

        smoke_path = next(
            path
            for path in paths
            if path.name == "causal_micro_lab_cluster_cd_grpo_smoke.yaml"
        )
        smoke_config = yaml.safe_load(smoke_path.read_text(encoding="utf-8"))
        self.assertEqual(
            smoke_config["causal_micro_lab_agent_name"],
            "cd_grpo_agent_loop",
        )
        self.assertEqual(smoke_config["default_agent_loop"], "cd_grpo_agent_loop")
        self.assertEqual(smoke_config["total_training_steps"], 10)

    def test_generation_log_probs_are_required_and_aligned(self):
        @dataclass
        class Output:
            log_probs: list[float] | None

        self.assertEqual(
            _generation_log_probs(
                Output([-0.1, -0.2]),
                expected_length=2,
                required=True,
            ),
            [-0.1, -0.2],
        )
        self.assertIsNone(
            _generation_log_probs(
                Output(None),
                expected_length=2,
                required=False,
            )
        )
        with self.assertRaisesRegex(RuntimeError, "returned none"):
            _generation_log_probs(
                Output(None),
                expected_length=2,
                required=True,
            )
        with self.assertRaisesRegex(RuntimeError, "length mismatch"):
            _generation_log_probs(
                Output([-0.1]),
                expected_length=2,
                required=True,
            )

    def test_cd_agent_loop_propagates_generation_log_probs(self):
        import inspect

        source = inspect.getsource(CDGRPOAgentLoop.run)
        self.assertIn("response_logprobs=response_logprobs", source)

    def test_archive_json_is_plain_and_stable(self):
        archive = BehaviorArchive(counts={("s", "b"): 2.0})
        encoded = json.dumps(archive.to_dict(), sort_keys=True)
        self.assertIn('"counts": [["s", "b", 2.0]]', encoded)


class CDGRPOLoggingTests(unittest.TestCase):
    def test_real_row_scatter_preserves_float_advantage_dtype(self):
        import torch

        integer_mask = torch.ones((4, 2), dtype=torch.long)
        real_indices = torch.tensor([0, 2], dtype=torch.long)
        real_advantages = torch.tensor(
            [[0.25, -0.25], [1.0, -1.0]],
            dtype=torch.float32,
        )

        scattered = scatter_real_rows(
            integer_mask,
            real_indices,
            real_advantages,
        )

        self.assertEqual(scattered.dtype, torch.float32)
        torch.testing.assert_close(scattered[real_indices], real_advantages)
        torch.testing.assert_close(scattered[1], torch.zeros(2))
        torch.testing.assert_close(scattered[3], torch.zeros(2))

    def test_synthetic_padding_is_excluded_from_cd_payloads(self):
        payload = {"state_id": "state-0", "status": "valid"}
        real_indices, payloads, eval_payloads = select_cd_payloads(
            [
                {
                    "cd_reward_payload": payload,
                    "cd_eval_payload": {"valid_mode_count": 4},
                },
                {},
            ],
            [{}, {"is_padding": True}],
        )

        self.assertEqual(real_indices, [0])
        self.assertEqual(payloads, [payload])
        self.assertEqual(eval_payloads, [{"valid_mode_count": 4}])

    def test_missing_cd_payload_on_real_rollout_is_rejected(self):
        with self.assertRaisesRegex(KeyError, "real rollout"):
            select_cd_payloads([{}], [{}])

    def test_nonvalid_payload_can_be_reconstructed_from_reward_metrics(self):
        real_indices, payloads, eval_payloads = select_cd_payloads(
            [
                {
                    "reward_extra_info": {
                        "validity": 0.0,
                        "parse_valid": 0.0,
                        "response_length_cap_hit": 1.0,
                        "valid_mode_count": 16.0,
                    }
                }
            ],
            [{}],
        )

        self.assertEqual(real_indices, [0])
        self.assertEqual(payloads[0]["status"], CandidateStatus.TRUNCATED.value)
        self.assertIsNone(payloads[0]["consequence_signature"])
        self.assertEqual(eval_payloads[0]["valid_mode_count"], 16)

    def test_missing_payload_on_valid_rollout_is_rejected(self):
        with self.assertRaisesRegex(KeyError, "valid rollout"):
            select_cd_payloads(
                [{"reward_extra_info": {"validity": 1.0}}],
                [{}],
            )

    def test_cd_payload_falls_back_to_reward_extra_info(self):
        payload = {"state_id": "state-0", "status": "valid"}
        eval_payload = {"valid_mode_count": 8}
        real_indices, payloads, eval_payloads = select_cd_payloads(
            [
                {
                    "reward_extra_info": {
                        "terminal_reward": 1.0,
                        "cd_reward_payload": payload,
                        "cd_eval_payload": eval_payload,
                    }
                }
            ],
            [{}],
        )

        self.assertEqual(real_indices, [0])
        self.assertEqual(payloads, [payload])
        self.assertEqual(eval_payloads, [eval_payload])

    def test_valid_payload_survives_as_transfer_queue_json_scalars(self):
        payload = {
            "state_id": "state-0",
            "status": CandidateStatus.VALID.value,
            "consequence_signature": "0101",
            "behavior_key": "mode-7",
        }
        eval_payload = {
            "valid_mode_count": 8,
            "separation_bucket": "high",
            "family_bucket": "mixed",
        }
        real_indices, payloads, eval_payloads = select_cd_payloads(
            [
                {
                    "reward_extra_info": {
                        "validity": 1.0,
                        "parse_valid": 1.0,
                        "response_length_cap_hit": 0.0,
                        "cd_reward_payload_json": json.dumps(payload),
                        "cd_eval_payload_json": json.dumps(eval_payload),
                    }
                }
            ],
            [{}],
        )

        self.assertEqual(real_indices, [0])
        self.assertEqual(payloads, [payload])
        self.assertEqual(eval_payloads, [eval_payload])

    def test_malformed_transfer_queue_payload_json_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            select_cd_payloads(
                [
                    {
                        "reward_extra_info": {
                            "validity": 1.0,
                            "cd_reward_payload_json": "{broken",
                        }
                    }
                ],
                [{}],
            )

    def test_incomplete_validation_auxiliary_series_is_excluded(self):
        sanitized, missing_rates = sanitize_validation_reward_extras(
            {
                "reward": [1.0, 0.0],
                "validity": [1.0, 0.0],
                "optional_diagnostic": [None, 2.0],
                "cd_reward_payload": [{"status": "valid"}, {"status": "invalid"}],
                "cd_eval_payload": [
                    {"valid_mode_count": 4},
                    {"valid_mode_count": 8},
                ],
                "cd_reward_payload_json": ["{}", "{}"],
                "cd_eval_payload_json": ["{}", "{}"],
            }
        )

        self.assertEqual(
            sanitized,
            {
                "reward": [1.0, 0.0],
                "validity": [1.0, 0.0],
            },
        )
        self.assertEqual(missing_rates, {"optional_diagnostic": 0.5})

    def test_missing_validation_reward_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "reward series"):
            sanitize_validation_reward_extras({"reward": [1.0, None]})

    def test_activation_advantage_archive_and_truncation_metrics(self):
        import torch

        statuses = [
            CandidateStatus.VALID,
            CandidateStatus.VALID,
            CandidateStatus.VALID,
            CandidateStatus.INVALID,
            CandidateStatus.INVALID,
            CandidateStatus.TRUNCATED,
            CandidateStatus.TRUNCATED,
            CandidateStatus.VALID,
            CandidateStatus.INVALID,
        ]
        signatures = [
            "0000",
            "0000",
            "1111",
            None,
            None,
            None,
            None,
            "0011",
            None,
        ]
        behavior_keys = ["a", "a", "b", None, None, None, None, "c", None]
        state_ids = (
            ["state-0"] * 3 + ["state-1"] * 2 + ["state-2"] * 2 + ["state-3"] * 2
        )
        payloads = [
            {
                "state_id": state_id,
                "status": status.value,
                "consequence_signature": signature,
                "behavior_key": behavior_key,
            }
            for state_id, status, signature, behavior_key in zip(
                state_ids, statuses, signatures, behavior_keys, strict=True
            )
        ]
        token_rewards = torch.tensor(
            [
                [1.0],
                [1.0],
                [1.0],
                [0.2],
                [0.2],
                [-0.2],
                [-0.2],
                [1.0],
                [0.2],
            ]
        )
        mask = torch.ones_like(token_rewards)
        archive = BehaviorArchive()

        advantages, _, metrics = compute_cd_grpo_advantages(
            token_level_rewards=token_rewards,
            response_mask=mask,
            index=["g0", "g0", "g0", "g1", "g1", "g2", "g2", "g3", "g3"],
            payloads=payloads,
            eval_payloads=None,
            config={"beta": 0.3, "archive": True},
            archive=archive,
        )

        self.assertEqual(advantages.shape, token_rewards.shape)
        self.assertEqual(metrics["cd_grpo/groups_with_0_valid_rate"], 0.5)
        self.assertEqual(metrics["cd_grpo/groups_with_1_valid_rate"], 0.25)
        self.assertEqual(metrics["cd_grpo/groups_with_2plus_valid_rate"], 0.25)
        self.assertEqual(metrics["cd_grpo/groups_with_2plus_unique_valid_rate"], 0.25)
        self.assertEqual(metrics["cd_grpo/diversity_signal_active_rate"], 0.25)
        self.assertEqual(metrics["cd_grpo/all_truncated_group_rate"], 0.25)
        self.assertEqual(
            metrics["cd_grpo/pairwise_consequence_distance_mean"], 2.0 / 3.0
        )
        self.assertEqual(
            metrics["cd_grpo/unique_pairwise_consequence_distance_mean"], 1.0
        )
        self.assertGreater(metrics["cd_grpo/diversity_advantage_abs_mean"], 0.0)
        self.assertGreater(metrics["cd_grpo/diversity_contribution_abs_mean"], 0.0)
        self.assertEqual(metrics["cd_grpo/archive_new_valid_completion_rate"], 1.0)
        self.assertEqual(metrics["cd_grpo/archive_new_unique_behavior_rate"], 1.0)

    def test_duplicate_valid_group_has_no_diversity_signal(self):
        import torch

        payloads = [
            {
                "state_id": "state",
                "status": CandidateStatus.VALID.value,
                "consequence_signature": "0101",
                "behavior_key": "same",
            }
            for _ in range(2)
        ]
        _, _, metrics = compute_cd_grpo_advantages(
            token_level_rewards=torch.ones((2, 1)),
            response_mask=torch.ones((2, 1)),
            index=["group", "group"],
            payloads=payloads,
            eval_payloads=None,
            config={"beta": 0.3, "archive": False},
            archive=BehaviorArchive(),
        )

        self.assertEqual(metrics["cd_grpo/groups_with_2plus_valid_rate"], 1.0)
        self.assertEqual(metrics["cd_grpo/groups_with_2plus_unique_valid_rate"], 0.0)
        self.assertEqual(metrics["cd_grpo/diversity_signal_active_rate"], 0.0)
        self.assertEqual(metrics["cd_grpo/pairwise_consequence_distance_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
