from pathlib import Path
import unittest

import yaml

try:
    import torch
except ImportError:
    torch = None

from scattered_discovery.verl.agent_loop import _behavior_hash_parts
from scattered_discovery.verl.ips_grpo_trainer import (
    compute_ips_grpo_advantages,
    sanitize_validation_reward_extras,
    select_ips_metadata,
)


ROOT = Path(__file__).resolve().parents[1]


class IPSGRPOTests(unittest.TestCase):
    def test_behavior_hash_is_stable_and_distinguishes_outcomes(self):
        self.assertEqual(_behavior_hash_parts("mode-a"), _behavior_hash_parts("mode-a"))
        self.assertNotEqual(
            _behavior_hash_parts("mode-a"),
            _behavior_hash_parts("mode-b"),
        )
        self.assertEqual(_behavior_hash_parts(None), (-1, -1))

    def test_ips_rewards_rare_valid_outcome_more_than_duplicate(self):
        if torch is None:
            self.skipTest("torch is not installed locally")
        rewards = torch.tensor([[1.0], [1.0], [1.0], [0.2]])
        mask = torch.ones_like(rewards)
        metadata = [
            {"valid": True, "valid_reward": 1.0, "behavior": (1, 1)},
            {"valid": True, "valid_reward": 1.0, "behavior": (1, 1)},
            {"valid": True, "valid_reward": 1.0, "behavior": (2, 2)},
            {"valid": False, "valid_reward": 0.0, "behavior": None},
        ]

        advantages, returns, metrics = compute_ips_grpo_advantages(
            token_level_rewards=rewards,
            response_mask=mask,
            index=["state"] * 4,
            metadata=metadata,
            epsilon=0.2,
        )

        self.assertTrue(torch.equal(advantages, returns))
        self.assertGreater(advantages[2, 0], advantages[0, 0])
        self.assertGreater(advantages[0, 0], advantages[3, 0])
        self.assertAlmostEqual(metrics["ips_grpo/weight_max"], 4.0)
        self.assertAlmostEqual(metrics["ips_grpo/duplicate_valid_rate"], 1 / 3)
        self.assertAlmostEqual(
            metrics["ips_grpo/unique_valid_outcomes_per_group"],
            2.0,
        )

    def test_epsilon_clips_singleton_weight_for_eight_rollouts(self):
        if torch is None:
            self.skipTest("torch is not installed locally")
        rewards = torch.tensor([[1.0]] + [[0.0]] * 7)
        mask = torch.ones_like(rewards)
        metadata = [{"valid": True, "valid_reward": 1.0, "behavior": (1, 1)}] + [
            {"valid": False, "valid_reward": 0.0, "behavior": None} for _ in range(7)
        ]

        _, _, metrics = compute_ips_grpo_advantages(
            token_level_rewards=rewards,
            response_mask=mask,
            index=["state"] * 8,
            metadata=metadata,
            epsilon=0.2,
        )

        self.assertAlmostEqual(metrics["ips_grpo/weight_max"], 5.0)
        self.assertAlmostEqual(metrics["ips_grpo/clipped_valid_rate"], 1.0)
        self.assertAlmostEqual(metrics["ips_grpo/scaled_score_max"], 5.0)

    def test_syntax_shaping_is_not_inverse_weighted(self):
        if torch is None:
            self.skipTest("torch is not installed locally")
        rewards = torch.tensor([[0.2], [0.0]])
        mask = torch.ones_like(rewards)
        metadata = [
            {"valid": False, "valid_reward": 0.0, "behavior": None},
            {"valid": False, "valid_reward": 0.0, "behavior": None},
        ]

        advantages, _, metrics = compute_ips_grpo_advantages(
            token_level_rewards=rewards,
            response_mask=mask,
            index=["state", "state"],
            metadata=metadata,
            epsilon=0.2,
        )

        self.assertGreater(advantages[0, 0], advantages[1, 0])
        self.assertAlmostEqual(metrics["ips_grpo/scaled_score_mean"], 0.1)
        self.assertAlmostEqual(metrics["ips_grpo/groups_without_valid_rate"], 1.0)

    def test_metadata_selection_uses_numeric_reward_fields_and_skips_padding(self):
        extra_fields = [
            {
                "reward_extra_info": {
                    "validity": 1.0,
                    "reward_valid_hypothesis": 1.0,
                    "ips_behavior_hash_hi": 12.0,
                    "ips_behavior_hash_lo": 34.0,
                }
            },
            {},
        ]
        real_indices, metadata = select_ips_metadata(
            extra_fields,
            [{"is_padding": False}, {"is_padding": True}],
        )

        self.assertEqual(real_indices, [0])
        self.assertEqual(metadata[0]["behavior"], (12, 34))

    def test_validation_sanitizer_removes_behavior_identifiers(self):
        sanitized, missing = sanitize_validation_reward_extras(
            {
                "reward": [1.0, 0.0],
                "validity": [1.0, 0.0],
                "ips_behavior_hash_hi": [12.0, -1.0],
                "ips_behavior_hash_lo": [34.0, -1.0],
            }
        )

        self.assertEqual(set(sanitized), {"reward", "validity"})
        self.assertEqual(missing, {})

    def test_run_config_is_compute_matched_to_validity(self):
        config_dir = ROOT / "configs" / "verl" / "runs"
        validity = yaml.safe_load(
            (config_dir / "causal_micro_lab_cluster_validity_grpo.yaml").read_text(
                encoding="utf-8"
            )
        )
        ips = yaml.safe_load(
            (config_dir / "causal_micro_lab_cluster_ips_grpo.yaml").read_text(
                encoding="utf-8"
            )
        )
        shared_keys = (
            "causal_micro_lab_dataset_output_dir",
            "causal_micro_lab_dataset_seed",
            "causal_micro_lab_target_counts",
            "causal_micro_lab_val_max_rows",
            "causal_micro_lab_test_max_rows",
            "causal_micro_lab_nonempty_output_reward",
            "causal_micro_lab_rule_marker_reward",
            "causal_micro_lab_parse_valid_reward",
            "causal_micro_lab_syntax_valid_reward",
            "causal_micro_lab_evidence_consistent_reward",
            "causal_micro_lab_valid_hypothesis_reward",
            "causal_micro_lab_length_penalty_start",
            "causal_micro_lab_length_penalty_max",
            "causal_micro_lab_mask_truncated",
            "model_id",
            "model_basename",
            "train_file",
            "val_file",
            "n_gpus_per_node",
            "train_batch_size",
            "ppo_mini_batch_size",
            "rollout_n",
            "actor_lr",
            "kl_loss_coef",
            "entropy_coeff",
            "temperature",
            "top_p",
            "total_training_steps",
            "max_prompt_length",
            "max_response_length",
        )
        for key in shared_keys:
            self.assertEqual(validity[key], ips[key], key)
        self.assertEqual(ips["train_batch_size"] * ips["rollout_n"], 128)
        self.assertEqual(ips["ips_grpo_epsilon"], 0.2)

    def test_slurm_launcher_uses_two_gpus_and_preflight(self):
        slurm = (
            ROOT / "scripts" / "cluster" / "sbatch_causal_micro_lab_ips_grpo.slurm"
        ).read_text(encoding="utf-8")

        self.assertIn("#SBATCH --gres=gpu:2", slurm)
        self.assertIn("check_ips_grpo_verl.py", slurm)
        self.assertIn("causal_micro_lab_cluster_ips_grpo.yaml", slurm)
        self.assertIn('if [[ -z "${WANDB_API_KEY:-}" ]]', slurm)


if __name__ == "__main__":
    unittest.main()
