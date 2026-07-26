from pathlib import Path
import unittest

import yaml

try:
    import torch
except ImportError:
    torch = None

from scattered_discovery.envs.causal_micro_lab.state_generator import find_states
from scattered_discovery.verl.agent_loop import (
    _behavior_hash_parts,
    build_latent_prompt,
    latent_id_for_rollout,
    negative_latent_id,
)
from scattered_discovery.verl.ips_grpo_trainer import (
    IPS_REWARD_EXTRA_DEFAULTS,
    compute_ips_grpo_advantages,
    compute_latent_ips_grpo_advantages,
    normalize_ips_reward_extra_info,
    sanitize_validation_reward_extras,
    select_ips_metadata,
)


ROOT = Path(__file__).resolve().parents[1]


class IPSGRPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = find_states(
            0,
            4,
            max_evidence=8,
            beam_width=32,
        )[0]
        cls.state_record = cls.state.to_record(include_private=True)

    def test_reward_extra_schema_is_fixed_across_success_and_failure_paths(self):
        complete = normalize_ips_reward_extra_info(
            {
                "reward_syntax_valid": 0.2,
                "validity": 1.0,
                "reward_valid_hypothesis": 1.0,
                "ips_behavior_hash_hi": 12,
                "ips_behavior_hash_lo": 34,
            },
            reward_score=1.0,
        )
        failure = normalize_ips_reward_extra_info({}, reward_score=0.0)

        self.assertEqual(set(complete), set(IPS_REWARD_EXTRA_DEFAULTS))
        self.assertEqual(set(failure), set(IPS_REWARD_EXTRA_DEFAULTS))
        self.assertEqual(failure["reward_syntax_valid"], 0.0)
        self.assertEqual(failure["ips_behavior_hash_hi"], -1.0)

    def test_behavior_hash_is_stable_and_distinguishes_outcomes(self):
        self.assertEqual(_behavior_hash_parts("mode-a"), _behavior_hash_parts("mode-a"))
        self.assertNotEqual(
            _behavior_hash_parts("mode-a"),
            _behavior_hash_parts("mode-b"),
        )
        self.assertEqual(_behavior_hash_parts(None), (-1, -1))

    def test_latent_schedule_covers_group_and_rotates_negative(self):
        latent_ids = [latent_id_for_rollout(index, 8) for index in range(8)]

        self.assertEqual(latent_ids, list(range(1, 9)))
        self.assertEqual(
            [negative_latent_id(value, 8) for value in latent_ids],
            [2, 3, 4, 5, 6, 7, 8, 1],
        )
        self.assertTrue(build_latent_prompt("evidence", 3).startswith("Strategy 3 |"))

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

    def test_latent_mi_is_validity_gated_and_ips_is_optional(self):
        if torch is None:
            self.skipTest("torch is not installed locally")
        rewards = torch.tensor([[1.0], [1.0], [1.0], [0.2]])
        mask = torch.ones_like(rewards)
        assigned = torch.tensor([[-1.0], [-1.0], [-0.1], [-0.1]])
        negative = torch.tensor([[-1.0], [-1.0], [-2.0], [-9.0]])
        metadata = [
            {
                "valid": True,
                "valid_reward": 1.0,
                "behavior": (1, 1),
                "latent_enabled": True,
                "latent_id": 1,
                "latent_negative_id": 2,
                "answer_token_count": 1,
            },
            {
                "valid": True,
                "valid_reward": 1.0,
                "behavior": (1, 1),
                "latent_enabled": True,
                "latent_id": 2,
                "latent_negative_id": 3,
                "answer_token_count": 1,
            },
            {
                "valid": True,
                "valid_reward": 1.0,
                "behavior": (2, 2),
                "latent_enabled": True,
                "latent_id": 3,
                "latent_negative_id": 4,
                "answer_token_count": 1,
            },
            {
                "valid": False,
                "valid_reward": 0.0,
                "behavior": None,
                "latent_enabled": True,
                "latent_id": 4,
                "latent_negative_id": 1,
                "answer_token_count": 1,
            },
        ]
        common = {
            "token_level_rewards": rewards,
            "response_mask": mask,
            "assigned_log_probs": assigned,
            "negative_log_probs": negative,
            "index": ["state"] * 4,
            "metadata": metadata,
            "epsilon": 0.2,
            "mi_alpha": 0.1,
            "mi_clip": 1.0,
            "latent_count": 4,
        }

        combined, _, combined_metrics = compute_latent_ips_grpo_advantages(
            **common,
            use_ips=True,
        )
        latent_only, _, latent_metrics = compute_latent_ips_grpo_advantages(
            **common,
            use_ips=False,
        )

        self.assertGreater(combined[2, 0], combined[0, 0])
        self.assertGreater(latent_only[2, 0], latent_only[0, 0])
        self.assertGreater(combined_metrics["latent_ips/weight_max"], 1.0)
        self.assertEqual(latent_metrics["latent_ips/weight_max"], 1.0)
        self.assertEqual(combined_metrics["latent_ips/mi_nonzero_rate"], 1 / 3)
        self.assertAlmostEqual(
            combined_metrics["latent_ips/groups_all_latents_present_rate"],
            1.0,
        )

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

    def test_latent_configs_are_compute_identical_except_ips(self):
        config_dir = ROOT / "configs" / "verl" / "runs"
        validity = yaml.safe_load(
            (config_dir / "causal_micro_lab_cluster_validity_grpo.yaml").read_text(
                encoding="utf-8"
            )
        )
        combined = yaml.safe_load(
            (config_dir / "causal_micro_lab_cluster_latent_ips_grpo.yaml").read_text(
                encoding="utf-8"
            )
        )
        latent_only = yaml.safe_load(
            (config_dir / "causal_micro_lab_cluster_latent_only_grpo.yaml").read_text(
                encoding="utf-8"
            )
        )
        ignored = {"experiment_name", "ips_grpo_latent_use_ips"}
        self.assertEqual(
            {key: value for key, value in combined.items() if key not in ignored},
            {key: value for key, value in latent_only.items() if key not in ignored},
        )
        self.assertTrue(combined["ips_grpo_latent_use_ips"])
        self.assertFalse(latent_only["ips_grpo_latent_use_ips"])
        self.assertEqual(combined["rollout_n"], combined["ips_grpo_latent_count"])
        self.assertEqual(combined["train_batch_size"] * combined["rollout_n"], 128)
        self.assertEqual(combined["total_training_steps"], 84)
        for key in (
            "causal_micro_lab_dataset_output_dir",
            "causal_micro_lab_dataset_seed",
            "causal_micro_lab_target_counts",
            "causal_micro_lab_nonempty_output_reward",
            "causal_micro_lab_parse_valid_reward",
            "causal_micro_lab_syntax_valid_reward",
            "causal_micro_lab_valid_hypothesis_reward",
            "causal_micro_lab_length_penalty_start",
            "causal_micro_lab_length_penalty_max",
            "causal_micro_lab_mask_truncated",
            "model_id",
            "train_file",
            "val_file",
            "n_gpus_per_node",
            "train_batch_size",
            "ppo_mini_batch_size",
            "rollout_n",
            "actor_lr",
            "temperature",
            "top_p",
            "max_prompt_length",
            "max_response_length",
        ):
            self.assertEqual(validity[key], combined[key], key)

    def test_slurm_launcher_uses_two_gpus_and_preflight(self):
        slurm = (
            ROOT / "scripts" / "cluster" / "sbatch_causal_micro_lab_ips_grpo.slurm"
        ).read_text(encoding="utf-8")

        self.assertIn("#SBATCH --gres=gpu:2", slurm)
        self.assertIn("check_ips_grpo_verl.py", slurm)
        self.assertIn("causal_micro_lab_cluster_ips_grpo.yaml", slurm)
        self.assertIn('if [[ -z "${WANDB_API_KEY:-}" ]]', slurm)

    def test_ips_launcher_wires_validation_postprocess_manager(self):
        launcher = (
            ROOT / "scripts" / "cluster" / "run_verl_discovery_grpo.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "agent_loop_manager_class="
            "scattered_discovery.verl.ips_grpo_trainer."
            "IPSGRPOAgentLoopManager",
            launcher,
        )

    def test_latent_slurm_launcher_has_both_variants(self):
        slurm = (
            ROOT / "scripts" / "cluster" / "sbatch_causal_micro_lab_latent_grpo.slurm"
        ).read_text(encoding="utf-8")

        self.assertIn("#SBATCH --gres=gpu:2", slurm)
        self.assertIn("check_ips_grpo_verl.py --latent", slurm)
        self.assertIn("causal_micro_lab_cluster_latent_ips_grpo.yaml", slurm)
        self.assertIn("causal_micro_lab_cluster_latent_only_grpo.yaml", slurm)
        self.assertIn('if [[ -z "${WANDB_API_KEY:-}" ]]', slurm)

    def test_latent_eval_cycles_labels_at_matched_k(self):
        latent = yaml.safe_load(
            (
                ROOT
                / "configs"
                / "verl"
                / "eval"
                / "causal_micro_lab_final_k16_latent.yaml"
            ).read_text(encoding="utf-8")
        )
        baseline = yaml.safe_load(
            (
                ROOT
                / "configs"
                / "verl"
                / "eval"
                / "causal_micro_lab_final_k16_base.yaml"
            ).read_text(encoding="utf-8")
        )
        launcher = (
            ROOT / "scripts" / "cluster" / "sbatch_causal_micro_lab_latent_eval.slurm"
        ).read_text(encoding="utf-8")

        self.assertEqual(latent["latent_count"], 8)
        for key in (
            "eval_file",
            "rollouts_per_spec",
            "prefix_ks",
            "max_response_length",
            "temperature",
            "top_p",
            "think",
            "thinking_fallback",
        ):
            self.assertEqual(latent[key], baseline[key], key)
        self.assertEqual(latent["rollouts_per_spec"], 16)
        self.assertEqual(latent["prefix_ks"], "4,8,12,16")
        self.assertEqual(latent["max_response_length"], 6000)
        self.assertIn("MERGED_HF_MODEL_PATH", launcher)


if __name__ == "__main__":
    unittest.main()
