from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIG_DIR = ROOT / "configs" / "verl" / "runs"


class ClusterValidityConfigTests(unittest.TestCase):
    def test_validity_and_cd_grpo_match_shared_compute_settings(self):
        validity = yaml.safe_load(
            (
                RUN_CONFIG_DIR / "causal_micro_lab_cluster_validity_grpo.yaml"
            ).read_text(encoding="utf-8")
        )
        cd_grpo = yaml.safe_load(
            (
                RUN_CONFIG_DIR / "causal_micro_lab_cluster_cd_grpo.yaml"
            ).read_text(encoding="utf-8")
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
            self.assertEqual(validity[key], cd_grpo[key], key)

        validity_responses = validity["train_batch_size"] * validity["rollout_n"]
        cd_responses = cd_grpo["train_batch_size"] * cd_grpo["rollout_n"]
        self.assertEqual(validity_responses, 128)
        self.assertEqual(cd_responses, 128)
        self.assertEqual(validity["rollout_n"], 8)
        self.assertEqual(cd_grpo["rollout_n"], 16)

    def test_validity_slurm_entrypoint_uses_two_gpu_config_and_wandb(self):
        slurm = (
            ROOT / "scripts" / "cluster" / "sbatch_causal_micro_lab_validity_grpo.slurm"
        ).read_text(encoding="utf-8")

        self.assertIn("#SBATCH --gres=gpu:2", slurm)
        self.assertIn("causal_micro_lab_cluster_validity_grpo.yaml", slurm)
        self.assertIn('if [[ -z "${WANDB_API_KEY:-}" ]]', slurm)
        self.assertIn("scripts/cluster/run_verl_pilot_grpo.sh", slurm)

    def test_causal_dataset_manifest_is_checked_on_every_launch(self):
        wrapper = (
            ROOT / "scripts" / "cluster" / "run_verl_pilot_grpo.sh"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "scripts/cluster/prepare_causal_micro_lab_dataset.sh",
            wrapper,
        )
        self.assertNotIn("missing_cml_file", wrapper)


if __name__ == "__main__":
    unittest.main()
