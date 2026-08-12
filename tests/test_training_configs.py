from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIG_DIR = ROOT / "configs" / "verl" / "runs"


class TrainingConfigTests(unittest.TestCase):
    def test_grpo_and_lifpo_match_shared_compute_settings(self):
        grpo = yaml.safe_load(
            (RUN_CONFIG_DIR / "causal_micro_lab_cluster_grpo.yaml").read_text(
                encoding="utf-8"
            )
        )
        lifpo = yaml.safe_load(
            (RUN_CONFIG_DIR / "causal_micro_lab_cluster_lifpo.yaml").read_text(
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
            self.assertEqual(grpo[key], lifpo[key], key)

        self.assertEqual(grpo["causal_micro_lab_generate_dataset_if_missing"], 0)
        self.assertEqual(lifpo["causal_micro_lab_generate_dataset_if_missing"], 0)

        grpo_responses = grpo["train_batch_size"] * grpo["rollout_n"]
        lifpo_responses = lifpo["train_batch_size"] * lifpo["rollout_n"]
        self.assertEqual(grpo_responses, 128)
        self.assertEqual(lifpo_responses, 128)
        self.assertEqual(grpo["rollout_n"], 8)
        self.assertEqual(lifpo["rollout_n"], 8)

    def test_grpo_slurm_entrypoint_uses_two_gpu_config_and_wandb(self):
        slurm = (
            ROOT / "scripts" / "cluster" / "sbatch_causal_micro_lab_grpo.slurm"
        ).read_text(encoding="utf-8")

        self.assertIn("#SBATCH --gres=gpu:2", slurm)
        self.assertIn("causal_micro_lab_cluster_grpo.yaml", slurm)
        self.assertIn('if [[ -z "${WANDB_API_KEY:-}" ]]', slurm)
        self.assertIn("scripts/cluster/run_verl_pilot_grpo.sh", slurm)

    def test_causal_dataset_manifest_is_checked_on_every_launch(self):
        wrapper = (ROOT / "scripts" / "cluster" / "run_verl_pilot_grpo.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "scripts/cluster/prepare_causal_micro_lab_dataset.sh",
            wrapper,
        )
        self.assertNotIn("missing_cml_file", wrapper)


if __name__ == "__main__":
    unittest.main()
