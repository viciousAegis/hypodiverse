from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class LIFPOPublicSurfaceTests(unittest.TestCase):
    def test_canonical_run_config_matches_reported_setup(self):
        config = yaml.safe_load(
            (
                ROOT
                / "configs"
                / "verl"
                / "runs"
                / "causal_micro_lab_cluster_lifpo.yaml"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(
            config["trainer_entrypoint"], "scattered_discovery.verl.lifpo_main"
        )
        self.assertEqual(
            config["agent_loop_config_path"], "configs/verl/lifpo_agent_loop.yaml"
        )
        self.assertEqual(config["default_agent_loop"], "lifpo_agent_loop")
        self.assertEqual(config["model_id"], "Qwen/Qwen3-4B")
        self.assertEqual(config["n_gpus_per_node"], 2)
        self.assertEqual(config["train_batch_size"], 16)
        self.assertEqual(config["rollout_n"], 8)
        self.assertEqual(config["max_response_length"], 6000)
        self.assertEqual(config["total_training_steps"], 100)
        self.assertEqual(config["experiment_name"], "hypodiverse_lifpo_r1")
        self.assertEqual(config["save_freq"], 5)
        self.assertEqual(config["test_freq"], 10)

        self.assertTrue(config["lifpo_latent_enabled"])
        self.assertEqual(config["lifpo_latent_count"], 8)
        self.assertEqual(config["lifpo_counterfactual_alpha"], 0.5)
        self.assertEqual(config["lifpo_counterfactual_clip"], 1.0)
        self.assertEqual(config["lifpo_counterfactual_token_scope"], "full_response")
        self.assertEqual(config["lifpo_counterfactual_reduction"], "sum")
        self.assertTrue(config["lifpo_counterfactual_valid_only"])
        self.assertTrue(config["lifpo_inverse_frequency_enabled"])
        self.assertEqual(config["lifpo_frequency_credit_mode"], "bonus")
        self.assertEqual(config["lifpo_frequency_credit_max"], 0.5)

    def test_canonical_eval_uses_eight_latent_identities(self):
        config = yaml.safe_load(
            (ROOT / "configs" / "verl" / "eval" / "hypodiverse_lifpo.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["run_name"], "hypodiverse_lifpo_final")
        self.assertEqual(config["latent_count"], 8)
        self.assertEqual(config["rollouts_per_spec"], 16)
        self.assertEqual(config["prefix_ks"], "4,8,12,16")
        self.assertEqual(config["max_response_length"], 6000)

    def test_lifpo_launcher_is_two_gpu_and_has_no_implicit_initializer(self):
        launcher = (
            ROOT / "scripts" / "cluster" / "sbatch_causal_micro_lab_lifpo.slurm"
        ).read_text(encoding="utf-8")

        self.assertIn("#SBATCH --gres=gpu:2", launcher)
        self.assertIn("causal_micro_lab_cluster_lifpo.yaml", launcher)
        self.assertIn("check_lifpo.py", launcher)
        self.assertIn("Qwen/Qwen3-4B", launcher)
        self.assertNotIn("INIT_RUN=", launcher)
        self.assertNotIn("training_initializers", launcher)

    def test_released_eval_supports_only_reported_methods(self):
        launcher = (
            ROOT / "scripts" / "cluster" / "sbatch_causal_micro_lab_released_eval.slurm"
        ).read_text(encoding="utf-8")
        self.assertIn("{base|grpo|lifpo}", launcher)
        self.assertIn("viciousa3gis/hypodiverse-grpo", launcher)
        self.assertIn("viciousa3gis/hypodiverse-lifpo", launcher)
        self.assertIn("hypodiverse_lifpo.yaml", launcher)

    def test_submitter_uses_released_models(self):
        submitter = (
            ROOT / "scripts" / "cluster" / "submit_hypodiverse_evals.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("submit base", submitter)
        self.assertIn("submit grpo", submitter)
        self.assertIn("submit lifpo", submitter)
        self.assertEqual(
            submitter.count("sbatch_causal_micro_lab_released_eval.slurm"), 3
        )


if __name__ == "__main__":
    unittest.main()
