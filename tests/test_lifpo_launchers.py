from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class LIFPOPublicSurfaceTests(unittest.TestCase):
    def test_canonical_config_preserves_evaluated_objective(self):
        run_dir = ROOT / "configs" / "verl" / "runs"
        evaluated = yaml.safe_load(
            (run_dir / "causal_micro_lab_cluster_latent_ips_grpo_v2.yaml").read_text(
                encoding="utf-8"
            )
        )
        canonical = yaml.safe_load(
            (run_dir / "causal_micro_lab_cluster_lifpo.yaml").read_text(
                encoding="utf-8"
            )
        )
        renamed = {
            "ips_grpo_epsilon": "lifpo_frequency_epsilon",
            "ips_grpo_probe_fraction": "lifpo_probe_fraction",
            "ips_grpo_length_penalty_start": "lifpo_length_penalty_start",
            "ips_grpo_latent_enabled": "lifpo_latent_enabled",
            "ips_grpo_latent_count": "lifpo_latent_count",
            "ips_grpo_latent_negative_offset": "lifpo_negative_offset",
            "ips_grpo_latent_mi_alpha": "lifpo_counterfactual_alpha",
            "ips_grpo_latent_mi_clip": "lifpo_counterfactual_clip",
            "ips_grpo_latent_mi_token_scope": "lifpo_counterfactual_token_scope",
            "ips_grpo_latent_mi_reduction": "lifpo_counterfactual_reduction",
            "ips_grpo_latent_mi_valid_only": "lifpo_counterfactual_valid_only",
            "ips_grpo_latent_use_ips": "lifpo_inverse_frequency_enabled",
            "ips_grpo_latent_ips_reward_mode": "lifpo_frequency_credit_mode",
            "ips_grpo_latent_ips_bonus_max": "lifpo_frequency_credit_max",
        }
        intentionally_changed = {
            "trainer_entrypoint",
            "agent_loop_config_path",
            "default_agent_loop",
            "experiment_name",
            "total_training_steps",
        }
        for old_key, expected in evaluated.items():
            if old_key in intentionally_changed:
                continue
            new_key = renamed.get(old_key, old_key)
            self.assertEqual(canonical[new_key], expected, old_key)

    def test_canonical_run_config_uses_public_lifpo_surface(self):
        run_dir = ROOT / "configs" / "verl" / "runs"
        lifpo = yaml.safe_load(
            (run_dir / "causal_micro_lab_cluster_lifpo.yaml").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(lifpo["total_training_steps"], 100)
        self.assertEqual(lifpo["experiment_name"], "causal_micro_lab_cluster_lifpo_r1")
        self.assertEqual(
            lifpo["trainer_entrypoint"], "scattered_discovery.verl.lifpo_main"
        )
        self.assertEqual(
            lifpo["agent_loop_config_path"], "configs/verl/lifpo_agent_loop.yaml"
        )
        self.assertEqual(lifpo["default_agent_loop"], "lifpo_agent_loop")
        self.assertTrue(lifpo["lifpo_latent_enabled"])
        self.assertTrue(lifpo["lifpo_inverse_frequency_enabled"])
        self.assertNotIn("ips_grpo_latent_enabled", lifpo)

    def test_canonical_eval_changes_only_public_run_name(self):
        eval_dir = ROOT / "configs" / "verl" / "eval"
        historical = yaml.safe_load(
            (eval_dir / "causal_micro_lab_final_k16_latent_v3.yaml").read_text(
                encoding="utf-8"
            )
        )
        lifpo = yaml.safe_load(
            (eval_dir / "causal_micro_lab_final_k16_lifpo_v3.yaml").read_text(
                encoding="utf-8"
            )
        )

        expected = dict(historical)
        expected["run_name"] = "causal_micro_lab_final_v3_k16_lifpo"
        self.assertEqual(lifpo, expected)
        self.assertEqual(lifpo["latent_count"], 8)

    def test_lifpo_launcher_is_two_gpu_and_has_no_implicit_warmup(self):
        launcher = (
            ROOT / "scripts" / "cluster" / "sbatch_causal_micro_lab_lifpo.slurm"
        ).read_text(encoding="utf-8")

        self.assertIn("#SBATCH --gres=gpu:2", launcher)
        self.assertIn("causal_micro_lab_cluster_lifpo.yaml", launcher)
        self.assertIn("check_lifpo.py", launcher)
        self.assertIn("Qwen/Qwen3-4B", launcher)
        self.assertIn("run_verl_pilot_grpo.sh", launcher)
        self.assertNotIn("INIT_RUN=", launcher)
        self.assertNotIn("training_initializers", launcher)
        self.assertNotIn("model_merger", launcher)

    def test_v3_checkpoint_wrapper_exposes_lifpo_and_deprecated_alias(self):
        wrapper = (
            ROOT
            / "scripts"
            / "cluster"
            / "sbatch_causal_micro_lab_v3_checkpoint_eval.slurm"
        ).read_text(encoding="utf-8")

        self.assertIn("{standard|lifpo|latent}", wrapper)
        self.assertIn("causal_micro_lab_final_k16_lifpo_v3.yaml", wrapper)
        self.assertIn("EVAL_MODE 'latent' is deprecated", wrapper)
        self.assertIn('INTERNAL_MODE="latent"', wrapper)
        self.assertIn('ARGS=("$TRAINING_RUN" "$INTERNAL_MODE")', wrapper)

    def test_v3_submitter_uses_only_frozen_thesis_models(self):
        submitter = (
            ROOT / "scripts" / "cluster" / "submit_causal_micro_lab_v3_evals.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("submit base", submitter)
        self.assertIn("submit grpo", submitter)
        self.assertIn(
            "causal_micro_lab_cluster_validity_grpo_r1 standard 90", submitter
        )
        self.assertIn("submit lifpo", submitter)
        self.assertIn(
            "causal_micro_lab_cluster_latent_ips_grpo_v2_fulltraj_k8_r1 lifpo 55",
            submitter,
        )
        self.assertNotIn("submit ips", submitter)
        self.assertNotIn("causal_micro_lab_cluster_ips_grpo", submitter)
        self.assertNotIn("submit latent", submitter)


if __name__ == "__main__":
    unittest.main()
