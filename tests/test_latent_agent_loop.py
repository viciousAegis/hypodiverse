from pathlib import Path
import unittest

import yaml

from scattered_discovery.verl.latent_agent_loop import (
    LATENT_REWARD_EXTRA_DEFAULTS,
    normalize_latent_reward_extra_info,
)


ROOT = Path(__file__).resolve().parents[1]


class LatentAgentLoopTests(unittest.TestCase):
    def test_reward_extra_schema_is_fixed(self):
        success = normalize_latent_reward_extra_info(
            {
                "reward_syntax_valid": 0.2,
                "reward_valid_hypothesis": 1.0,
                "latent_id": 3,
            },
            reward_score=1.0,
        )
        sparse = normalize_latent_reward_extra_info({}, reward_score=0.0)

        self.assertEqual(set(success), set(LATENT_REWARD_EXTRA_DEFAULTS))
        self.assertEqual(set(sparse), set(LATENT_REWARD_EXTRA_DEFAULTS))
        self.assertEqual(success["latent_id"], 3.0)
        self.assertEqual(sparse["reward_syntax_valid"], 0.0)
        self.assertEqual(sparse["ips_behavior_hash_hi"], -1.0)

    def test_run_configs_use_explicit_latent_agent(self):
        config_dir = ROOT / "configs" / "verl" / "runs"
        for filename in (
            "causal_micro_lab_cluster_latent_ips_grpo.yaml",
            "causal_micro_lab_cluster_latent_only_grpo.yaml",
        ):
            config = yaml.safe_load(
                (config_dir / filename).read_text(encoding="utf-8")
            )
            self.assertNotIn("causal_micro_lab_agent_name", config)
            self.assertEqual(config["default_agent_loop"], "latent_grpo_agent_loop")
            self.assertEqual(
                config["agent_loop_config_path"],
                "configs/verl/latent_grpo_agent_loop.yaml",
            )
            self.assertEqual(
                config["causal_micro_lab_dataset_output_dir"],
                "data/causal_micro_lab/trainable",
            )
            self.assertTrue(
                config["train_file"].startswith("data/causal_micro_lab/trainable/")
            )

    def test_agent_config_maps_generic_and_explicit_names(self):
        entries = yaml.safe_load(
            (ROOT / "configs" / "verl" / "latent_grpo_agent_loop.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            [entry["name"] for entry in entries],
            ["causal_micro_lab_agent_loop", "latent_grpo_agent_loop"],
        )

    def test_slurm_launcher_runs_latent_preflight(self):
        launcher = (
            ROOT / "scripts" / "cluster" / "sbatch_causal_micro_lab_latent_grpo.slurm"
        ).read_text(encoding="utf-8")
        self.assertIn("python scripts/cluster/check_latent_grpo.py", launcher)


if __name__ == "__main__":
    unittest.main()
