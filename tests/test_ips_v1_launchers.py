from pathlib import Path
import unittest

import yaml

from scattered_discovery.verl.ips_grpo_main import require_v1_cli


ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIG_DIR = ROOT / "configs" / "verl" / "runs"


class IPSV1LauncherTests(unittest.TestCase):
    def test_entrypoint_rejects_missing_or_disabled_v1(self):
        with self.assertRaises(SystemExit):
            require_v1_cli([])
        with self.assertRaises(SystemExit):
            require_v1_cli(["trainer.use_v1=False"])
        require_v1_cli(["trainer.use_v1=True"])
        require_v1_cli(["+trainer.use_v1=true"])

    def test_corrected_configs_are_v1_and_use_fresh_experiments(self):
        pairs = (
            (
                "causal_micro_lab_cluster_ips_grpo.yaml",
                "causal_micro_lab_cluster_ips_grpo_v1.yaml",
            ),
            (
                "causal_micro_lab_cluster_latent_ips_grpo.yaml",
                "causal_micro_lab_cluster_latent_ips_grpo_v1.yaml",
            ),
            (
                "causal_micro_lab_cluster_latent_only_grpo.yaml",
                "causal_micro_lab_cluster_latent_only_grpo_v1.yaml",
            ),
        )
        for old_name, corrected_name in pairs:
            old = yaml.safe_load(
                (RUN_CONFIG_DIR / old_name).read_text(encoding="utf-8")
            )
            corrected = yaml.safe_load(
                (RUN_CONFIG_DIR / corrected_name).read_text(encoding="utf-8")
            )
            self.assertTrue(corrected["trainer_use_v1"])
            self.assertNotEqual(
                old["experiment_name"],
                corrected["experiment_name"],
            )
            self.assertIn("_v1_", corrected["experiment_name"])

    def test_corrected_launchers_force_v1_last(self):
        scripts = (
            "sbatch_causal_micro_lab_ips_grpo_v1.slurm",
            "sbatch_causal_micro_lab_latent_grpo_v1.slurm",
        )
        for name in scripts:
            text = (ROOT / "scripts" / "cluster" / name).read_text(
                encoding="utf-8"
            )
            self.assertIn("--run-config \"$RUN_CONFIG_PATH\"", text)
            command = [
                line
                for line in text.splitlines()
                if line.startswith("scripts/cluster/run_verl_pilot_grpo.sh")
            ]
            self.assertEqual(len(command), 1)
            self.assertTrue(command[0].endswith('"$@" trainer.use_v1=True'))

    def test_tq_worker_forces_method_specific_agent_loop(self):
        trainer_source = (
            ROOT
            / "src"
            / "scattered_discovery"
            / "verl"
            / "ips_grpo_trainer.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'kwargs["agent_name"] = self.ips_agent_loop_name',
            trainer_source,
        )
        self.assertIn('"latent_grpo_agent_loop"', trainer_source)
        self.assertIn('"ips_grpo_agent_loop"', trainer_source)


if __name__ == "__main__":
    unittest.main()
