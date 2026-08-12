from pathlib import Path
import unittest

import yaml

from scattered_discovery.verl.lifpo_main import require_v1_cli
from scattered_discovery.verl.lifpo_trainer import (
    LIFPO_METADATA_KEYS,
    compute_lifpo_advantages,
)


ROOT = Path(__file__).resolve().parents[1]


class LIFPOAPITests(unittest.TestCase):
    def test_canonical_advantage_api_is_exported(self):
        self.assertTrue(callable(compute_lifpo_advantages))
        self.assertIn("behavior_hash_hi", LIFPO_METADATA_KEYS)
        self.assertIn("behavior_hash_lo", LIFPO_METADATA_KEYS)

    def test_lifpo_requires_v1_runner(self):
        with self.assertRaises(SystemExit):
            require_v1_cli([])
        require_v1_cli(["trainer.use_v1=True"])

    def test_agent_config_registers_canonical_names(self):
        entries = yaml.safe_load(
            (ROOT / "configs" / "verl" / "lifpo_agent_loop.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            [entry["name"] for entry in entries],
            [
                "causal_micro_lab_agent_loop",
                "lifpo_agent_loop",
            ],
        )

    def test_shared_launcher_accepts_public_entrypoint(self):
        launcher = (
            ROOT / "scripts" / "cluster" / "run_verl_discovery_grpo.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("scattered_discovery.verl.lifpo_main", launcher)
        self.assertIn("LIFPOAgentLoopManager", launcher)
        self.assertIn("+algorithm.lifpo.frequency_credit_mode=", launcher)

    def test_rollout_loop_reads_public_lifpo_algorithm_block(self):
        source = (
            ROOT / "src" / "scattered_discovery" / "verl" / "agent_loop.py"
        ).read_text(encoding="utf-8")
        self.assertIn('self.config.algorithm.get("lifpo", {})', source)
        self.assertIn('"negative_offset"', source)


if __name__ == "__main__":
    unittest.main()
