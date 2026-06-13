import json
import tempfile
import unittest
from pathlib import Path

from scattered_discovery.eval.envspecs import load_env_specs, summarize_records


class EnvSpecEvalTests(unittest.TestCase):
    def test_load_env_specs_from_jsonl_rows(self):
        spec = {
            "env_type": "hypospace_boolean",
            "task": {"variables": ["x", "y"], "target_expression": "x AND y"},
            "protocol": "single",
            "max_steps": 3,
            "max_commit": 1,
            "seed": 7,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "eval.jsonl"
            path.write_text(
                json.dumps({"env_spec_json": json.dumps(spec)}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_env_specs(path), [spec])

    def test_summarize_records(self):
        records = [
            {
                "score": {
                    "reward": 1.0,
                    "valid_committed_count": 1,
                    "valid_unique_count": 1,
                    "committed_count": 1,
                    "false_count": 0,
                    "non_final_count": 0,
                    "unsupported_count": 0,
                    "duplicate_count": 0,
                    "validity": 1.0,
                    "uniqueness": 1.0,
                    "parse_failures": 0,
                    "invalid_actions": 0,
                    "metrics": {"recovery": 1.0},
                },
                "model_seconds": 2.0,
            },
            {
                "score": {
                    "reward": 0.0,
                    "valid_committed_count": 0,
                    "valid_unique_count": 0,
                    "committed_count": 1,
                    "false_count": 1,
                    "non_final_count": 0,
                    "unsupported_count": 0,
                    "duplicate_count": 0,
                    "validity": 0.0,
                    "uniqueness": 1.0,
                    "parse_failures": 1,
                    "invalid_actions": 1,
                    "metrics": {"recovery": 0.0},
                },
                "model_seconds": 3.0,
            },
        ]
        summary = summarize_records(records)
        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["reward_mean"], 0.5)
        self.assertEqual(summary["valid_unique_count_mean"], 0.5)
        self.assertEqual(summary["validity_mean"], 0.5)
        self.assertEqual(summary["uniqueness_mean"], 1.0)
        self.assertEqual(summary["non_final_count_mean"], 0.0)
        self.assertEqual(summary["model_seconds_total"], 5.0)


if __name__ == "__main__":
    unittest.main()
