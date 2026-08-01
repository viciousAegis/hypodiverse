from __future__ import annotations

import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scattered_discovery.envs.causal_micro_lab.eval import (
    _bootstrap_metric_name,
    _log_wandb_bootstrap_report,
)
from scattered_discovery.envs.causal_micro_lab.report import (
    _aggregate_rows,
    _aggregate_primary_rows,
    _bootstrap_grouped_rows,
)


def _metric_row(
    state_id: str,
    *,
    passed: bool,
    unique_modes: float = 0.0,
    exact_coverage: float = 0.0,
) -> dict[str, object]:
    return {
        "state_id": state_id,
        "K": 16,
        "M": 4,
        "separation_bucket": "high",
        "family_bucket": "mixed",
        "pass_at_k": float(passed),
        "valid_mode_rate": unique_modes / 16,
        "num_unique_valid_modes": unique_modes,
        "exact_coverage": exact_coverage,
        "budget_normalized_coverage": exact_coverage,
        "family_coverage": exact_coverage,
        "effective_mode_count": unique_modes,
        "mode_entropy": 0.0,
        "dominant_mode_mass": float(passed),
        "duplicity": 0.0,
        "generated_mode_separation": 0.0,
        "generated_to_available_separation": 0.0,
        "predictive_diversity_recovery": exact_coverage,
        "predictive_coverage_auc": exact_coverage,
        "predictive_placement_regret": 1.0 - exact_coverage,
        "full_outcome_generated_separation": exact_coverage,
    }


class CausalMicroLabReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            _metric_row("s0", passed=False),
            _metric_row("s1", passed=False),
            _metric_row(
                "s2",
                passed=True,
                unique_modes=1.0,
                exact_coverage=0.25,
            ),
            _metric_row(
                "s3",
                passed=True,
                unique_modes=2.0,
                exact_coverage=0.5,
            ),
        ]

    def test_primary_metrics_separate_success_from_conditional_diversity(self) -> None:
        result = _aggregate_primary_rows(self.rows, ("K", "M"))[0]

        self.assertEqual(result["support_states"], 4)
        self.assertEqual(result["successful_states"], 2)
        self.assertEqual(result["pass_at_k"], 0.5)
        self.assertEqual(
            result["predictive_diversity_recovery_given_success"],
            0.375,
        )
        self.assertEqual(result["predictive_coverage_auc_given_success"], 0.375)
        self.assertEqual(result["predictive_placement_regret_given_success"], 0.625)
        self.assertEqual(result["modes_recovered_given_success"], 1.5)
        self.assertEqual(
            result["fraction_modes_recovered_given_success"],
            0.375,
        )

    def test_general_aggregates_condition_diversity_on_success(self) -> None:
        result = _aggregate_rows(self.rows, ("K", "M"))[0]

        self.assertEqual(result["pass_at_k"], 0.5)
        self.assertEqual(result["successful_states"], 2)
        self.assertEqual(
            result["predictive_diversity_recovery_given_success"],
            0.375,
        )
        self.assertNotIn("predictive_diversity_recovery", result)

    def test_bootstrap_resamples_states_with_correct_conditioning(self) -> None:
        results = _bootstrap_grouped_rows(
            self.rows,
            slice_name="by_k_m",
            group_keys=("K", "M"),
            samples=500,
            rng=random.Random(7),
        )
        by_metric = {str(row["metric"]): row for row in results}

        pass_row = by_metric["pass_at_k"]
        self.assertEqual(pass_row["conditioning"], "all_states")
        self.assertEqual(pass_row["support_states"], 4)
        self.assertEqual(pass_row["successful_states"], 2)
        self.assertEqual(pass_row["mean"], 0.5)
        self.assertLessEqual(pass_row["ci95_low"], 0.5)
        self.assertGreaterEqual(pass_row["ci95_high"], 0.5)

        modes_row = by_metric["modes_recovered_given_success"]
        self.assertEqual(modes_row["conditioning"], "successful_states")
        self.assertEqual(modes_row["support_states"], 4)
        self.assertEqual(modes_row["successful_states"], 2)
        self.assertEqual(modes_row["mean"], 1.5)
        self.assertLessEqual(modes_row["ci95_low"], 1.5)
        self.assertGreaterEqual(modes_row["ci95_high"], 1.5)

        coverage_row = by_metric["fraction_modes_recovered_given_success"]
        self.assertEqual(coverage_row["mean"], 0.375)
        self.assertLessEqual(coverage_row["ci95_low"], 0.375)
        self.assertGreaterEqual(coverage_row["ci95_high"], 0.375)

        coverage_auc = by_metric["predictive_coverage_auc_given_success"]
        self.assertEqual(coverage_auc["conditioning"], "successful_states")
        self.assertEqual(coverage_auc["mean"], 0.375)

    def test_bootstrap_metric_name_includes_slice_labels(self) -> None:
        name = _bootstrap_metric_name(
            {
                "slice": "by_k_separation",
                "K": "16",
                "M": "",
                "separation_bucket": "high",
                "family_bucket": "",
                "metric": "modes_recovered_given_success",
            }
        )
        self.assertEqual(
            name,
            (
                "bootstrap_ci95/by_k_separation/K_16/"
                "separation_bucket_high/modes_recovered_given_success"
            ),
        )

    def test_wandb_report_logs_tables_scalars_and_artifact(self) -> None:
        header = (
            "slice,K,M,separation_bucket,family_bucket,metric,conditioning,"
            "support_states,successful_states,mean,ci95_low,ci95_high,"
            "bootstrap_samples\n"
        )
        row = "by_k_m,16,4,,,pass_at_k,all_states,24,20,0.8,0.6,0.9,1000\n"

        class FakeTable:
            def __init__(self, *, columns, data):
                self.columns = columns
                self.data = data

        class FakeArtifact:
            def __init__(self, *, name, type, metadata):
                self.name = name
                self.type = type
                self.metadata = metadata
                self.directories = []

            def add_dir(self, path):
                self.directories.append(path)

        class FakeRun:
            id = "run123"

            def __init__(self):
                self.logged = []
                self.artifacts = []

            def log(self, values):
                self.logged.append(values)

            def log_artifact(self, artifact):
                self.artifacts.append(artifact)

        fake_wandb = types.SimpleNamespace(
            Table=FakeTable,
            Artifact=FakeArtifact,
        )
        with tempfile.TemporaryDirectory() as temporary:
            report_dir = Path(temporary)
            for filename in (
                "bootstrap_ci95.csv",
                "primary_bootstrap_ci95_by_k_m.csv",
                "primary_bootstrap_ci95_by_k_separation.csv",
                "primary_bootstrap_ci95_by_k_family.csv",
            ):
                (report_dir / filename).write_text(
                    header + row,
                    encoding="utf-8",
                )
            run = FakeRun()
            with patch.dict(sys.modules, {"wandb": fake_wandb}):
                _log_wandb_bootstrap_report(
                    run,
                    report_dir=report_dir,
                    bootstrap_samples=1000,
                )

        self.assertEqual(len(run.logged), 2)
        self.assertIn("eval_tables/bootstrap_ci95", run.logged[0])
        self.assertEqual(
            run.logged[1]["bootstrap_ci95/by_k_m/K_16/M_4/pass_at_k/ci95_low"],
            0.6,
        )
        self.assertEqual(len(run.artifacts), 1)
        self.assertEqual(run.artifacts[0].type, "evaluation-report")
        self.assertEqual(run.artifacts[0].metadata["bootstrap_samples"], 1000)


if __name__ == "__main__":
    unittest.main()
