from __future__ import annotations

import random
import unittest

from scattered_discovery.envs.causal_micro_lab.report import (
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
        self.assertEqual(result["modes_recovered_given_success"], 1.5)
        self.assertEqual(
            result["fraction_modes_recovered_given_success"],
            0.375,
        )

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


if __name__ == "__main__":
    unittest.main()
