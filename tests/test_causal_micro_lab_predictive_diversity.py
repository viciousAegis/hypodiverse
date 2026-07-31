from __future__ import annotations

from dataclasses import replace
import math
import unittest

from scattered_discovery.envs.causal_micro_lab.parser import parse_record_state
from scattered_discovery.envs.causal_micro_lab.benchmark_v2 import (
    CONTINUOUS_SEPARATION_LABEL,
    assert_clean_eval_audit,
    eval_overlap_audit,
    select_continuous_states,
    separation_distribution_by_m,
)
from scattered_discovery.envs.causal_micro_lab.predictive_diversity import (
    PredictiveDistanceMatrix,
    mode_prediction_distance,
    prediction_target_indices,
    theoretical_binary_pairwise_max,
)
from scattered_discovery.envs.causal_micro_lab.signatures import build_mode_table
from scattered_discovery.envs.causal_micro_lab.state_generator import (
    EvidenceItem,
    assign_absolute_separation_buckets,
    find_states,
    separation_for_modes,
)


class PredictiveDiversityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.table = build_mode_table()
        cls.state = find_states(
            0,
            8,
            max_evidence=8,
            beam_width=64,
            mode_table=cls.table,
            max_results=1,
        )[0]

    def test_prediction_target_validation(self):
        self.assertEqual(prediction_target_indices(("Y",)), (2,))
        self.assertEqual(prediction_target_indices(("Z1", "Y")), (0, 2))
        with self.assertRaises(ValueError):
            prediction_target_indices(())
        with self.assertRaises(ValueError):
            prediction_target_indices(("UNKNOWN",))

    def test_binary_pairwise_theoretical_maximum(self):
        self.assertEqual(theoretical_binary_pairwise_max(1), 0.0)
        self.assertAlmostEqual(theoretical_binary_pairwise_max(4), 2.0 / 3.0)
        self.assertAlmostEqual(theoretical_binary_pairwise_max(8), 4.0 / 7.0)
        self.assertAlmostEqual(theoretical_binary_pairwise_max(16), 8.0 / 15.0)

    def test_state_separation_defaults_to_target_outcome(self):
        mode_ids = self.state.valid_mode_ids
        observed = self.state.observed_experiment_ids()
        mean_separation, minimum, maximum = separation_for_modes(
            mode_ids,
            observed,
            mode_table=self.table,
        )
        matrix = PredictiveDistanceMatrix(
            mode_ids,
            observed,
            mode_table=self.table,
        )
        summary = matrix.separation_summary()
        self.assertAlmostEqual(mean_separation, summary.mean)
        self.assertAlmostEqual(minimum, summary.minimum)
        self.assertAlmostEqual(maximum, summary.maximum)

        left, right = mode_ids[:2]
        manual = sum(
            self.table.modes_by_id[left].signature[query_id][2]
            != self.table.modes_by_id[right].signature[query_id][2]
            for query_id in matrix.query_ids
        ) / len(matrix.query_ids)
        self.assertAlmostEqual(
            mode_prediction_distance(
                left,
                right,
                matrix.query_ids,
                mode_table=self.table,
            ),
            manual,
        )

    def test_oracle_normalized_predictive_diversity_recovery(self):
        matrix = PredictiveDistanceMatrix(
            self.state.valid_mode_ids,
            self.state.observed_experiment_ids(),
            mode_table=self.table,
        )
        oracle_mass, oracle_modes = matrix.optimal_subset(4)
        self.assertGreater(oracle_mass, 0.0)
        oracle = matrix.predictive_diversity_recovery(oracle_modes, budget=4)
        self.assertAlmostEqual(oracle.score, 1.0)
        self.assertEqual(oracle.valid_unique_modes, 4)

        collapsed = matrix.predictive_diversity_recovery(
            (oracle_modes[0],) * 4,
        )
        self.assertEqual(collapsed.score, 0.0)
        self.assertEqual(collapsed.valid_unique_modes, 1)

        partial = matrix.predictive_diversity_recovery(
            (oracle_modes[0], oracle_modes[1], None, "not-a-mode"),
        )
        self.assertGreaterEqual(partial.score, 0.0)
        self.assertLess(partial.score, 1.0)
        self.assertEqual(partial.valid_unique_modes, 2)

    def test_oracle_subset_is_deterministic(self):
        matrix = PredictiveDistanceMatrix(
            self.state.valid_mode_ids,
            self.state.observed_experiment_ids(),
            mode_table=self.table,
        )
        first = matrix.optimal_subset(4)
        second = matrix.optimal_subset(4)
        self.assertEqual(first, second)
        self.assertEqual(tuple(sorted(first[1])), first[1])
        self.assertTrue(math.isfinite(first[0]))

    def test_absolute_separation_bands_do_not_relabel_by_quantile(self):
        states = [
            replace(self.state, state_id="low", mean_separation=0.08),
            replace(self.state, state_id="gap", mean_separation=0.16),
            replace(self.state, state_id="medium", mean_separation=0.25),
            replace(self.state, state_id="high", mean_separation=0.40),
        ]
        bucketed = {
            state.state_id: state.separation_bucket
            for state in assign_absolute_separation_buckets(states)
        }
        self.assertEqual(bucketed["low"], "low")
        self.assertEqual(bucketed["gap"], "out_of_band")
        self.assertEqual(bucketed["medium"], "medium")
        self.assertEqual(bucketed["high"], "high")
        with self.assertRaises(ValueError):
            assign_absolute_separation_buckets(
                states,
                bands=(("a", 0.0, 0.2), ("b", 0.1, 0.3)),
            )

    def test_legacy_state_metadata_is_migrated_consistently(self):
        record = self.state.to_record(mode_table=self.table, include_private=True)
        metadata = record["metadata"]
        metadata.pop("separation_definition")
        metadata.pop("separation_targets")
        metadata["mean_separation"] = 0.999
        metadata["separation_bucket"] = "legacy-quantile"
        migrated = parse_record_state(record)
        expected = PredictiveDistanceMatrix(
            migrated.valid_mode_ids,
            migrated.observed_experiment_ids(),
            mode_table=self.table,
        ).separation_summary()
        self.assertAlmostEqual(migrated.mean_separation, expected.mean)
        self.assertIn(
            migrated.separation_bucket,
            {"low", "medium", "high", "out_of_band"},
        )
        self.assertNotEqual(migrated.separation_bucket, "legacy-quantile")

    def test_v2_continuous_selection_is_deterministic_and_excludes_overlap(self):
        candidates = []
        for index in range(12):
            candidates.append(
                replace(
                    self.state,
                    state_id=f"state-{index:02d}",
                    hidden_mode_id=f"hidden-{index:02d}",
                    evidence=(
                        EvidenceItem(
                            experiment_id=index,
                            outcome=(index % 2, (index // 2) % 2, (index // 4) % 2),
                        ),
                    ),
                    mean_separation=index / 22,
                    separation_bucket="legacy-label",
                )
            )
        excluded = candidates[3]
        kwargs = {
            "states_per_m": 6,
            "seed": 17,
            "target_counts": (8,),
            "excluded_hidden_mode_ids": {excluded.hidden_mode_id},
        }
        first = select_continuous_states(candidates, **kwargs)
        second = select_continuous_states(reversed(candidates), **kwargs)
        self.assertEqual(
            [state.state_id for state in first],
            [state.state_id for state in second],
        )
        self.assertTrue(
            all(
                state.separation_bucket == CONTINUOUS_SEPARATION_LABEL
                for state in first
            )
        )
        distribution = separation_distribution_by_m(first, target_counts=(8,))["8"]
        self.assertEqual(distribution["count"], 6)
        self.assertEqual(distribution["minimum"], 1 / 22)
        self.assertEqual(distribution["maximum"], 0.5)
        self.assertNotIn(
            excluded.hidden_mode_id, {state.hidden_mode_id for state in first}
        )

    def test_v2_overlap_audit_rejects_leakage(self):
        state = replace(
            self.state,
            state_id="audit-state",
            hidden_mode_id="test-hidden",
        )
        clean = eval_overlap_audit(
            [state],
            train_mode_ids={"train-hidden"},
            val_mode_ids={"val-hidden"},
            test_mode_ids={"test-hidden"},
        )
        assert_clean_eval_audit(clean)
        leaked = eval_overlap_audit(
            [state],
            train_mode_ids={"test-hidden"},
            val_mode_ids=set(),
            test_mode_ids={"test-hidden"},
            excluded_states=[state],
        )
        with self.assertRaises(RuntimeError):
            assert_clean_eval_audit(leaked)


if __name__ == "__main__":
    unittest.main()
