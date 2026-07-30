from __future__ import annotations

import unittest
from dataclasses import replace

from scattered_discovery.envs.causal_micro_lab.closed_loop import (
    TrajectoryRuntime,
    _restore_runtimes,
    _step_record,
    select_initial_states,
    summarize_trajectories,
)
from scattered_discovery.envs.causal_micro_lab.planner import (
    oracle_disagreement_experiment,
    prediction_distributions,
)
from scattered_discovery.envs.causal_micro_lab.prompt_builder import build_prompt
from scattered_discovery.envs.causal_micro_lab.signatures import build_mode_table
from scattered_discovery.envs.causal_micro_lab.state_generator import (
    find_states,
    make_state,
)
from scattered_discovery.envs.causal_micro_lab.verifier import verify_output


class CausalMicroLabClosedLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = build_mode_table()
        cls.state = find_states(
            0,
            4,
            max_evidence=8,
            beam_width=32,
            mode_table=cls.table,
        )[0]

    def _generation_record(self, mode_id: str, rollout_index: int) -> dict:
        output = self.table.modes_by_id[mode_id].canonical.render_flat_rules()
        verification = verify_output(
            output,
            self.state,
            mode_table=self.table,
        )
        return {
            "state_index": 0,
            "rollout_index": rollout_index,
            "latent_id": 0,
            "prompt": build_prompt(self.state),
            "output": output,
            "thinking": "",
            "verification": verification.as_dict(),
            "request_error": None,
            "model_seconds": 0.1,
            "initial_finish_reason": "stop",
            "initial_completion_tokens": 20,
            "fallback_used": False,
            "fallback_finish_reason": None,
            "fallback_completion_tokens": None,
        }

    def test_duplicate_predictions_are_preserved(self):
        left, right = self.state.valid_mode_ids[:2]
        mode_ids = [left, left, left, right]
        distributions = prediction_distributions(
            mode_ids,
            self.state,
            mode_table=self.table,
        )
        self.assertTrue(distributions)
        self.assertTrue(
            any(
                outcome["count"] == 3
                for distribution in distributions.values()
                for outcome in distribution["outcomes"].values()
            )
        )

    def test_step_updates_evidence_and_keeps_private_data_out_of_prompt(self):
        left, right = self.state.valid_mode_ids[:2]
        records = [
            self._generation_record(mode_id, index)
            for index, mode_id in enumerate([left, left, left, right])
        ]
        runtime = TrajectoryRuntime(
            index=0,
            initial=self.state,
            current=self.state,
        )
        record, updated = _step_record(
            runtime,
            records,
            run_name="test",
            model="oracle",
            seed=1,
            deduplicate_planner_modes=False,
            mode_table=self.table,
        )
        self.assertEqual(record["planner_mode_ids"], [left, left, left, right])
        self.assertNotIn("hidden_mode_id", record["prompt"])
        self.assertNotIn(self.state.hidden_mode_id, record["prompt"])
        self.assertEqual(updated.evidence_size, self.state.evidence_size + 1)
        self.assertLessEqual(updated.valid_mode_count, self.state.valid_mode_count)

        restored = _restore_runtimes(
            [self.state],
            [record],
            mode_table=self.table,
        )
        self.assertEqual(restored[0].current.state_id, updated.state_id)
        self.assertEqual(len(restored[0].records), 1)

    def test_initial_state_selection_uses_both_counts_and_unique_worlds(self):
        modes = list(self.table.modes_by_id)
        candidates = []
        for mode_count in (16, 32):
            for index, bucket in enumerate(("low", "medium", "high")):
                candidates.append(
                    replace(
                        self.state,
                        state_id=f"{mode_count}-{bucket}",
                        hidden_mode_id=modes.pop(),
                        valid_mode_ids=tuple(
                            list(self.table.modes_by_id)[:mode_count]
                        ),
                        separation_bucket=bucket,
                    )
                )
        selected = select_initial_states(
            candidates,
            initial_mode_counts=(16, 32),
            trajectories_per_count=3,
            seed=7,
        )
        self.assertEqual(
            [state.valid_mode_count for state in selected],
            [16, 16, 16, 32, 32, 32],
        )
        self.assertEqual(len({state.hidden_mode_id for state in selected}), 6)

    def test_budget_efficiency_uses_full_budget_for_failures(self):
        failing = TrajectoryRuntime(
            index=0,
            initial=self.state,
            current=self.state,
        )
        summary = summarize_trajectories(
            [failing],
            max_steps=8,
            bootstrap_samples=0,
            seed=1,
            mode_table=self.table,
        )
        self.assertEqual(summary["mean_experiments_used"], 8.0)
        self.assertEqual(summary["identifications_per_100_experiments"], 0.0)
        self.assertEqual(
            summary["endpoint_rows"][0]["experiments_executed"],
            0,
        )

    def test_successful_trajectory_stops_charging_at_identification(self):
        state = find_states(
            0,
            2,
            max_evidence=8,
            beam_width=32,
            mode_table=self.table,
        )[0]
        experiment_id = oracle_disagreement_experiment(
            state,
            mode_table=self.table,
        )
        self.assertIsNotNone(experiment_id)
        updated = make_state(
            hidden_mode=self.table.modes_by_id[state.hidden_mode_id],
            evidence_ids=tuple(
                sorted((*state.observed_experiment_ids(), experiment_id))
            ),
            mode_table=self.table,
            compute_separation=True,
        )
        self.assertEqual(updated.valid_mode_count, 1)
        runtime = TrajectoryRuntime(
            index=0,
            initial=state,
            current=updated,
            records=[
                {
                    "experiment_step": 1,
                    "version_space_size_after": 1,
                    "entropy_regret": 0.0,
                    "identified": True,
                }
            ],
        )
        summary = summarize_trajectories(
            [runtime],
            max_steps=8,
            bootstrap_samples=0,
            seed=1,
            mode_table=self.table,
        )
        self.assertEqual(summary["mean_experiments_used"], 1.0)
        self.assertEqual(summary["identifications_per_100_experiments"], 100.0)


if __name__ == "__main__":
    unittest.main()
