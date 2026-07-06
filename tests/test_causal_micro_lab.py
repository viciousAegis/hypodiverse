import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scattered_discovery.envs.base import EnvSpec
from scattered_discovery.envs.causal_micro_lab.dsl import CausalMicroLabError, Rule
from scattered_discovery.envs.causal_micro_lab.enumerate_hypotheses import (
    enumerate_hypotheses,
)
from scattered_discovery.envs.causal_micro_lab.interventions import (
    Experiment,
    enumerate_experiments,
)
from scattered_discovery.envs.causal_micro_lab.parser import (
    HypothesisParseError,
    parse_hypothesis,
    parse_hypothesis_json,
    parse_hypothesis_rules,
)
from scattered_discovery.envs.causal_micro_lab.planner import (
    oracle_disagreement_experiment,
    run_oracle_closed_loop,
    select_disagreement_experiment,
)
from scattered_discovery.envs.causal_micro_lab.prompt_builder import build_prompt
from scattered_discovery.envs.causal_micro_lab.rewards import group_metrics
from scattered_discovery.envs.causal_micro_lab.signatures import (
    build_mode_table,
    mode_id_for_signature,
)
from scattered_discovery.envs.causal_micro_lab.simulator import (
    prediction_signature,
    run_experiment,
)
from scattered_discovery.envs.causal_micro_lab.state_generator import (
    find_states,
    valid_modes_for_evidence,
)
from scattered_discovery.envs.causal_micro_lab.tables import (
    build_split_dataset,
    oracle_group_eval,
    sft_rows_for_states,
    split_mode_ids,
    state_rows,
    verl_rows_for_states,
    write_table,
)
from scattered_discovery.envs.causal_micro_lab.verifier import verify_output
from scattered_discovery.envs.causal_micro_lab.eval import (
    evaluate_states,
    summarize_grouped_records,
    summarize_records,
)
from scattered_discovery.backends.base import ChatResponse
from scattered_discovery.envs.factory import make_env
from scattered_discovery.causal_micro_lab_viewer import (
    BANKS,
    bank_summary,
    create_bank,
    state_snapshot,
    verify_payload,
)
from scattered_discovery.verl.agent_loop import CausalMicroLabAgentLoop


class CausalMicroLabTests(unittest.TestCase):
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

    def test_enumeration_counts_and_stable_modes(self):
        self.assertEqual(len(enumerate_experiments()), 40)
        self.assertEqual(len(enumerate_hypotheses()), 15600)
        rebuilt = build_mode_table()
        self.assertEqual(len(self.table.modes), len(rebuilt.modes))
        self.assertEqual(
            [mode.mode_id for mode in self.table.modes[:20]],
            [mode.mode_id for mode in rebuilt.modes[:20]],
        )

    def test_rule_validation_rejects_bad_arity_and_noncanonical_binary_inputs(self):
        with self.assertRaises(CausalMicroLabError):
            Rule(target="Z1", operator="COPY", inputs=("X1", "X2"))
        with self.assertRaises(CausalMicroLabError):
            Rule(target="Z2", operator="AND", inputs=("Z1", "X1"))
        with self.assertRaises(CausalMicroLabError):
            Rule(target="Y", operator="OR", inputs=("Z1", "Z1"))

    def test_simulator_intervention_semantics(self):
        hypothesis = parse_hypothesis_json(
            json.dumps(
                {
                    "rules": [
                        {"target": "Z1", "operator": "COPY", "inputs": ["X1"]},
                        {"target": "Z2", "operator": "COPY", "inputs": ["Z1"]},
                        {"target": "Y", "operator": "COPY", "inputs": ["Z2"]},
                    ]
                }
            )
        )
        observed = run_experiment(
            hypothesis,
            Experiment(0, (1, 0, 0), "OBSERVE"),
        )
        intervened = run_experiment(
            hypothesis,
            Experiment(1, (1, 0, 0), "DO_Z1_0"),
        )
        self.assertEqual(observed, (1, 1, 1))
        self.assertEqual(intervened, (0, 0, 0))

    def test_semantic_mode_id_matches_signature(self):
        mode = self.table.modes[0]
        signature = prediction_signature(mode.canonical, self.table.experiments)
        self.assertEqual(mode.mode_id, mode_id_for_signature(signature))

    def test_state_generation_and_prompt_privacy(self):
        for target in (2, 4, 8, 16):
            states = find_states(
                0,
                target,
                max_evidence=8,
                beam_width=32,
                mode_table=self.table,
            )
            self.assertTrue(states)
            self.assertEqual(states[0].valid_mode_count, target)
        prompt = build_prompt(self.state)
        self.assertIn("Evidence:", prompt)
        self.assertIn("Experiment 1:", prompt)
        self.assertIn("Return exactly three lines", prompt)
        self.assertNotIn("valid_mode_ids", prompt)
        self.assertNotIn("hidden_mode_id", prompt)

    def test_parser_and_verifier_accept_valid_and_reject_invalid(self):
        valid_mode = self.table.modes_by_id[self.state.valid_mode_ids[0]]
        valid_text = valid_mode.canonical.render_json()
        result = verify_output(valid_text, self.state, mode_table=self.table)
        self.assertTrue(result.parse_valid)
        self.assertTrue(result.evidence_consistent)
        self.assertTrue(result.is_currently_valid_mode)
        rule_text = valid_mode.canonical.render_rules()
        self.assertEqual(
            parse_hypothesis(rule_text),
            parse_hypothesis_rules(rule_text),
        )
        rule_result = verify_output(rule_text, self.state, mode_table=self.table)
        self.assertTrue(rule_result.parse_valid)
        self.assertTrue(rule_result.is_currently_valid_mode)
        flat_text = valid_mode.canonical.render_flat_rules()
        flat_result = verify_output(flat_text, self.state, mode_table=self.table)
        self.assertTrue(flat_result.parse_valid)
        self.assertTrue(flat_result.is_currently_valid_mode)
        reversed_binary = "Z1: AND X2 X1\nZ2: COPY X2\nY: COPY X3"
        parsed = parse_hypothesis_rules(reversed_binary)
        self.assertEqual(parsed.z1_rule.inputs, ("X1", "X2"))
        no_paren_unary = "Z1 = NOT X2\nZ2: COPY X2\nY: COPY X3"
        parsed = parse_hypothesis_rules(no_paren_unary)
        self.assertEqual(parsed.z1_rule.inputs, ("X2",))

        with self.assertRaises(HypothesisParseError):
            parse_hypothesis_json('{"rules": []}')
        with self.assertRaises(HypothesisParseError):
            parse_hypothesis_rules("Y = AND(Z1, Z1)")
        invalid = verify_output("Z1 = COPY(X1)", self.state, mode_table=self.table)
        self.assertFalse(invalid.parse_valid)

    def test_env_rewards_nonempty_final_output(self):
        env = make_env(
            {
                "env_type": "causal_micro_lab",
                "task": {"state": state_rows([self.state])[0]},
            }
        )
        invalid_nonempty = env.step("not a hypothesis")
        self.assertEqual(invalid_nonempty.score.reward, 0.2)
        self.assertEqual(
            invalid_nonempty.score.breakdown.as_dict()["nonempty_output"],
            0.2,
        )
        self.assertEqual(invalid_nonempty.metrics["nonempty_output"], 1.0)

        empty_env = make_env(
            {
                "env_type": "causal_micro_lab",
                "task": {"state": state_rows([self.state])[0]},
            }
        )
        empty = empty_env.step("")
        self.assertEqual(empty.score.reward, 0.0)
        self.assertEqual(empty.score.breakdown.as_dict()["nonempty_output"], 0.0)
        self.assertEqual(empty.metrics["nonempty_output"], 0.0)

        valid_mode = self.table.modes_by_id[self.state.valid_mode_ids[0]]
        valid_env = make_env(
            {
                "env_type": "causal_micro_lab",
                "task": {
                    "state": state_rows([self.state])[0],
                    "nonempty_output_reward": 0.1,
                },
            }
        )
        valid = valid_env.step(valid_mode.canonical.render_flat_rules())
        self.assertEqual(valid.score.reward, 1.0)
        self.assertEqual(valid.score.breakdown.as_dict()["valid_hypothesis"], 1.0)
        self.assertEqual(valid.score.breakdown.as_dict()["nonempty_output"], 0.0)

    def test_version_space_group_metrics_and_planner(self):
        self.assertEqual(
            set(valid_modes_for_evidence(self.state.evidence, mode_table=self.table)),
            set(self.state.valid_mode_ids),
        )
        outputs = [
            self.table.modes_by_id[mode_id].canonical.render_json()
            for mode_id in self.state.valid_mode_ids[:2]
        ]
        results = [
            verify_output(output, self.state, mode_table=self.table)
            for output in outputs
        ]
        metrics = group_metrics(results, self.state)
        self.assertEqual(metrics["num_unique_valid_modes"], 2.0)
        self.assertEqual(metrics["available_valid_modes"], 4.0)
        self.assertEqual(metrics["exact_coverage"], 0.5)
        selected = select_disagreement_experiment(
            list(self.state.valid_mode_ids[:2]),
            self.state,
            mode_table=self.table,
        )
        self.assertIsNotNone(selected)
        self.assertNotIn(selected, self.state.observed_experiment_ids())
        self.assertIsNotNone(
            oracle_disagreement_experiment(self.state, mode_table=self.table)
        )
        trace = run_oracle_closed_loop(self.state, max_steps=8, mode_table=self.table)
        self.assertLessEqual(trace.final_version_space_size(), self.state.valid_mode_count)

    def test_dataset_rows_and_oracle_eval(self):
        rows = state_rows([self.state], mode_table=self.table)
        self.assertIn("private", rows[0])
        self.assertIn("maximum_separation", rows[0]["metadata"])
        sft_rows = sft_rows_for_states([self.state], targets_per_state=2, mode_table=self.table)
        self.assertEqual(len(sft_rows), 2)
        self.assertIn("Z1:", sft_rows[0]["response"])
        self.assertNotIn('"rules"', sft_rows[0]["response"])
        verl_rows = verl_rows_for_states([self.state], mode_table=self.table)
        parsed = json.loads(verl_rows[0]["env_spec_json"])
        self.assertEqual(parsed["env_type"], "causal_micro_lab")
        metrics = oracle_group_eval(self.state, samples=4, mode_table=self.table)
        self.assertEqual(metrics["budget_normalized_coverage"], 1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_table(rows, Path(tmpdir) / "states.jsonl")
            self.assertTrue(path.exists())

    def test_split_dataset_builder_outputs_all_splits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = build_split_dataset(
                output_dir=tmpdir,
                target_counts=(2,),
                states_per_count={"train": 1, "val": 1, "test": 1},
                seed=7,
                beam_width=32,
                targets_per_state=1,
                mode_table=self.table,
            )
            for split in ("train", "val", "test"):
                self.assertTrue(outputs[f"states_{split}"].exists())
                self.assertTrue(outputs[f"sft_{split}"].exists())
                self.assertTrue(outputs[f"verl_{split}"].exists())

            split_ids = split_mode_ids(seed=7, mode_table=self.table)
            train_rows = [
                json.loads(line)
                for line in outputs["sft_train"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(train_rows)
            self.assertIn(train_rows[0]["target_mode_id"], split_ids["train"])
            state_rows_from_disk = [
                json.loads(line)
                for line in outputs["states_train"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertIn("private", state_rows_from_disk[0])
            self.assertTrue(outputs["manifest"].exists())

    def test_build_eval_rows_from_frozen_states_cli(self):
        from scattered_discovery.envs.causal_micro_lab.cli import (
            build_eval_rows_from_states_main,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            states_path = Path(tmpdir) / "states_val.jsonl"
            output_path = Path(tmpdir) / "verl_val.jsonl"
            write_table(state_rows([self.state], mode_table=self.table), states_path)
            argv = [
                "causal-micro-lab-build-eval-rows",
                "--states",
                str(states_path),
                "--output",
                str(output_path),
                "--data-source-prefix",
                "test_source",
            ]
            with patch("sys.argv", argv):
                build_eval_rows_from_states_main()
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["data_source"], "test_source")
            spec = json.loads(rows[0]["env_spec_json"])
            self.assertEqual(spec["env_type"], "causal_micro_lab")
            self.assertEqual(spec["task"]["state"]["state_id"], self.state.state_id)
            self.assertEqual(rows[0]["extra_info"]["min_global_steps"], 0)
            self.assertEqual(rows[0]["extra_info"]["max_global_steps"], 0)

    def test_causal_micro_lab_eval_summary(self):
        records = [
            {
                "model_seconds": 0.1,
                "output": "Z1 = COPY(X1)",
                "verification": {
                    "parse_valid": True,
                    "syntax_valid": True,
                    "evidence_consistent": True,
                    "is_currently_valid_mode": True,
                    "semantic_mode_id": "a",
                },
                "state_metadata": {
                    "valid_mode_count": 4,
                    "separation_bucket": "low",
                    "family_bucket": "mixed",
                },
            },
            {
                "model_seconds": 0.2,
                "output": "",
                "verification": {
                    "parse_valid": False,
                    "syntax_valid": False,
                    "evidence_consistent": False,
                    "is_currently_valid_mode": False,
                    "semantic_mode_id": None,
                },
                "state_metadata": {
                    "valid_mode_count": 8,
                    "separation_bucket": "high",
                    "family_bucket": "mixed",
                },
            },
        ]
        summary = summarize_records(records)
        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["parse_valid"], 0.5)
        self.assertEqual(summary["unique_valid_modes"], 1)
        self.assertEqual(summary["invalid_format_count"], 1)
        self.assertTrue(summary["invalid_format_errors"])
        self.assertIn("4", summary["by_M"])

    def test_causal_micro_lab_grouped_eval_duplicate_floor(self):
        mode_ids = list(self.state.valid_mode_ids)
        records = []
        for rollout_index, mode_id in enumerate(mode_ids + mode_ids):
            records.append(
                {
                    "sample_id": f"{self.state.state_id}:sample{rollout_index:04d}",
                    "state_id": self.state.state_id,
                    "rollout_index": rollout_index,
                    "verification": {
                        "parse_valid": True,
                        "syntax_valid": True,
                        "evidence_consistent": True,
                        "is_currently_valid_mode": True,
                        "semantic_mode_id": mode_id,
                        "prediction_signature": None,
                        "mechanism_family": None,
                        "error": None,
                    },
                }
            )
        summary = summarize_grouped_records(records, [self.state])
        self.assertEqual(summary["states"], 1)
        self.assertEqual(summary["pass_at_k"], 1.0)
        self.assertEqual(summary["exact_coverage"], 1.0)
        self.assertEqual(summary["budget_normalized_coverage"], 1.0)
        self.assertEqual(summary["duplicate_valid_modes"], 4.0)
        self.assertEqual(summary["unavoidable_duplicate_valid_modes"], 4.0)
        self.assertEqual(summary["extra_duplicate_valid_modes"], 0.0)
        prefix_summary = summarize_grouped_records(
            records,
            [self.state],
            max_rollout_index=4,
        )
        self.assertEqual(prefix_summary["k"], 4.0)
        self.assertEqual(prefix_summary["duplicate_valid_modes"], 0.0)

    def test_causal_micro_lab_eval_records_have_stable_sample_ids(self):
        class StaticBackend:
            def chat(self, messages, options=None):
                del messages, options
                valid_mode = self_outer.table.modes_by_id[
                    self_outer.state.valid_mode_ids[0]
                ]
                return ChatResponse(content=valid_mode.canonical.render_rules())

        self_outer = self
        records = evaluate_states(
            states=[self.state],
            backend=StaticBackend(),
            model="static",
            rollouts_per_state=2,
        )
        self.assertEqual(
            [record["sample_id"] for record in records],
            [
                f"{self.state.state_id}:sample0000",
                f"{self.state.state_id}:sample0001",
            ],
        )

    def test_factory_env_and_agent_loop_import(self):
        env = make_env(
            EnvSpec(
                env_type="causal_micro_lab",
                task={"state": self.state.to_record(mode_table=self.table)},
                max_steps=1,
                seed=1,
            )
        )
        self.assertIn("single-shot", env.system_prompt())
        valid_mode = self.table.modes_by_id[self.state.valid_mode_ids[0]]
        step = env.step(valid_mode.canonical.render_json())
        self.assertTrue(step.done)
        self.assertEqual(step.score.valid_unique_count, 1)
        self.assertIsNotNone(CausalMicroLabAgentLoop)

    def test_viewer_snapshot_and_verify_payload(self):
        bank_id, bank = create_bank(
            {
                "target_counts": "2",
                "states_per_count": 1,
                "seed": 3,
                "beam_width": 32,
                "separation_bucket": "low",
            },
            mode_table=self.table,
        )
        self.addCleanup(lambda: BANKS.pop(bank_id, None))
        summary = bank_summary(bank_id, bank, mode_table=self.table)
        self.assertEqual(summary["state_count"], 1)
        self.assertEqual(summary["separation_bucket"], "low")
        snapshot = state_snapshot(bank_id, bank, mode_table=self.table)
        selected = snapshot["selected"]
        self.assertIn("available_experiments", selected)
        self.assertIn("maximum_separation", selected)
        self.assertIn("m_trajectory", selected)
        self.assertEqual(selected["separation_bucket"], "low")
        self.assertIn("prompt", selected)
        self.assertNotIn("valid_mode_ids", selected["prompt"])
        result = verify_payload(
            {
                "bank_id": bank_id,
                "state_id": selected["state_id"],
                "text": selected["valid_mode_examples"][0]["canonical_json"],
            },
            mode_table=self.table,
        )
        self.assertTrue(result["verification"]["is_currently_valid_mode"])


if __name__ == "__main__":
    unittest.main()
