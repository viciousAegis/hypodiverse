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
    parse_hypothesis_set,
    parse_verbalized_hypothesis_set,
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
from scattered_discovery.envs.causal_micro_lab.verifier import (
    verify_output,
    verify_output_set,
    verify_verbalized_output_set,
)
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
        shorthand = "Z1: X1 AND X2\nZ2: Z1\nY: X3"
        parsed = parse_hypothesis_rules(shorthand)
        self.assertEqual(parsed.z1_rule.operator, "AND")
        self.assertEqual(parsed.z1_rule.inputs, ("X1", "X2"))
        self.assertEqual(parsed.z2_rule.operator, "COPY")
        self.assertEqual(parsed.z2_rule.inputs, ("Z1",))
        self.assertEqual(parsed.y_rule.operator, "COPY")
        self.assertEqual(parsed.y_rule.inputs, ("X3",))

        with self.assertRaises(HypothesisParseError):
            parse_hypothesis_json('{"rules": []}')
        with self.assertRaises(HypothesisParseError):
            parse_hypothesis_rules("Y = AND(Z1, Z1)")
        invalid = verify_output("Z1 = COPY(X1)", self.state, mode_table=self.table)
        self.assertFalse(invalid.parse_valid)

    def test_multi_answer_parser_and_verifier(self):
        valid_text = self.table.modes_by_id[
            self.state.valid_mode_ids[0]
        ].canonical.render_flat_rules()
        second_valid = self.table.modes_by_id[
            self.state.valid_mode_ids[1]
        ].canonical.render_flat_rules()
        payload = (
            f"<answer1>\n{valid_text}\n</answer1>\n"
            "<answer2>\nnot a hypothesis\n</answer2>\n"
            f"<answer4>\n{second_valid}\n</answer4>\n"
        )
        candidates = parse_hypothesis_set(payload, expected_count=4)
        self.assertEqual([item.index for item in candidates], [1, 2, 4])
        result = verify_output_set(
            payload,
            self.state,
            expected_count=4,
            mode_table=self.table,
        )
        self.assertFalse(result.format_valid)
        self.assertEqual(result.candidate_count, 3)
        self.assertEqual(result.parse_valid_count, 2)
        self.assertEqual(len(result.unique_valid_mode_ids), 2)
        self.assertEqual(result.coverage_per_k(), 0.5)
        self.assertEqual(result.coverage_per_available(self.state), 0.5)

    def test_verbalized_sampling_parser_and_verifier(self):
        parts = []
        for index, mode_id in enumerate(self.state.valid_mode_ids, start=1):
            parts.extend(
                [
                    f"ANSWER {index}",
                    self.table.modes_by_id[mode_id].canonical.render_flat_rules(),
                    f"PROBABILITY: {0.1 * index}",
                ]
            )
        payload = "\n".join(parts)
        candidates = parse_verbalized_hypothesis_set(payload, expected_count=4)
        self.assertEqual([item.index for item in candidates], [1, 2, 3, 4])
        for actual, expected in zip(
            [item.probability for item in candidates],
            [0.1, 0.2, 0.3, 0.4],
            strict=True,
        ):
            self.assertAlmostEqual(actual or 0.0, expected)

        result = verify_verbalized_output_set(
            payload,
            self.state,
            expected_count=4,
            mode_table=self.table,
        )
        self.assertTrue(result.format_valid)
        self.assertTrue(result.probability_format_valid)
        self.assertAlmostEqual(result.probability_sum_error or 0.0, 0.0)
        self.assertIsNotNone(result.probability_entropy)
        self.assertEqual(len(result.unique_valid_mode_ids), 4)
        self.assertEqual(result.coverage_per_available(self.state), 1.0)

        malformed = payload.replace("PROBABILITY: 0.4", "PROBABILITY: 0.3")
        malformed_result = verify_verbalized_output_set(
            malformed,
            self.state,
            expected_count=4,
            mode_table=self.table,
        )
        self.assertFalse(malformed_result.format_valid)
        self.assertFalse(malformed_result.probability_format_valid)
        self.assertAlmostEqual(malformed_result.probability_sum_error or 0.0, 0.1)

        extra_answer = f"{payload}\nANSWER 5\n{parts[1]}\nPROBABILITY: 0"
        extra_result = verify_verbalized_output_set(
            extra_answer,
            self.state,
            expected_count=4,
            mode_table=self.table,
        )
        self.assertFalse(extra_result.format_valid)
        self.assertEqual(extra_result.candidate_count, 5)

    def test_verbalized_sampling_prompt_uses_plain_blocks(self):
        prompt = build_prompt(
            self.state,
            output_mode="verbalized_sampling",
            answer_count=4,
        )
        self.assertIn("ANSWER 1", prompt)
        self.assertIn("ANSWER 4", prompt)
        self.assertIn("PROBABILITY: 0.25", prompt)
        self.assertIn("Z2: COPY Z1", prompt)
        self.assertIn("Z1: AND X1 X2", prompt)
        self.assertNotIn("<answer", prompt)
        self.assertNotIn("valid_mode_ids", prompt)
        prompt_k8 = build_prompt(
            self.state,
            output_mode="verbalized_sampling",
            answer_count=8,
        )
        self.assertIn("ANSWER 8", prompt_k8)
        self.assertIn("PROBABILITY: 0.125", prompt_k8)

    def test_env_rewards_nonempty_final_output(self):
        env = make_env(
            {
                "env_type": "causal_micro_lab",
                "task": {"state": state_rows([self.state])[0]},
            }
        )
        invalid_nonempty = env.step("not a hypothesis")
        self.assertEqual(invalid_nonempty.score.reward, 0.0)
        self.assertEqual(
            invalid_nonempty.score.breakdown.as_dict()["nonempty_output"],
            0.0,
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
                    "valid_hypothesis_reward": 1.5,
                },
            }
        )
        valid = valid_env.step(valid_mode.canonical.render_flat_rules())
        self.assertEqual(valid.score.reward, 1.5)
        self.assertEqual(valid.score.breakdown.as_dict()["valid_hypothesis"], 1.5)
        self.assertEqual(valid.score.breakdown.as_dict()["nonempty_output"], 0.0)

        invalid_mode = next(
            mode
            for mode in self.table.modes
            if mode.mode_id not in self.state.valid_mode_ids
        )
        default_invalid_env = make_env(
            {
                "env_type": "causal_micro_lab",
                "task": {"state": state_rows([self.state])[0]},
            }
        )
        default_invalid = default_invalid_env.step(
            invalid_mode.canonical.render_flat_rules()
        )
        self.assertEqual(default_invalid.score.reward, 0.2)
        default_breakdown = default_invalid.score.breakdown.as_dict()
        self.assertEqual(default_breakdown["nonempty_output"], 0.0)
        self.assertEqual(default_breakdown["format"], 0.0)
        self.assertEqual(default_breakdown["commit_format"], 0.2)

        dense_env = make_env(
            {
                "env_type": "causal_micro_lab",
                "task": {
                    "state": state_rows([self.state])[0],
                    "nonempty_output_reward": 0.2,
                    "rule_marker_reward": 0.05,
                    "parse_valid_reward": 0.1,
                    "syntax_valid_reward": 0.2,
                    "evidence_consistent_reward": 0.4,
                },
            }
        )
        dense = dense_env.step(invalid_mode.canonical.render_flat_rules())
        dense_breakdown = dense.score.breakdown.as_dict()
        self.assertFalse(dense.metrics["evidence_consistent"])
        self.assertEqual(dense_breakdown["nonempty_output"], 0.2)
        self.assertEqual(dense_breakdown["format"], 0.05)
        self.assertEqual(dense_breakdown["admissible"], 0.1)
        self.assertEqual(dense_breakdown["commit_format"], 0.2)
        self.assertEqual(dense.score.reward, 0.55)

    def test_multi_answer_rlvr_env_reward(self):
        valid_text = self.table.modes_by_id[
            self.state.valid_mode_ids[0]
        ].canonical.render_flat_rules()
        parts = []
        for index in range(1, 5):
            parts.extend(
                [
                    f"<answer{index}>",
                    valid_text if index == 1 else "not a hypothesis",
                    f"</answer{index}>",
                ]
            )
        env = make_env(
            {
                "env_type": "causal_micro_lab",
                "task": {
                    "state": self.state.to_record(
                        mode_table=self.table,
                        include_private=True,
                    ),
                    "output_mode": "multi_answer_rlvr",
                    "answer_count": 4,
                    "multi_answer_format_reward": 0.5,
                    "multi_answer_accuracy_reward": 0.5,
                    "multi_answer_accuracy_mode": "any_valid",
                },
                "max_steps": 1,
            }
        )
        step = env.step("\n".join(parts))
        self.assertTrue(step.parse_ok)
        self.assertAlmostEqual(step.reward, 1.0)
        self.assertEqual(step.metrics["multi_answer_expected_count"], 4)
        self.assertEqual(step.metrics["multi_answer_unique_valid_modes"], 1)
        self.assertEqual(step.metrics["multi_answer_coverage_per_k"], 0.25)

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
        self.assertLessEqual(
            trace.final_version_space_size(), self.state.valid_mode_count
        )

    def test_dataset_rows_and_oracle_eval(self):
        rows = state_rows([self.state], mode_table=self.table)
        self.assertIn("private", rows[0])
        self.assertIn("maximum_separation", rows[0]["metadata"])
        sft_rows = sft_rows_for_states(
            [self.state], targets_per_state=2, mode_table=self.table
        )
        self.assertEqual(len(sft_rows), 2)
        self.assertIn("Z1:", sft_rows[0]["response"])
        self.assertNotIn('"rules"', sft_rows[0]["response"])
        verl_rows = verl_rows_for_states(
            [self.state],
            task_overrides={
                "nonempty_output_reward": 0.0,
                "syntax_valid_reward": 0.2,
                "valid_hypothesis_reward": 1.0,
                "output_mode": "multi_answer_rlvr",
                "answer_count": 4,
                "multi_answer_accuracy_mode": "any_valid",
            },
            agent_overrides={
                "length_penalty_start": 3072,
                "length_penalty_max": -0.2,
                "mask_truncated": False,
            },
            mode_table=self.table,
        )
        parsed = json.loads(verl_rows[0]["env_spec_json"])
        self.assertEqual(parsed["env_type"], "causal_micro_lab")
        self.assertEqual(parsed["task"]["nonempty_output_reward"], 0.0)
        self.assertEqual(parsed["task"]["syntax_valid_reward"], 0.2)
        self.assertEqual(parsed["task"]["valid_hypothesis_reward"], 1.0)
        self.assertEqual(parsed["task"]["output_mode"], "multi_answer_rlvr")
        self.assertEqual(parsed["task"]["answer_count"], 4)
        self.assertEqual(parsed["task"]["multi_answer_accuracy_mode"], "any_valid")
        self.assertIn("<answer4>", verl_rows[0]["prompt"])
        self.assertEqual(parsed["agent"]["length_penalty_start"], 3072)
        self.assertEqual(parsed["agent"]["length_penalty_max"], -0.2)
        self.assertFalse(parsed["agent"]["mask_truncated"])
        env_from_row = make_env(parsed)
        self.assertIsNotNone(env_from_row.reset())
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
                for line in outputs["sft_train"]
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertTrue(train_rows)
            self.assertIn(train_rows[0]["target_mode_id"], split_ids["train"])
            state_rows_from_disk = [
                json.loads(line)
                for line in outputs["states_train"]
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertIn("private", state_rows_from_disk[0])
            self.assertTrue(outputs["manifest"].exists())

    def test_split_dataset_builder_caps_val_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = build_split_dataset(
                output_dir=tmpdir,
                target_counts=(2, 4),
                states_per_count={"train": 1, "val": 2, "test": 1},
                max_rows_per_split={"val": 3},
                seed=11,
                beam_width=32,
                targets_per_state=1,
                mode_table=self.table,
            )
            val_rows = [
                json.loads(line)
                for line in outputs["verl_val"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(val_rows), 3)
            manifest = json.loads(
                outputs["manifest"].read_text(encoding="utf-8").splitlines()[-1]
            )
            self.assertEqual(manifest["max_rows_per_split"]["val"], 3)

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

    def test_causal_micro_lab_eval_cycles_latent_prompt_labels(self):
        valid_mode = self.table.modes_by_id[self.state.valid_mode_ids[0]]

        class LatentBackend:
            def __init__(self):
                self.prompts = []

            def chat(self, messages, options=None):
                del options
                self.prompts.append(messages[-1].content)
                return ChatResponse(content=valid_mode.canonical.render_rules())

        backend = LatentBackend()
        records = evaluate_states(
            states=[self.state],
            backend=backend,
            model="static",
            rollouts_per_state=3,
            latent_count=2,
        )

        self.assertEqual(
            [prompt.split(" |", 1)[0] for prompt in backend.prompts],
            ["Strategy 1", "Strategy 2", "Strategy 1"],
        )
        self.assertEqual([record["latent_id"] for record in records], [1, 2, 1])
        self.assertTrue(
            all(record["verification"]["is_currently_valid_mode"] for record in records)
        )

    def test_causal_micro_lab_eval_finalizes_length_capped_thinking(self):
        valid_mode = self.table.modes_by_id[self.state.valid_mode_ids[0]]

        class FallbackBackend:
            def __init__(self):
                self.calls = []

            def chat(self, messages, options=None):
                self.calls.append((messages, options))
                if len(self.calls) == 1:
                    return ChatResponse(
                        content="",
                        thinking="I should use a direct rule.",
                        finish_reason="length",
                        completion_tokens=4096,
                    )
                return ChatResponse(
                    content=valid_mode.canonical.render_rules(),
                    finish_reason="stop",
                    completion_tokens=24,
                )

        backend = FallbackBackend()
        records = evaluate_states(
            states=[self.state],
            backend=backend,
            model="static",
            rollouts_per_state=1,
            thinking_fallback=True,
            fallback_num_predict=256,
        )
        self.assertEqual(len(backend.calls), 2)
        self.assertFalse(backend.calls[1][1].think)
        self.assertEqual(backend.calls[1][1].num_predict, 256)
        self.assertTrue(records[0]["fallback_used"])
        self.assertTrue(records[0]["fallback_produced_output"])
        self.assertEqual(records[0]["initial_completion_tokens"], 4096)
        self.assertTrue(records[0]["verification"]["is_currently_valid_mode"])

    def test_verbalized_sampling_eval_fallback(self):
        parts = []
        for index, mode_id in enumerate(self.state.valid_mode_ids, start=1):
            parts.extend(
                [
                    f"ANSWER {index}",
                    self.table.modes_by_id[mode_id].canonical.render_flat_rules(),
                    "PROBABILITY: 0.25",
                ]
            )
        final_output = "\n".join(parts)

        class FallbackBackend:
            def __init__(self):
                self.calls = []

            def chat(self, messages, options=None):
                self.calls.append((messages, options))
                if len(self.calls) == 1:
                    return ChatResponse(
                        content="",
                        thinking="I should compare several possible programs.",
                        finish_reason="length",
                        completion_tokens=16000,
                    )
                return ChatResponse(
                    content=final_output,
                    finish_reason="stop",
                    completion_tokens=128,
                )

        backend = FallbackBackend()
        records = evaluate_states(
            states=[self.state],
            backend=backend,
            model="static",
            output_mode="verbalized_sampling",
            answer_count=4,
            thinking_fallback=True,
            fallback_num_predict=512,
        )
        self.assertEqual(len(backend.calls), 2)
        self.assertFalse(backend.calls[1][1].think)
        self.assertEqual(backend.calls[1][1].num_predict, 512)
        self.assertIn(
            "plain ANSWER 1 through ANSWER 4 blocks", backend.calls[1][0][-1].content
        )
        verification = records[0]["verification"]
        self.assertEqual(verification["output_mode"], "verbalized_sampling")
        self.assertTrue(verification["format_valid"])
        self.assertTrue(verification["probability_format_valid"])
        self.assertEqual(verification["valid_count"], 4)

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
