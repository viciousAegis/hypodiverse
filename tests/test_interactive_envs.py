import json
import tempfile
import unittest
from pathlib import Path

from scattered_discovery.backends.base import ChatMessage, ChatResponse
from scattered_discovery.envs.base import EnvSpec
from scattered_discovery.envs.factory import make_env
from scattered_discovery.envs.hypospace_boolean import parse_boolean_expr, truth_table
from scattered_discovery.rollout.local_loop import run_local_episode
from scattered_discovery.verl.make_dataset import (
    build_specs,
    generate_from_config,
    specs_to_rows,
    write_jsonl,
)


class ScriptedBackend:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[ChatMessage]] = []

    def chat(self, messages: list[ChatMessage], options=None) -> ChatResponse:
        del options
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("scripted backend ran out of responses")
        return ChatResponse(content=self.responses.pop(0))


class InteractiveEnvTests(unittest.TestCase):
    def test_hypospace_causal_interactive_commit(self):
        env = make_env(
            EnvSpec(
                env_type="hypospace_causal",
                task={
                    "nodes": ["A", "B", "C"],
                    "max_edges": 2,
                    "target_edges": [["A", "B"], ["B", "C"]],
                    "query_budget": 2,
                },
                protocol="single",
                max_steps=5,
                max_commit=1,
                seed=1,
            )
        )
        self.assertIn("INTERVENE", env.reset())
        first = env.step("ACTION: INTERVENE A")
        self.assertTrue(first.parse_ok)
        self.assertIn("downstream_changed: B, C", first.observation)
        self.assertNotIn("Compatible graphs", first.observation)
        self.assertEqual(first.metrics["current_version_space_size"], 3)
        second = env.step("ACTION: INTERVENE B")
        self.assertTrue(second.parse_ok)
        final = env.step("ACTION: COMMIT graph(A->B, B->C)")
        self.assertTrue(final.done)
        self.assertIsNotNone(final.score)
        self.assertEqual(final.score.valid_unique_count, 1)
        self.assertGreaterEqual(final.score.reward, 1.0)
        self.assertGreater(final.score.breakdown.valid_hypothesis, 0.0)

    def test_local_episode_preempts_after_consecutive_invalid_actions(self):
        record = run_local_episode(
            spec=EnvSpec(
                env_type="hypospace_causal",
                task={
                    "nodes": ["A", "B", "C"],
                    "max_edges": 2,
                    "target_edges": [["A", "B"], ["B", "C"]],
                    "query_budget": 2,
                },
                protocol="single",
                max_steps=5,
                max_commit=1,
                max_consecutive_invalid=2,
                seed=11,
            ),
            backend=ScriptedBackend(["not an action", "still not an action"]),
            max_consecutive_invalid=2,
        )
        self.assertEqual(record["steps"], 2)
        self.assertEqual(
            record["diagnostics"]["early_stop_reason"],
            "consecutive_invalid_actions",
        )
        self.assertEqual(record["diagnostics"]["consecutive_invalid_at_stop"], 2)
        self.assertEqual(
            record["score"]["metrics"]["early_stop_reason"],
            "consecutive_invalid_actions",
        )
        self.assertEqual(record["score"]["parse_failures"], 2)
        self.assertEqual(record["score"]["invalid_actions"], 2)

    def test_hypospace_boolean_interactive_commit(self):
        env = make_env(
            EnvSpec(
                env_type="hypospace_boolean",
                task={
                    "variables": ["x", "y"],
                    "operators": ["AND", "OR", "NOT", "XOR", "NOR"],
                    "max_depth": 2,
                    "target_expression": "x AND y",
                    "query_budget": 2,
                },
                protocol="single",
                max_steps=5,
                max_commit=1,
                seed=2,
            )
        )
        self.assertIn("QUERY", env.reset())
        first = env.step("ACTION: QUERY x=1,y=1")
        self.assertIn("-> 1", first.observation)
        self.assertNotIn("Compatible expressions", first.observation)
        second = env.step("ACTION: QUERY x=1,y=0")
        self.assertIn("-> 0", second.observation)
        self.assertNotIn("Compatible expressions", second.observation)
        final = env.step("ACTION: COMMIT expr(x AND y)")
        self.assertTrue(final.done)
        self.assertIsNotNone(final.score)
        self.assertEqual(final.score.valid_unique_count, 1)
        self.assertEqual(final.score.reward, 1.0)
        self.assertGreater(final.score.breakdown.valid_hypothesis, 0.0)

    def test_hypospace_3d_interactive_commit(self):
        env = make_env(
            EnvSpec(
                env_type="hypospace_3d",
                task={
                    "grid_size": 2,
                    "max_height": 3,
                    "max_blocks": 3,
                    "target_heights": [[1, 0], [2, 1]],
                    "query_budget": 2,
                },
                protocol="single",
                max_steps=5,
                max_commit=1,
                seed=3,
            )
        )
        self.assertIn("VIEW", env.reset())
        first = env.step("ACTION: VIEW top")
        self.assertIn("[1 0; 1 1]", first.observation)
        self.assertNotIn("Compatible structures", first.observation)
        second = env.step("ACTION: VIEW front")
        self.assertIn("[2 1]", second.observation)
        self.assertNotIn("Compatible structures", second.observation)
        final = env.step("ACTION: COMMIT heights([1 0; 2 1])")
        self.assertTrue(final.done)
        self.assertIsNotNone(final.score)
        self.assertEqual(final.score.valid_unique_count, 1)
        self.assertEqual(final.score.reward, 1.0)
        self.assertGreater(final.score.breakdown.valid_hypothesis, 0.0)

    def test_prompt_hooks_differ_by_environment(self):
        scattered = make_env(
            EnvSpec(
                env_type="scattered_causal",
                task={
                    "world": {
                        "num_branches": 1,
                        "branch_depth": 2,
                        "distractors_per_node": 0,
                        "base_budget": 4,
                    },
                    "world_seed": 1,
                    "episode_seed": 2,
                    "dispersion": 1.0,
                },
                protocol="single",
                max_steps=4,
                max_commit=1,
                seed=1,
            )
        )
        self.assertIn("synthetic causal discovery", scattered.system_prompt())
        scattered.reset()
        scattered_step = scattered.step("ACTION: INTERVENE x00")
        self.assertIn(
            "Updated public state", scattered.observation_prompt(scattered_step)
        )

        hypospace = make_env(
            EnvSpec(
                env_type="hypospace_causal",
                task={
                    "nodes": ["A", "B", "C"],
                    "max_edges": 2,
                    "target_edges": [["A", "B"], ["B", "C"]],
                    "query_budget": 2,
                },
                protocol="single",
                max_steps=5,
                max_commit=1,
                seed=1,
            )
        )
        self.assertIn("interactive scientific discovery", hypospace.system_prompt())
        hypospace.reset()
        hypospace_step = hypospace.step("ACTION: INTERVENE A")
        self.assertIn(
            "Environment observation", hypospace.observation_prompt(hypospace_step)
        )
        self.assertNotIn(
            "Updated public state", hypospace.observation_prompt(hypospace_step)
        )

    def test_hypospace_version_space_size_is_debug_opt_in(self):
        hidden = make_env(
            EnvSpec(
                env_type="hypospace_causal",
                task={
                    "nodes": ["A", "B", "C"],
                    "max_edges": 2,
                    "target_edges": [["A", "B"], ["B", "C"]],
                    "query_budget": 2,
                },
                seed=1,
            )
        )
        hidden.reset()
        hidden_step = hidden.step("ACTION: INTERVENE A")
        self.assertNotIn("Compatible graphs remaining", hidden_step.observation)
        self.assertEqual(hidden.diagnostics()["current_version_space_size"], 3)

        visible = make_env(
            EnvSpec(
                env_type="hypospace_causal",
                task={
                    "nodes": ["A", "B", "C"],
                    "max_edges": 2,
                    "target_edges": [["A", "B"], ["B", "C"]],
                    "query_budget": 2,
                    "show_version_space_size": True,
                },
                seed=1,
            )
        )
        visible.reset()
        visible_step = visible.step("ACTION: INTERVENE A")
        self.assertIn("Compatible graphs remaining: 3", visible_step.observation)

    def test_hypospace_causal_local_rollout_with_scripted_backend(self):
        backend = ScriptedBackend(
            [
                "ACTION: INTERVENE A",
                "ACTION: INTERVENE B",
                "ACTION: COMMIT graph(A->B, B->C)",
            ]
        )
        record = run_local_episode(
            spec=EnvSpec(
                env_type="hypospace_causal",
                task={
                    "nodes": ["A", "B", "C"],
                    "max_edges": 2,
                    "target_edges": [["A", "B"], ["B", "C"]],
                    "query_budget": 2,
                },
                protocol="single",
                max_steps=5,
                max_commit=1,
                seed=1,
            ),
            backend=backend,
            output_transcript=True,
        )
        self.assertEqual(record["score"]["valid_unique_count"], 1)
        self.assertEqual(record["score"]["reward"], 1.0)
        self.assertEqual(record["diagnostics"]["budget_used"], 2)
        self.assertEqual(len(backend.calls), 3)
        self.assertIn("Environment observation", backend.calls[1][-1].content)
        self.assertNotIn("Updated public state", backend.calls[1][-1].content)

    def test_boolean_parser_truth_table(self):
        variables = ("x", "y")
        expr = parse_boolean_expr(
            "expr(NOT (x AND y))", variables, {"AND", "OR", "NOT"}
        )
        self.assertEqual(truth_table(expr, variables), (1, 1, 1, 0))

    def test_dataset_jsonl_rows_include_env_spec(self):
        specs = build_specs(
            env_type="hypospace_causal",
            count=2,
            seed=7,
            protocol="single",
            max_steps=4,
            max_commit=1,
            task_overrides={"nodes": ("A", "B", "C"), "max_edges": 2},
        )
        rows = specs_to_rows(specs, agent_name="discovery_agent_loop")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["agent_name"], "discovery_agent_loop")
        parsed = json.loads(rows[0]["env_spec_json"])
        self.assertEqual(parsed["env_type"], "hypospace_causal")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tasks.jsonl"
            write_jsonl(rows, path)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)

    def test_dataset_builder_supports_hypospace_3d(self):
        specs = build_specs(
            env_type="hypospace_3d",
            count=1,
            seed=11,
            protocol="single",
            max_steps=4,
            max_commit=1,
            task_overrides={"grid_size": 2, "max_height": 3, "max_blocks": 3},
        )
        parsed = json.loads(
            specs_to_rows(specs, agent_name="discovery_agent_loop")[0]["env_spec_json"]
        )
        self.assertEqual(parsed["env_type"], "hypospace_3d")
        self.assertEqual(parsed["task"]["grid_size"], 2)

    def test_yaml_dataset_config_supports_dispersion_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output = tmp_path / "mixed.jsonl"
            config = tmp_path / "datasets.yaml"
            config.write_text(
                f"""
defaults:
  agent_name: discovery_agent_loop
  protocol: single
  max_steps: 4
  max_commit: 1
datasets:
  - env_type: scattered_causal
    data_source: scattered_causal_mixed
    output: {output}
    count: 5
    seed: 23
    dispersion_values: [0.0, 0.5, 1.0]
    task:
      world:
        num_branches: 2
        branch_depth: 2
""",
                encoding="utf-8",
            )
            self.assertEqual(generate_from_config(config), [output])
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(
                [row["data_source"] for row in rows], ["scattered_causal_mixed"] * 5
            )
            dispersions = [
                json.loads(row["env_spec_json"])["task"]["dispersion"] for row in rows
            ]
            self.assertEqual(dispersions, [0.0, 0.5, 1.0, 0.0, 0.5])

    def test_yaml_dataset_config_supports_world_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output = tmp_path / "varied.jsonl"
            config = tmp_path / "datasets.yaml"
            config.write_text(
                f"""
defaults:
  agent_name: discovery_agent_loop
  protocol: single
  max_steps: 6
  max_commit: 1
datasets:
  - env_type: scattered_causal
    data_source: scattered_causal_varied
    output: {output}
    count: 12
    seed: 101
    dispersion_values: [0.0, 0.25, 0.5, 0.75, 1.0]
    world_values:
      num_branches: [3, 4, 5]
      branch_depth: [2, 3, 4]
      distractors_per_node: [1, 2]
      base_budget: [5, 7, 9]
    task:
      world:
        noise_sigma: 0.3
""",
                encoding="utf-8",
            )
            self.assertEqual(generate_from_config(config), [output])
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            specs = [json.loads(row["env_spec_json"]) for row in rows]
            dispersions = [spec["task"]["dispersion"] for spec in specs]
            self.assertEqual(
                dispersions,
                [0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25],
            )
            worlds = [spec["task"]["world"] for spec in specs]
            self.assertGreater(len({world["num_branches"] for world in worlds}), 1)
            self.assertGreater(len({world["branch_depth"] for world in worlds}), 1)
            self.assertEqual({world["noise_sigma"] for world in worlds}, {0.3})

    def test_yaml_dataset_config_rejects_overlapping_world_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output = tmp_path / "bad.jsonl"
            config = tmp_path / "datasets.yaml"
            config.write_text(
                f"""
datasets:
  - env_type: scattered_causal
    output: {output}
    count: 1
    seed: 1
    world_values:
      num_branches: [3, 4]
    task:
      world:
        num_branches: 4
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "world_values and task.world"):
                generate_from_config(config)

    def test_yaml_dataset_config_controls_scattered_causal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output = tmp_path / "scattered.jsonl"
            config = tmp_path / "datasets.yaml"
            config.write_text(
                f"""
defaults:
  agent_name: discovery_agent_loop
  protocol: set
  max_steps: 7
  max_commit: 3
datasets:
  - env_type: scattered_causal
    output: {output}
    count: 2
    seed: 17
    reward_profile: shaped
    reward:
      valid_hypothesis_reward: 1.5
      false_penalty: 0.7
      unsupported_penalty: 0.4
      budget_penalty: 0.01
    task:
      dispersion: 0.5
      budget: 9
      world:
        num_branches: 3
        branch_depth: 2
        distractors_per_node: 1
        true_mean: 1.2
        false_mean: -0.1
        noise_sigma: 0.2
        accept_threshold: 0.75
        reject_threshold: 0.2
        base_budget: 11
        test_cost: 2
        intervene_cost: 1
        invalid_action_cost: 3
      agent:
        include_hidden_debug_in_prompt: true
        max_evidence_items: 5
""",
                encoding="utf-8",
            )
            self.assertEqual(generate_from_config(config), [output])
            rows = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 2)
            row = json.loads(rows[0])
            parsed = json.loads(row["env_spec_json"])
            self.assertEqual(parsed["env_type"], "scattered_causal")
            self.assertEqual(parsed["protocol"], "set")
            self.assertEqual(parsed["max_steps"], 7)
            self.assertEqual(parsed["max_commit"], 3)
            self.assertEqual(parsed["task"]["dispersion"], 0.5)
            self.assertEqual(parsed["task"]["budget"], 9)
            self.assertEqual(parsed["task"]["world"]["num_branches"], 3)
            self.assertEqual(parsed["task"]["world"]["noise_sigma"], 0.2)
            self.assertEqual(parsed["task"]["reward_profile"], "shaped")
            self.assertEqual(parsed["task"]["reward"]["budget_penalty"], 0.01)
            self.assertTrue(parsed["task"]["agent"]["include_hidden_debug_in_prompt"])
            self.assertEqual(parsed["task"]["world_seed"], 17)
            self.assertEqual(parsed["task"]["episode_seed"], 17172)


if __name__ == "__main__":
    unittest.main()
