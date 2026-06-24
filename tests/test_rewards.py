import json
import subprocess
import sys
import unittest

from scattered_discovery.envs.base import EnvSpec
from scattered_discovery.envs.factory import make_env
from scattered_discovery.rewards import (
    REWARD_PROFILES,
    REWARD_DEFAULTS,
    reward_config_for_env,
    reward_config_from_task,
)


class RewardConfigTests(unittest.TestCase):
    def test_defaults_are_per_environment(self):
        self.assertEqual(
            set(REWARD_DEFAULTS),
            {
                "hypospace_causal",
                "hypospace_boolean",
                "hypospace_3d",
                "scattered_causal",
            },
        )
        self.assertNotIn(
            "duplicate_penalty", reward_config_for_env("hypospace_causal").as_dict()
        )
        self.assertEqual(reward_config_for_env("hypospace_causal").false_penalty, 0.0)
        self.assertEqual(reward_config_for_env("hypospace_causal").format_reward, 0.0)
        self.assertEqual(REWARD_PROFILES["shaped"].false_penalty, 0.5)
        self.assertEqual(REWARD_PROFILES["shaped"].format_reward, 0.03)
        self.assertEqual(
            REWARD_PROFILES["terminal_clean_invalid_bonus"].clean_invalid_final_reward,
            0.2,
        )

    def test_nested_task_reward_overrides_defaults(self):
        config = reward_config_from_task(
            "hypospace_causal",
            {"reward": {"valid_hypothesis_reward": 2.5, "false_penalty": 0.75}},
        )
        self.assertEqual(config.valid_hypothesis_reward, 2.5)
        self.assertEqual(config.false_penalty, 0.75)
        self.assertEqual(config.unsupported_penalty, 0.0)

    def test_shaped_profile_can_be_selected(self):
        config = reward_config_from_task(
            "hypospace_boolean",
            {"reward_profile": "shaped"},
        )
        self.assertEqual(config.valid_hypothesis_reward, 1.0)
        self.assertEqual(config.false_penalty, 0.5)
        self.assertEqual(config.non_final_penalty, 0.25)
        self.assertEqual(config.unsupported_penalty, 0.25)
        self.assertEqual(config.format_reward, 0.03)

    def test_clean_invalid_bonus_profile_can_be_selected(self):
        config = reward_config_from_task(
            "scattered_causal",
            {"reward_profile": "terminal_clean_invalid_bonus"},
        )
        self.assertEqual(config.valid_hypothesis_reward, 1.0)
        self.assertEqual(config.clean_invalid_final_reward, 0.2)
        self.assertEqual(config.false_penalty, 0.0)
        self.assertEqual(config.format_reward, 0.0)

    def test_factory_passes_reward_config_to_env(self):
        env = make_env(
            EnvSpec(
                env_type="hypospace_causal",
                task={
                    "nodes": ["A", "B"],
                    "max_edges": 1,
                    "target_edges": [["A", "B"]],
                    "query_budget": 0,
                    "reward": {
                        "valid_hypothesis_reward": 2.0,
                        "format_reward": 0.0,
                        "admissible_reward": 0.0,
                        "commit_format_reward": 0.0,
                    },
                },
                protocol="single",
                max_steps=1,
                max_commit=1,
                seed=1,
            )
        )
        env.reset()
        final = env.step("ACTION: COMMIT graph(A->B)")
        self.assertIsNotNone(final.score)
        self.assertEqual(final.score.breakdown.valid_hypothesis, 2.0)
        self.assertEqual(final.score.reward, 2.0)

    def test_duplicate_set_zeroes_reward_without_penalty(self):
        env = make_env(
            EnvSpec(
                env_type="hypospace_causal",
                task={
                    "nodes": ["A", "B"],
                    "max_edges": 1,
                    "target_edges": [["A", "B"]],
                    "query_budget": 0,
                    "reward": {
                        "format_reward": 0.0,
                        "admissible_reward": 0.0,
                        "commit_format_reward": 0.0,
                    },
                },
                protocol="set",
                max_steps=1,
                max_commit=2,
                seed=1,
            )
        )
        env.reset()
        final = env.step("ACTION: COMMIT [graph(A->B); graph(A->B)]")
        self.assertIsNotNone(final.score)
        self.assertEqual(final.score.duplicate_count, 1)
        self.assertEqual(final.score.valid_committed_count, 2)
        self.assertEqual(final.score.valid_unique_count, 1)
        self.assertEqual(final.score.validity, 1.0)
        self.assertEqual(final.score.uniqueness, 0.5)
        self.assertEqual(final.score.reward, 0.0)

    def test_rewards_cli_prints_json(self):
        output = subprocess.check_output(
            [
                sys.executable,
                "-m",
                "scattered_discovery.rewards",
                "--env-type",
                "hypospace_boolean",
            ],
            text=True,
        )
        payload = json.loads(output)
        self.assertEqual(payload["hypospace_boolean"]["valid_hypothesis_reward"], 1.0)
        self.assertNotIn("duplicate_penalty", payload["hypospace_boolean"])


if __name__ == "__main__":
    unittest.main()
