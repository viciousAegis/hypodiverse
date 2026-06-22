import unittest

from scattered_discovery.config import WorldConfig
from scattered_discovery.envs.scattered_causal import ScatteredDiscoveryEnv
from scattered_discovery.envs.scattered_dsl import (
    CommitAction,
    Edge,
    PathExpr,
    canonical_key,
    parse_action_line,
)


class DSLAndEnvTests(unittest.TestCase):
    def test_parse_commit_set(self) -> None:
        action = parse_action_line(
            "ACTION: COMMIT [path(x00,x01,x02); path(x10,x11,x12)]"
        )
        self.assertIsInstance(action, CommitAction)
        self.assertEqual(action.mode, "set")
        self.assertEqual(len(action.exprs), 2)

    def test_parse_legacy_commit_set_alias(self) -> None:
        action = parse_action_line(
            "ACTION: COMMIT_SET [path(x00,x01,x02); path(x10,x11,x12)]"
        )
        self.assertIsInstance(action, CommitAction)
        self.assertEqual(action.mode, "set")
        self.assertEqual(len(action.exprs), 2)

    def test_world_commit_requires_terminal_and_evidence(self) -> None:
        config = WorldConfig(
            num_branches=1,
            branch_depth=2,
            distractors_per_node=0,
            noise_sigma=0.1,
            base_budget=5,
        )
        env = ScatteredDiscoveryEnv(
            config,
            world_seed=1,
            episode_seed=2,
            dispersion=1.0,
            protocol="single",
            max_commit=1,
        )
        branch = env.world.branches[0]
        terminal = PathExpr(branch.path)
        result = env.step(f"ACTION: COMMIT path({','.join(branch.path)})")
        self.assertTrue(result.done)
        self.assertIsNotNone(result.score)
        assert result.score is not None
        self.assertEqual(result.score.valid_unique_count, 0)
        self.assertEqual(result.score.unsupported_count, 1)
        self.assertEqual(result.score.non_final_count, 0)

        env = ScatteredDiscoveryEnv(
            config,
            world_seed=1,
            episode_seed=2,
            dispersion=1.0,
            protocol="single",
            max_commit=1,
        )
        intermediate = branch.path[:2]
        result = env.step(f"ACTION: COMMIT path({','.join(intermediate)})")
        self.assertTrue(result.done)
        self.assertIsNotNone(result.score)
        assert result.score is not None
        self.assertEqual(result.score.valid_unique_count, 0)
        self.assertEqual(result.score.non_final_count, 1)
        self.assertEqual(result.score.unsupported_count, 0)

        env = ScatteredDiscoveryEnv(
            config,
            world_seed=1,
            episode_seed=2,
            dispersion=1.0,
            protocol="single",
            max_commit=1,
        )
        for src, dst in zip(branch.path[:-1], branch.path[1:], strict=True):
            env.known_variables.add(dst)
            key = canonical_key(Edge(src, dst))
            env.evidence.update(key, 1.0)
        result = env.step(f"ACTION: COMMIT path({','.join(branch.path)})")
        self.assertTrue(result.done)
        self.assertIsNotNone(result.score)
        assert result.score is not None
        self.assertEqual(result.score.valid_unique_count, 1)
        self.assertIn(canonical_key(terminal), result.score.valid_keys)

    def test_path_level_evidence_does_not_support_final_commit(self) -> None:
        config = WorldConfig(
            num_branches=1,
            branch_depth=2,
            distractors_per_node=0,
            noise_sigma=0.1,
            base_budget=5,
        )
        env = ScatteredDiscoveryEnv(
            config,
            world_seed=1,
            episode_seed=2,
            dispersion=1.0,
            protocol="single",
            max_commit=1,
        )
        branch = env.world.branches[0]
        path_text = ",".join(branch.path)
        for variable in branch.path:
            env.known_variables.add(variable)
        env.step(f"ACTION: TEST path({path_text})")

        result = env.step(f"ACTION: COMMIT path({path_text})")
        self.assertTrue(result.done)
        self.assertIsNotNone(result.score)
        assert result.score is not None
        self.assertEqual(result.score.valid_unique_count, 0)
        self.assertEqual(result.score.unsupported_count, 1)

    def test_public_state_hides_verifier_status_by_default(self) -> None:
        config = WorldConfig(
            num_branches=1,
            branch_depth=2,
            distractors_per_node=0,
            noise_sigma=0.1,
            base_budget=5,
        )
        env = ScatteredDiscoveryEnv(
            config,
            world_seed=1,
            episode_seed=2,
            dispersion=1.0,
            protocol="single",
            max_commit=1,
        )

        initial_state = env.public_state_text()
        self.assertNotIn("Accepted claims", initial_state)
        self.assertNotIn("Rejected claims", initial_state)

        root = sorted(env.world.initial_variables)[0]
        observation = env.step(f"ACTION: INTERVENE {root}").observation
        visible_state = env.public_state_text()
        visible_text = f"{observation}\n{visible_state}".lower()

        self.assertIn("measured_effect", observation)
        self.assertIn("mean_measurement", visible_state)
        for leaked_token in (
            "accepted claims",
            "rejected claims",
            "posterior",
            "status=",
            "supporting",
            "against",
            "ambiguous",
        ):
            self.assertNotIn(leaked_token, visible_text)

        debug_state = env.public_state_text(include_evidence_status=True)
        self.assertIn("Accepted claims", debug_state)
        self.assertIn("Rejected claims", debug_state)
        self.assertIn("posterior=", debug_state)
        self.assertIn("status=", debug_state)


if __name__ == "__main__":
    unittest.main()
