import unittest

from scattered_discovery.algos import get_algorithm, list_algorithms


class AlgorithmConfigTests(unittest.TestCase):
    def test_grpo_overrides_are_verl_native(self):
        algorithm = get_algorithm("grpo")
        self.assertFalse(algorithm.requires_custom_trainer)
        self.assertEqual(
            algorithm.verl_overrides(),
            ("algorithm.adv_estimator=grpo", "algorithm.use_kl_in_reward=False"),
        )

    def test_set_reward_grpo_is_environment_protocol(self):
        algorithm = get_algorithm("set_reward_grpo")
        self.assertFalse(algorithm.requires_custom_trainer)
        self.assertIn("protocol=set", " ".join(algorithm.notes()))

    def test_echo_recipe_is_explicitly_experimental(self):
        algorithm = get_algorithm("echo_grpo")
        self.assertTrue(algorithm.requires_custom_trainer)
        self.assertIn("+discovery_algorithm.name=echo_grpo", algorithm.verl_overrides())

    def test_registry_lists_algorithms(self):
        names = {algorithm.name for algorithm in list_algorithms()}
        self.assertEqual(names, {"echo_grpo", "grpo", "set_reward_grpo"})


if __name__ == "__main__":
    unittest.main()
