from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - CPU-only minimal installs.
    torch = None

from scattered_discovery.verl.agent_loop import (
    _behavior_hash_parts,
    latent_id_for_rollout,
    negative_latent_id,
)
from scattered_discovery.verl.lifpo_trainer import (
    LIFPO_REWARD_EXTRA_DEFAULTS,
    compute_inverse_frequency_advantages,
    compute_lifpo_advantages,
    finish_trainer_wandb,
    merge_lifpo_output_extra_fields,
    normalize_lifpo_reward_extra_info,
    sanitize_validation_reward_extras,
    select_lifpo_metadata,
)


class LIFPOTrainerTests(unittest.TestCase):
    def test_reward_metadata_has_one_fixed_numeric_schema(self):
        complete = normalize_lifpo_reward_extra_info(
            {
                "validity": 1.0,
                "reward_valid_hypothesis": 1.0,
                "behavior_hash_hi": 12,
                "behavior_hash_lo": 34,
            },
            reward_score=1.0,
        )
        failure = normalize_lifpo_reward_extra_info({}, reward_score=0.0)

        self.assertEqual(set(complete), set(LIFPO_REWARD_EXTRA_DEFAULTS))
        self.assertEqual(set(failure), set(LIFPO_REWARD_EXTRA_DEFAULTS))
        self.assertEqual(failure["behavior_hash_hi"], -1.0)
        self.assertEqual(complete["terminal_reward"], 1.0)

    def test_wandb_is_finished_once(self):
        calls: list[int] = []

        class Backend:
            def finish(self, *, exit_code: int) -> None:
                calls.append(exit_code)

        class Tracking:
            def __init__(self) -> None:
                self.logger = {"wandb": Backend(), "console": object()}

        class Trainer:
            def __init__(self) -> None:
                self.logger = Tracking()

        trainer = Trainer()
        finish_trainer_wandb(trainer, exit_code=0)
        finish_trainer_wandb(trainer, exit_code=1)

        self.assertEqual(calls, [0])
        self.assertNotIn("wandb", trainer.logger.logger)

    def test_latent_schedule_covers_each_identity_once(self):
        latent_ids = [latent_id_for_rollout(index, 8) for index in range(8)]
        self.assertEqual(latent_ids, list(range(1, 9)))
        self.assertEqual(
            [negative_latent_id(value, 8) for value in latent_ids],
            [2, 3, 4, 5, 6, 7, 8, 1],
        )

    def test_behavior_hash_is_stable_and_distinguishes_outcomes(self):
        self.assertEqual(_behavior_hash_parts("a"), _behavior_hash_parts("a"))
        self.assertNotEqual(_behavior_hash_parts("a"), _behavior_hash_parts("b"))
        self.assertEqual(_behavior_hash_parts(None), (-1, -1))

    def test_metadata_selection_skips_padding_and_accepts_flattened_fields(self):
        extra_fields = [
            {
                "validity": 1.0,
                "reward_valid_hypothesis": 1.0,
                "behavior_hash_hi": 12.0,
                "behavior_hash_lo": 34.0,
                "latent_enabled": 1.0,
                "latent_id": 3.0,
                "latent_negative_id": 4.0,
                "latent_answer_token_count": 9.0,
            },
            {},
        ]

        real_indices, metadata = select_lifpo_metadata(
            extra_fields,
            [{"is_padding": False}, {"is_padding": True}],
        )

        self.assertEqual(real_indices, [0])
        self.assertEqual(metadata[0]["behavior"], (12, 34))
        self.assertEqual(metadata[0]["latent_id"], 3)
        self.assertEqual(metadata[0]["answer_token_count"], 9)

    def test_generated_fields_survive_input_collision(self):
        generated = {
            "reward_extra_info": {
                "validity": 1.0,
                "reward_valid_hypothesis": 1.0,
                "behavior_hash_hi": 12.0,
                "behavior_hash_lo": 34.0,
            },
            "latent_negative_prompt_ids": [101, 102],
        }
        merged = merge_lifpo_output_extra_fields(
            generated,
            {"dataset_private_field": "preserved", "reward_extra_info": {}},
        )

        self.assertEqual(merged["dataset_private_field"], "preserved")
        self.assertEqual(merged["validity"], 1.0)
        self.assertEqual(merged["behavior_hash_hi"], 12.0)
        self.assertEqual(merged["latent_negative_prompt_ids"], [101, 102])

    def test_validation_sanitizer_removes_behavior_identifiers(self):
        sanitized, missing = sanitize_validation_reward_extras(
            {
                "reward": [1.0, 0.0],
                "validity": [1.0, 0.0],
                "behavior_hash_hi": [12.0, -1.0],
                "behavior_hash_lo": [34.0, -1.0],
            }
        )
        self.assertEqual(set(sanitized), {"reward", "validity"})
        self.assertEqual(missing, {})

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_frequency_credit_rewards_a_rare_valid_outcome(self):
        rewards = torch.tensor([[1.0], [1.0], [1.0], [0.2]])
        metadata = [
            {"valid": True, "valid_reward": 1.0, "behavior": (1, 1)},
            {"valid": True, "valid_reward": 1.0, "behavior": (1, 1)},
            {"valid": True, "valid_reward": 1.0, "behavior": (2, 2)},
            {"valid": False, "valid_reward": 0.0, "behavior": None},
        ]
        advantages, returns, metrics = compute_inverse_frequency_advantages(
            token_level_rewards=rewards,
            response_mask=torch.ones_like(rewards),
            index=["state"] * 4,
            metadata=metadata,
            epsilon=0.2,
        )

        self.assertTrue(torch.equal(advantages, returns))
        self.assertGreater(advantages[2, 0], advantages[0, 0])
        self.assertGreater(advantages[0, 0], advantages[3, 0])
        self.assertAlmostEqual(metrics["lifpo/weight_max"], 4.0)
        self.assertAlmostEqual(metrics["lifpo/duplicate_valid_rate"], 1 / 3)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_counterfactual_credit_is_validity_gated_and_bounded(self):
        rewards = torch.tensor([[1.0], [1.0], [1.0], [0.2]])
        metadata = [
            {
                "valid": valid,
                "valid_reward": 1.0 if valid else 0.0,
                "behavior": behavior,
                "latent_enabled": True,
                "latent_id": index + 1,
                "latent_negative_id": (index + 1) % 4 + 1,
                "answer_token_count": 1,
            }
            for index, (valid, behavior) in enumerate(
                [(True, (1, 1)), (True, (1, 1)), (True, (2, 2)), (False, None)]
            )
        ]
        advantages, returns, metrics = compute_lifpo_advantages(
            token_level_rewards=rewards,
            response_mask=torch.ones_like(rewards),
            assigned_log_probs=torch.tensor([[-1.0], [-1.0], [-0.1], [-0.1]]),
            negative_log_probs=torch.tensor([[-1.0], [-1.0], [-2.0], [-9.0]]),
            index=["state"] * 4,
            metadata=metadata,
            epsilon=0.2,
            counterfactual_alpha=0.5,
            counterfactual_clip=1.0,
            inverse_frequency_enabled=True,
            latent_count=4,
            frequency_credit_mode="bonus",
            frequency_credit_max=0.5,
            counterfactual_token_scope="full_response",
            counterfactual_reduction="sum",
            counterfactual_valid_only=True,
        )

        self.assertTrue(torch.equal(advantages, returns))
        self.assertGreater(advantages[2, 0], advantages[0, 0])
        self.assertEqual(metrics["lifpo/counterfactual_reward_bound"], 0.5)
        self.assertLessEqual(metrics["lifpo/counterfactual_reward_max_abs"], 0.5)
        self.assertEqual(metrics["lifpo/counterfactual_valid_only"], 1.0)
        self.assertEqual(metrics["lifpo/groups_all_latents_present_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
