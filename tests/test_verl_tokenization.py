import unittest

from scattered_discovery.verl.agent_loop import (
    CausalMicroLabAgentLoop,
    DiscoveryAgentLoop,
    _add_dispersion_grouped_metrics,
    _apply_chat_template_no_thinking,
    _causal_micro_lab_length_cap_penalty,
)
from scattered_discovery.verl.qwen3_tokenization import fixed_base_message_token_ids


class FakeTokenizer:
    eos_token_id = 0
    last_enable_thinking = None

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt=False,
        tokenize=False,
        enable_thinking=None,
    ):
        self.last_enable_thinking = enable_thinking
        rendered = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


class VerlTokenizationTests(unittest.TestCase):
    def test_fixed_base_message_token_ids_returns_message_tokens(self):
        tokenizer = FakeTokenizer()
        token_ids = fixed_base_message_token_ids(
            tokenizer,
            {
                "role": "assistant",
                "content": "<think>short</think>\nACTION: QUERY x=1,y=1",
            },
        )
        decoded = "".join(chr(token_id) for token_id in token_ids)
        self.assertIn("ACTION: QUERY", decoded)

    def test_agent_loop_imports_without_verl_installed(self):
        self.assertIsNotNone(DiscoveryAgentLoop)
        self.assertIsNotNone(CausalMicroLabAgentLoop)

    def test_no_thinking_chat_template_sets_qwen_flag(self):
        tokenizer = FakeTokenizer()
        token_ids = _apply_chat_template_no_thinking(
            tokenizer,
            [{"role": "user", "content": "Return the answer."}],
        )
        self.assertEqual(tokenizer.last_enable_thinking, False)
        decoded = "".join(chr(token_id) for token_id in token_ids)
        self.assertIn("<|im_start|>assistant", decoded)

    def test_dispersion_grouped_metrics_use_count_and_sum(self):
        metrics = {
            "terminal_reward": 1.0,
            "valid_unique_count": 1.0,
            "validity": 1.0,
            "recovery": 0.25,
            "parse_failures": 0.0,
            "invalid_actions": 0.0,
            "unsupported_count": 0.0,
            "early_stop_consecutive_invalid": 0.0,
            "reward_valid_hypothesis": 1.0,
            "reward_clean_invalid_final": 0.0,
        }
        _add_dispersion_grouped_metrics(metrics, task={"dispersion": 0.25})

        self.assertEqual(metrics["task_dispersion"], 0.25)
        self.assertEqual(metrics["dispersion/0p25/count"], 1.0)
        self.assertEqual(metrics["dispersion/0p25/terminal_reward_sum"], 1.0)
        self.assertEqual(metrics["dispersion/0p25/recovery_sum"], 0.25)
        self.assertEqual(metrics["dispersion/0/count"], 0.0)
        self.assertEqual(metrics["dispersion/0/terminal_reward_sum"], 0.0)

    def test_causal_micro_lab_length_cap_penalty_only_hits_invalid_capped_outputs(self):
        self.assertEqual(
            _causal_micro_lab_length_cap_penalty(
                response_length=2048,
                max_response_length=2048,
                is_valid=False,
            ),
            0.0,
        )
        self.assertEqual(
            _causal_micro_lab_length_cap_penalty(
                response_length=2047,
                max_response_length=2048,
                is_valid=False,
            ),
            0.0,
        )
        self.assertEqual(
            _causal_micro_lab_length_cap_penalty(
                response_length=2048,
                max_response_length=2048,
                is_valid=True,
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
