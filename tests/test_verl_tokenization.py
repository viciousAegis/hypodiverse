import unittest

from scattered_discovery.verl.agent_loop import DiscoveryAgentLoop
from scattered_discovery.verl.qwen3_tokenization import fixed_base_message_token_ids


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(
        self, messages, add_generation_prompt=False, tokenize=False
    ):
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


if __name__ == "__main__":
    unittest.main()
