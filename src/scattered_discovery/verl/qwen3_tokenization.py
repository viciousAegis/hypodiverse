from __future__ import annotations

from typing import Any


BASE_CHAT_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


def fixed_base_message_token_ids(tokenizer: Any, message: dict[str, str]) -> list[int]:
    """Tokenize one message with veRL's Qwen3-compatible fixed-base method.

    Qwen3 chat templates can drop previous reasoning content when rendering a full
    multi-turn history. Tokenizing each response-side message against a fixed base
    conversation avoids unstable deltas.
    """

    previous = tokenizer.apply_chat_template(
        BASE_CHAT_HISTORY,
        add_generation_prompt=True,
        tokenize=False,
    )
    current = tokenizer.apply_chat_template(
        [*BASE_CHAT_HISTORY, message],
        add_generation_prompt=False,
        tokenize=False,
    )
    if isinstance(previous, list) or isinstance(current, list):
        raise TypeError("fixed_base_message_token_ids expects string chat templates")
    if current.startswith(previous):
        delta = current[len(previous) :]
    else:
        # Some tokenizers vary whitespace around the assistant prompt. Fall back to
        # encoding the full standalone render rather than silently returning empty ids.
        delta = current
    return tokenizer.encode(delta, add_special_tokens=False)


def observation_token_ids(tokenizer: Any, content: str) -> list[int]:
    return fixed_base_message_token_ids(tokenizer, {"role": "user", "content": content})
