from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scattered_discovery.backends.base import (
    ChatBackend,
    ChatMessage,
    ChatOptions,
    ChatResponse,
    normalize_chat_response,
)


def _apply_chat_template(
    tokenizer: Any,
    messages: list[ChatMessage],
    *,
    add_generation_prompt: bool,
    think: bool | str | None,
) -> str:
    payload = [
        {"role": message.role, "content": message.content}
        for message in messages
    ]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if isinstance(think, bool):
        kwargs["enable_thinking"] = think
    try:
        rendered = tokenizer.apply_chat_template(payload, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        rendered = tokenizer.apply_chat_template(payload, **kwargs)
    if not isinstance(rendered, str):
        raise TypeError("apply_chat_template(..., tokenize=False) returned non-string")
    return rendered


@dataclass
class HuggingFaceBackend(ChatBackend):
    """Single-process local Transformers chat backend.

    This is intended for SFT smoke/eval checks where avoiding SGLang startup and
    serving-version issues is more useful than maximum throughput.
    """

    model: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    think: bool | str | None = None

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_obj = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self.model_obj.eval()

    def chat(
        self, messages: list[ChatMessage], options: ChatOptions | None = None
    ) -> ChatResponse:
        max_tokens = (
            options.num_predict
            if options and options.num_predict is not None
            else self.max_tokens
        )
        think = options.think if options and options.think is not None else self.think
        prompt = _apply_chat_template(
            self.tokenizer,
            messages,
            add_generation_prompt=True,
            think=think,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {
            key: value.to(self.model_obj.device)
            for key, value in inputs.items()
        }
        do_sample = self.temperature > 0
        with self._torch.inference_mode():
            generated = self.model_obj.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = int(inputs["input_ids"].shape[-1])
        new_tokens = generated[0][prompt_len:]
        content = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return normalize_chat_response(content)
