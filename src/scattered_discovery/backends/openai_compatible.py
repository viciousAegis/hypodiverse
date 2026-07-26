from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from scattered_discovery.backends.base import (
    ChatBackend,
    ChatMessage,
    ChatOptions,
    ChatResponse,
    normalize_chat_response,
)


@dataclass(frozen=True)
class OpenAICompatibleBackend(ChatBackend):
    """Chat backend for vLLM, SGLang, or any OpenAI-compatible server."""

    model: str
    base_url: str = "http://127.0.0.1:30000/v1"
    api_key: str | None = None
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    request_timeout_s: float = 180.0
    think: bool | str | None = None

    def chat(
        self, messages: list[ChatMessage], options: ChatOptions | None = None
    ) -> ChatResponse:
        max_tokens = (
            options.num_predict
            if options and options.num_predict is not None
            else self.max_tokens
        )
        think = options.think if options and options.think is not None else self.think
        temperature = (
            options.temperature
            if options and options.temperature is not None
            else self.temperature
        )
        top_p = options.top_p if options and options.top_p is not None else self.top_p
        payload = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if isinstance(think, bool):
            # vLLM and recent SGLang expose Qwen3 thinking control through
            # chat_template_kwargs. Servers that ignore unknown OpenAI-compatible
            # fields will simply keep their model default.
            payload["chat_template_kwargs"] = {"enable_thinking": think}
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.request_timeout_s
            ) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach OpenAI-compatible server at {self.base_url}."
            ) from exc

        choices = raw.get("choices") or []
        if not choices:
            raise RuntimeError(f"Unexpected chat completion response: {raw!r}")
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content", "")
        thinking = message.get("reasoning_content", message.get("thinking", ""))
        if thinking is None:
            thinking = ""
        if not isinstance(content, str) or not isinstance(thinking, str):
            raise RuntimeError(f"Unexpected chat completion message: {message!r}")
        usage = raw.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        return normalize_chat_response(
            content=content,
            thinking=thinking,
            finish_reason=choice.get("finish_reason"),
            completion_tokens=(
                int(completion_tokens)
                if isinstance(completion_tokens, int | float)
                else None
            ),
        )
