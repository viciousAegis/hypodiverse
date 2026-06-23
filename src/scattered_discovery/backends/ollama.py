from __future__ import annotations

import json
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
class OllamaBackend(ChatBackend):
    model: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    top_p: float = 0.9
    num_predict: int = 320
    request_timeout_s: float = 180.0
    think: bool | str | None = "low"

    def chat(
        self, messages: list[ChatMessage], options: ChatOptions | None = None
    ) -> ChatResponse:
        num_predict = (
            options.num_predict
            if options and options.num_predict is not None
            else self.num_predict
        )
        think = options.think if options and options.think is not None else self.think
        payload = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": num_predict,
            },
        }
        if think is not None:
            payload["think"] = think
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.request_timeout_s
            ) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. Is `ollama serve` running?"
            ) from exc
        message = raw.get("message", {})
        content = message.get("content", "")
        thinking = message.get("thinking", "")
        if not isinstance(content, str) or not isinstance(thinking, str):
            raise RuntimeError(f"Unexpected Ollama response: {raw!r}")
        return normalize_chat_response(content=content, thinking=thinking)
