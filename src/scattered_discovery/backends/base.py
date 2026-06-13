from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class ChatResponse:
    content: str
    thinking: str = ""


@dataclass(frozen=True)
class ChatOptions:
    think: bool | str | None = None
    num_predict: int | None = None


class ChatBackend(Protocol):
    def chat(
        self, messages: list[ChatMessage], options: ChatOptions | None = None
    ) -> ChatResponse: ...
