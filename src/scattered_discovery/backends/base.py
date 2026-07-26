from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol

_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class ChatResponse:
    content: str
    thinking: str = ""
    finish_reason: str | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class ChatOptions:
    think: bool | str | None = None
    num_predict: int | None = None
    temperature: float | None = None
    top_p: float | None = None


def split_visible_thinking(content: str) -> tuple[str, str]:
    """Remove Qwen-style visible thinking blocks from assistant content."""

    thinking_parts: list[str] = []

    def replace_closed(match: re.Match[str]) -> str:
        thinking = match.group(1).strip()
        if thinking:
            thinking_parts.append(thinking)
        return ""

    stripped = _THINK_BLOCK_RE.sub(replace_closed, content)
    lower = stripped.lower()
    open_idx = lower.find("<think>")
    if open_idx >= 0:
        thinking = stripped[open_idx + len("<think>") :].strip()
        if thinking:
            thinking_parts.append(thinking)
        stripped = stripped[:open_idx]

    lower = stripped.lower()
    close_idx = lower.rfind("</think>")
    if close_idx >= 0:
        thinking = stripped[:close_idx].strip()
        if thinking:
            thinking_parts.append(thinking)
        stripped = stripped[close_idx + len("</think>") :]

    return stripped.strip(), "\n\n".join(thinking_parts)


def normalize_chat_response(
    content: str,
    thinking: str = "",
    *,
    finish_reason: str | None = None,
    completion_tokens: int | None = None,
) -> ChatResponse:
    final_content, visible_thinking = split_visible_thinking(content)
    parts = [part for part in (thinking.strip(), visible_thinking.strip()) if part]
    return ChatResponse(
        content=final_content,
        thinking="\n\n".join(parts),
        finish_reason=finish_reason,
        completion_tokens=completion_tokens,
    )


class ChatBackend(Protocol):
    def chat(
        self, messages: list[ChatMessage], options: ChatOptions | None = None
    ) -> ChatResponse: ...
