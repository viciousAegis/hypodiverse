from __future__ import annotations

from typing import Literal


LOCAL_SYSTEM_PROMPT = """You are an interactive scientific discovery agent.
Return exactly one ACTION line per turn. Use observations to decide which query to run next
and which final hypothesis or hypothesis set to commit."""


VERL_SYSTEM_PROMPT = """You are an interactive scientific discovery agent.
Return exactly one ACTION line per turn. Use observations to choose useful queries and
commit final hypotheses only when the evidence supports them."""


def system_prompt_for_runtime(runtime: Literal["local", "verl"] = "local") -> str:
    if runtime == "verl":
        return VERL_SYSTEM_PROMPT
    return LOCAL_SYSTEM_PROMPT


def next_action_observation_prompt(observation: str) -> str:
    return f"Environment observation:\n{observation}\n\nReturn the next ACTION line."
