from __future__ import annotations

import json
import time
from typing import Any
from uuid import uuid4

from scattered_discovery.backends.base import split_visible_thinking
from scattered_discovery.envs.factory import make_env
from scattered_discovery.verl.qwen3_tokenization import observation_token_ids

try:  # pragma: no cover - exercised on the cluster with veRL installed.
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
        register,
    )
except ImportError:  # pragma: no cover - lets local unit tests import this module.
    AgentLoopBase = object  # type: ignore[assignment]
    AgentLoopMetrics = None  # type: ignore[assignment]
    AgentLoopOutput = None  # type: ignore[assignment]

    def register(_name: str):  # type: ignore[no-redef]
        def decorator(cls):
            return cls

        return decorator


def _as_text(value: Any) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _truncate_response_budget(
    response_ids: list[int],
    response_mask: list[int],
    *,
    max_response_length: int,
) -> tuple[list[int], list[int]]:
    if len(response_ids) <= max_response_length:
        return response_ids, response_mask
    return response_ids[:max_response_length], response_mask[:max_response_length]


@register("discovery_agent_loop")
class DiscoveryAgentLoop(AgentLoopBase):  # type: ignore[misc]
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> Any:
        if AgentLoopOutput is None or AgentLoopMetrics is None:
            raise RuntimeError("DiscoveryAgentLoop requires veRL to be installed.")

        env_spec_json = _as_text(kwargs["env_spec_json"])
        env_spec = json.loads(env_spec_json)
        env = make_env(env_spec)
        max_steps = int(env_spec.get("max_steps", 8))
        max_response_length = int(self.rollout_config.response_length)
        request_id = _as_text(kwargs.get("uid", uuid4().hex))

        messages = [
            {"role": "system", "content": env.system_prompt("verl")},
            {"role": "user", "content": env.reset()},
        ]
        prompt_ids = await self.apply_chat_template(messages)
        response_ids: list[int] = []
        response_mask: list[int] = []
        transcript: list[dict[str, Any]] = []
        num_preempted = 0
        started = time.monotonic()
        score = None

        for _ in range(max_steps):
            turn_prompt_ids = await self.apply_chat_template(messages)
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=turn_prompt_ids,
                sampling_params=sampling_params,
            )
            token_ids = list(output.token_ids)
            num_preempted = max(num_preempted, int(output.num_preempted or 0))
            response_ids.extend(token_ids)
            response_mask.extend([1] * len(token_ids))

            raw_assistant_text = self.tokenizer.decode(
                token_ids, skip_special_tokens=True
            )
            assistant_text, thinking_text = split_visible_thinking(raw_assistant_text)
            step = env.step(assistant_text)
            messages.append({"role": "assistant", "content": assistant_text})
            assistant_item = {
                "role": "assistant",
                "content": assistant_text,
                "action_text": step.action_text,
                "parse_ok": step.parse_ok,
            }
            if thinking_text:
                assistant_item["thinking"] = thinking_text
            transcript.append(assistant_item)
            if step.done:
                score = step.score
                break

            observation_message = env.observation_prompt(step, "verl")
            messages.append({"role": "user", "content": observation_message})
            obs_ids = observation_token_ids(self.tokenizer, observation_message)
            response_ids.extend(obs_ids)
            response_mask.extend([0] * len(obs_ids))
            transcript.append({"role": "user", "content": step.observation})

            response_ids, response_mask = _truncate_response_budget(
                response_ids,
                response_mask,
                max_response_length=max_response_length,
            )
            if len(response_ids) >= max_response_length:
                break

        if score is None:
            score = env.force_finalize()

        response_ids, response_mask = _truncate_response_budget(
            response_ids,
            response_mask,
            max_response_length=max_response_length,
        )
        if not response_ids:
            # veRL expects at least one response token for tensor assembly. EOS is a
            # neutral fallback when the model produced nothing before finalization.
            response_ids = [self.tokenizer.eos_token_id]
            response_mask = [0]

        elapsed = time.monotonic() - started
        score_dict = score.as_dict()
        diagnostics = env.diagnostics()
        metrics = {
            "terminal_reward": score.reward,
            "num_turns": len(transcript),
            "budget_used": diagnostics.get("budget_used", 0),
            "parse_failures": score.parse_failures,
            "invalid_actions": score.invalid_actions,
            "valid_unique_count": score.valid_unique_count,
            "valid_committed_count": score.valid_committed_count,
            "non_final_count": score.non_final_count,
            "validity": score.validity,
            "uniqueness": score.uniqueness,
            "final_version_space_size": score.metrics.get(
                "final_version_space_size", 0
            ),
            "current_version_space_size": score.metrics.get(
                "current_version_space_size", 0
            ),
            "recovery": score.metrics.get("recovery", 0.0),
        }
        for key, value in score.breakdown.as_dict().items():
            metrics[f"reward_{key}"] = value

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            reward_score=score.reward,
            num_turns=len(transcript),
            metrics=AgentLoopMetrics(
                generate_sequences=elapsed,
                tool_calls=0.0,
                compute_score=0.0,
                num_preempted=num_preempted,
            ),
            extra_fields={
                "reward_extra_info": metrics,
                "score": score_dict,
                "diagnostics": diagnostics,
                "transcript": transcript[:20],
            },
        )
