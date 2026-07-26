from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any
from uuid import uuid4

from scattered_discovery.backends.base import split_visible_thinking
from scattered_discovery.envs.causal_micro_lab.consequence_reward import (
    base_candidate_reward,
    evaluate_consequences,
)
from scattered_discovery.envs.factory import make_env
from scattered_discovery.verl.qwen3_tokenization import observation_token_ids

DISPERSION_BUCKETS = (0.0, 0.25, 0.5, 0.75, 1.0)
DISPERSION_GROUPED_METRICS = (
    "terminal_reward",
    "valid_unique_count",
    "validity",
    "recovery",
    "parse_failures",
    "invalid_actions",
    "unsupported_count",
    "early_stop_consecutive_invalid",
    "reward_valid_hypothesis",
    "reward_clean_invalid_final",
)

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


def _behavior_hash_parts(behavior_key: str | None) -> tuple[int, int]:
    """Encode an outcome identity as two exactly representable numeric fields."""
    if not behavior_key:
        return -1, -1
    digest = hashlib.sha256(behavior_key.encode("utf-8")).digest()
    return (
        int.from_bytes(digest[:4], byteorder="big", signed=False),
        int.from_bytes(digest[4:8], byteorder="big", signed=False),
    )


def _truncate_response_budget(
    response_ids: list[int],
    response_mask: list[int],
    *,
    max_response_length: int,
) -> tuple[list[int], list[int]]:
    if len(response_ids) <= max_response_length:
        return response_ids, response_mask
    return response_ids[:max_response_length], response_mask[:max_response_length]


def _generation_log_probs(
    output: Any,
    *,
    expected_length: int,
    required: bool,
) -> list[float] | None:
    values = getattr(output, "log_probs", None)
    if values is None:
        if required:
            raise RuntimeError(
                "The rollout requested log probabilities, but the generation "
                "server returned none."
            )
        return None
    if len(values) != expected_length:
        raise RuntimeError(
            "Generation token/log-prob length mismatch: "
            f"{expected_length} tokens versus {len(values)} log probabilities."
        )
    return [float(value) for value in values]


def _apply_chat_template_no_thinking(
    tokenizer: Any, messages: list[dict[str, str]]
) -> list[int]:
    kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": True,
        "enable_thinking": False,
    }
    try:
        token_ids = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        token_ids = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return list(token_ids)


def _causal_micro_lab_length_cap_penalty(
    *,
    response_length: int,
    max_response_length: int,
    soft_start: int | None = None,
    max_penalty: float = -0.2,
) -> float:
    if max_response_length <= 0:
        return 0.0
    start = soft_start if soft_start is not None else int(max_response_length * 0.75)
    start = max(0, min(start, max_response_length))
    if response_length <= start:
        return 0.0
    if response_length >= max_response_length:
        return max_penalty
    width = max(1, max_response_length - start)
    fraction = (response_length - start) / width
    return max_penalty * fraction


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _config_int(config: dict[str, Any], key: str, env_name: str, default: int) -> int:
    if key in config and config[key] is not None:
        return int(config[key])
    return _env_int(env_name, default)


def _config_float(
    config: dict[str, Any], key: str, env_name: str, default: float
) -> float:
    if key in config and config[key] is not None:
        return float(config[key])
    return _env_float(env_name, default)


def _config_bool(
    config: dict[str, Any], key: str, env_name: str, default: bool
) -> bool:
    if key in config and config[key] is not None:
        return bool(config[key])
    raw = os.environ.get(env_name)
    if raw is None or raw == "":
        return default
    return raw == "1"


def _dispersion_label(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _task_dispersion(task: dict[str, Any]) -> float | None:
    raw = task.get("dispersion")
    if raw is None:
        return None
    return float(raw)


def _add_dispersion_grouped_metrics(
    metrics: dict[str, float],
    *,
    task: dict[str, Any],
) -> None:
    dispersion = _task_dispersion(task)
    if dispersion is not None:
        metrics["task_dispersion"] = dispersion

    for bucket in DISPERSION_BUCKETS:
        label = _dispersion_label(bucket)
        active = dispersion is not None and abs(dispersion - bucket) < 1e-9
        metrics[f"dispersion/{label}/count"] = 1.0 if active else 0.0
        for key in DISPERSION_GROUPED_METRICS:
            metrics[f"dispersion/{label}/{key}_sum"] = (
                float(metrics.get(key, 0.0)) if active else 0.0
            )


@register("discovery_agent_loop")
class DiscoveryAgentLoop(AgentLoopBase):  # type: ignore[misc]
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> Any:
        if AgentLoopOutput is None or AgentLoopMetrics is None:
            raise RuntimeError("DiscoveryAgentLoop requires veRL to be installed.")

        env_spec_json = _as_text(kwargs["env_spec_json"])
        env_spec = json.loads(env_spec_json)
        env = make_env(env_spec)
        max_steps = int(env_spec.get("max_steps", 8))
        max_consecutive_invalid = int(env_spec.get("max_consecutive_invalid", 2))
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
        consecutive_invalid = 0
        early_stop_reason: str | None = None

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
            invalid_step = (not step.parse_ok) or (
                "not admissible" in step.observation.lower()
            )
            consecutive_invalid = consecutive_invalid + 1 if invalid_step else 0
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
            if (
                max_consecutive_invalid > 0
                and consecutive_invalid >= max_consecutive_invalid
            ):
                early_stop_reason = "consecutive_invalid_actions"
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
        score.metrics["early_stop_reason"] = early_stop_reason
        score.metrics["max_consecutive_invalid"] = max_consecutive_invalid
        score.metrics["consecutive_invalid_at_stop"] = consecutive_invalid

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
            "early_stop_consecutive_invalid": 1.0
            if early_stop_reason == "consecutive_invalid_actions"
            else 0.0,
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
        _add_dispersion_grouped_metrics(metrics, task=env_spec.get("task", {}))

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
                "min_global_steps": 0,
                "max_global_steps": 0,
                "reward_extra_info": metrics,
                "score": score_dict,
                "diagnostics": diagnostics,
                "transcript": transcript[:20],
            },
        )


@register("causal_micro_lab_agent_loop")
class CausalMicroLabAgentLoop(AgentLoopBase):  # type: ignore[misc]
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> Any:
        if AgentLoopOutput is None or AgentLoopMetrics is None:
            raise RuntimeError("CausalMicroLabAgentLoop requires veRL to be installed.")

        env_spec_json = _as_text(kwargs["env_spec_json"])
        env_spec = json.loads(env_spec_json)
        agent_config = env_spec.get("agent") or {}
        env = make_env(env_spec)
        max_response_length = int(self.rollout_config.response_length)
        request_id = _as_text(kwargs.get("uid", uuid4().hex))

        messages = [
            {"role": "system", "content": env.system_prompt("verl")},
            {"role": "user", "content": env.reset()},
        ]
        if os.environ.get("CAUSAL_MICRO_LAB_DISABLE_THINKING", "0") == "1":
            prompt_ids = _apply_chat_template_no_thinking(self.tokenizer, messages)
        else:
            prompt_ids = await self.apply_chat_template(messages)
        started = time.monotonic()
        output = await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )
        elapsed = time.monotonic() - started
        response_ids = list(output.token_ids)
        raw_response_length = len(response_ids)
        response_mask = [1] * len(response_ids)
        raw_assistant_text = self.tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        )
        assistant_text, thinking_text = split_visible_thinking(raw_assistant_text)
        step = env.step(assistant_text)
        score = step.score if step.score is not None else env.force_finalize()
        length_penalty_start = _config_int(
            agent_config,
            "length_penalty_start",
            "CAUSAL_MICRO_LAB_LENGTH_PENALTY_START",
            int(max_response_length * 0.75),
        )
        length_penalty_max = _config_float(
            agent_config,
            "length_penalty_max",
            "CAUSAL_MICRO_LAB_LENGTH_PENALTY_MAX",
            -0.2,
        )
        length_cap_penalty = _causal_micro_lab_length_cap_penalty(
            response_length=raw_response_length,
            max_response_length=max_response_length,
            soft_start=length_penalty_start,
            max_penalty=length_penalty_max,
        )
        reward_score = score.reward + length_cap_penalty
        response_length_cap_hit = (
            raw_response_length >= max_response_length
            if max_response_length > 0
            else False
        )
        mask_truncated = (
            _config_bool(
                agent_config,
                "mask_truncated",
                "CAUSAL_MICRO_LAB_MASK_TRUNCATED",
                False,
            )
            and response_length_cap_hit
        )

        response_ids, response_mask = _truncate_response_budget(
            response_ids,
            response_mask,
            max_response_length=max_response_length,
        )
        if mask_truncated:
            response_mask = [0] * len(response_mask)
        if not response_ids:
            response_ids = [self.tokenizer.eos_token_id]
            response_mask = [0]

        diagnostics = env.diagnostics()
        metrics = {
            "terminal_reward": reward_score,
            "base_terminal_reward": float(score.reward),
            "num_turns": 1.0,
            "parse_failures": float(score.parse_failures),
            "invalid_actions": float(score.invalid_actions),
            "valid_unique_count": float(score.valid_unique_count),
            "valid_committed_count": float(score.valid_committed_count),
            "validity": float(score.validity),
            "uniqueness": float(score.uniqueness),
            "final_version_space_size": float(
                score.metrics.get("final_version_space_size", 0)
            ),
            "current_version_space_size": float(
                score.metrics.get("current_version_space_size", 0)
            ),
            "recovery": float(score.metrics.get("recovery", 0.0)),
            "parse_valid": float(score.metrics.get("parse_valid", 0.0)),
            "syntax_valid": float(score.metrics.get("syntax_valid", 0.0)),
            "evidence_consistent": float(score.metrics.get("evidence_consistent", 0.0)),
            "valid_mode_count": float(score.metrics.get("valid_mode_count", 0.0)),
            "response_length_raw": float(raw_response_length),
            "response_length_penalty_start": float(length_penalty_start),
            "response_length_penalty_max": float(length_penalty_max),
            "response_length_cap_hit": float(response_length_cap_hit),
            "response_length_loss_masked": float(mask_truncated),
            "reward_length_cap": float(length_cap_penalty),
        }
        for key, value in score.breakdown.as_dict().items():
            metrics[f"reward_{key}"] = float(value)

        transcript = [
            {
                "role": "assistant",
                "content": assistant_text,
                "parse_ok": step.parse_ok,
            }
        ]
        if thinking_text:
            transcript[0]["thinking"] = thinking_text

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            reward_score=reward_score,
            num_turns=1,
            metrics=AgentLoopMetrics(
                generate_sequences=elapsed,
                tool_calls=0.0,
                compute_score=0.0,
                num_preempted=int(output.num_preempted or 0),
            ),
            extra_fields={
                "min_global_steps": 0,
                "max_global_steps": 0,
                "reward_extra_info": metrics,
                "score": {**score.as_dict(), "reward": reward_score},
                "diagnostics": diagnostics,
                "transcript": transcript,
            },
        )


@register("cd_grpo_agent_loop")
class CDGRPOAgentLoop(AgentLoopBase):  # type: ignore[misc]
    """Single-shot rollout with an oracle-free train-time reward payload."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> Any:
        if AgentLoopOutput is None or AgentLoopMetrics is None:
            raise RuntimeError("CDGRPOAgentLoop requires veRL to be installed.")

        env_spec = json.loads(_as_text(kwargs["env_spec_json"]))
        state_record = json.loads(_as_text(kwargs["state_json"]))
        task_config = env_spec.get("task") or {}
        agent_config = env_spec.get("agent") or {}
        max_response_length = int(self.rollout_config.response_length)
        request_id = _as_text(kwargs.get("uid", uuid4().hex))
        prompt = _as_text(kwargs.get("raw_prompt") or kwargs.get("prompt"))
        messages = [
            {
                "role": "system",
                "content": "You are solving a single-shot scientific hypothesis generation task.",
            },
            {"role": "user", "content": prompt},
        ]
        prompt_ids = await self.apply_chat_template(messages)

        started = time.monotonic()
        output = await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )
        elapsed = time.monotonic() - started
        response_ids = list(output.token_ids)
        raw_response_length = len(response_ids)
        response_logprobs = _generation_log_probs(
            output,
            expected_length=raw_response_length,
            required=bool(sampling_params.get("logprobs", False)),
        )
        cap_hit = max_response_length > 0 and raw_response_length >= max_response_length
        raw_assistant_text = self.tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        )
        assistant_text, thinking_text = split_visible_thinking(raw_assistant_text)

        method_config = self.config.algorithm.get(
            "cd_grpo",
            self.config.algorithm.get("ips_grpo", {}),
        )
        consequence = evaluate_consequences(
            assistant_text,
            state_record,
            truncated=cap_hit,
            probe_fraction=float(method_config.get("probe_fraction", 1.0)),
        )
        length_penalty_start = _config_int(
            method_config,
            "length_penalty_start",
            "CD_GRPO_LENGTH_PENALTY_START",
            _config_int(
                agent_config,
                "length_penalty_start",
                "CAUSAL_MICRO_LAB_LENGTH_PENALTY_START",
                int(max_response_length * 0.75),
            ),
        )
        length_penalty_max = _config_float(
            agent_config,
            "length_penalty_max",
            "CAUSAL_MICRO_LAB_LENGTH_PENALTY_MAX",
            -0.2,
        )
        length_penalty = _causal_micro_lab_length_cap_penalty(
            response_length=raw_response_length,
            max_response_length=max_response_length,
            soft_start=length_penalty_start,
            max_penalty=length_penalty_max,
        )
        base_reward, syntax_reward, validity_reward = base_candidate_reward(
            consequence,
            syntax_valid_reward=float(task_config.get("syntax_valid_reward", 0.2)),
            valid_hypothesis_reward=float(
                task_config.get("valid_hypothesis_reward", 1.0)
            ),
        )
        reward_score = base_reward + length_penalty

        response_mask = [1] * len(response_ids)
        response_ids, response_mask = _truncate_response_budget(
            response_ids,
            response_mask,
            max_response_length=max_response_length,
        )
        if response_logprobs is not None:
            response_logprobs = response_logprobs[: len(response_ids)]
        mask_truncated = (
            _config_bool(
                agent_config,
                "mask_truncated",
                "CAUSAL_MICRO_LAB_MASK_TRUNCATED",
                False,
            )
            and cap_hit
        )
        if mask_truncated:
            response_mask = [0] * len(response_mask)
        if not response_ids:
            response_ids = [self.tokenizer.eos_token_id]
            response_mask = [0]
            if response_logprobs is not None:
                response_logprobs = [0.0]

        reward_payload = {
            "status": consequence.status.value,
            "state_id": consequence.state_id,
            "consequence_signature": consequence.consequence_signature,
            "behavior_key": consequence.behavior_key,
        }
        metadata = state_record.get("metadata") or {}
        eval_payload = {
            "valid_mode_count": int(metadata.get("valid_mode_count", 0)),
            "separation_bucket": str(metadata.get("separation_bucket", "unknown")),
            "family_bucket": str(metadata.get("family_bucket", "unknown")),
        }
        behavior_hash_hi, behavior_hash_lo = _behavior_hash_parts(
            consequence.behavior_key
        )
        metrics = {
            "terminal_reward": float(reward_score),
            "base_terminal_reward": float(base_reward),
            "validity": float(consequence.valid),
            "parse_valid": float(
                consequence.status.value not in ("truncated", "parse_fail")
            ),
            "evidence_consistent": float(consequence.evidence_consistent),
            "response_length_raw": float(raw_response_length),
            "response_length_cap_hit": float(cap_hit),
            "response_length_loss_masked": float(mask_truncated),
            "reward_length_cap": float(length_penalty),
            "reward_syntax_valid": float(syntax_reward),
            "reward_valid_hypothesis": float(validity_reward),
            "cd_probe_count": float(len(consequence.probe_experiment_ids)),
            "ips_behavior_hash_hi": float(behavior_hash_hi),
            "ips_behavior_hash_lo": float(behavior_hash_lo),
            "valid_mode_count": float(eval_payload["valid_mode_count"]),
        }
        transcript = [{"role": "assistant", "content": assistant_text}]
        if thinking_text:
            transcript[0]["thinking"] = thinking_text

        reward_extra_info = {
            **metrics,
            # TransferQueue consistently preserves reward_extra_info across veRL
            # versions. Keep scalar JSON copies because TensorDict conversion
            # can drop nested mappings in some veRL/TransferQueue versions.
            "cd_reward_payload": reward_payload,
            "cd_eval_payload": eval_payload,
            "cd_reward_payload_json": json.dumps(
                reward_payload,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "cd_eval_payload_json": json.dumps(
                eval_payload,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            reward_score=reward_score,
            num_turns=1,
            metrics=AgentLoopMetrics(
                generate_sequences=elapsed,
                tool_calls=0.0,
                compute_score=0.0,
                num_preempted=int(output.num_preempted or 0),
            ),
            extra_fields={
                "min_global_steps": 0,
                "max_global_steps": 0,
                "reward_extra_info": reward_extra_info,
                "cd_reward_payload": reward_payload,
                "cd_eval_payload": eval_payload,
                "transcript": transcript,
            },
        )


@register("ips_grpo_agent_loop")
class IPSGRPOAgentLoop(CDGRPOAgentLoop):  # type: ignore[misc]
    """Consequence-aware rollout used by the IPS-GRPO trainer."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> Any:
        output = await super().run(sampling_params, **kwargs)
        reward_extra_info = output.extra_fields.get("reward_extra_info", {})
        for key in (
            "cd_reward_payload",
            "cd_eval_payload",
            "cd_reward_payload_json",
            "cd_eval_payload_json",
        ):
            reward_extra_info.pop(key, None)
            output.extra_fields.pop(key, None)
        from scattered_discovery.verl.ips_grpo_trainer import (
            normalize_ips_reward_extra_info,
        )

        output.extra_fields["reward_extra_info"] = normalize_ips_reward_extra_info(
            reward_extra_info,
            reward_score=output.reward_score,
        )
        return output
