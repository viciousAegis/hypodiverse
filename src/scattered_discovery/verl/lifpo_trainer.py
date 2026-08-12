from __future__ import annotations

from collections import Counter, defaultdict
import math
import time
from typing import Any


LIFPO_METADATA_KEYS = {
    "behavior_hash_hi",
    "behavior_hash_lo",
    "reward_payload",
    "eval_payload",
    "reward_payload_json",
    "eval_payload_json",
}

LIFPO_REWARD_EXTRA_DEFAULTS = {
    "terminal_reward": 0.0,
    "base_terminal_reward": 0.0,
    "validity": 0.0,
    "parse_valid": 0.0,
    "evidence_consistent": 0.0,
    "response_length_raw": 0.0,
    "response_length_cap_hit": 0.0,
    "response_length_loss_masked": 0.0,
    "reward_length_cap": 0.0,
    "reward_syntax_valid": 0.0,
    "reward_valid_hypothesis": 0.0,
    "probe_count": 0.0,
    "behavior_hash_hi": -1.0,
    "behavior_hash_lo": -1.0,
    "valid_mode_count": 0.0,
    "latent_enabled": 0.0,
    "latent_id": 0.0,
    "latent_negative_id": 0.0,
    "latent_answer_token_count": 0.0,
}


def normalize_lifpo_reward_extra_info(
    value: Any,
    *,
    reward_score: Any = 0.0,
) -> dict[str, float]:
    """Return the fixed scalar schema required by veRL batch postprocessing."""
    source = value if isinstance(value, dict) else {}
    defaults = dict(LIFPO_REWARD_EXTRA_DEFAULTS)
    defaults["terminal_reward"] = float(reward_score or 0.0)
    defaults["base_terminal_reward"] = float(reward_score or 0.0)
    return {key: float(source.get(key, default)) for key, default in defaults.items()}


def merge_lifpo_output_extra_fields(
    generated: dict[str, Any],
    input_extra_fields: Any,
) -> dict[str, Any]:
    """Preserve generated rollout metadata across veRL's kwargs overwrite."""
    merged = dict(input_extra_fields) if isinstance(input_extra_fields, dict) else {}
    merged.update(generated)
    reward_info = normalize_lifpo_reward_extra_info(
        merged.get("reward_extra_info"),
    )
    merged["reward_extra_info"] = reward_info
    # Some TransferQueue/TensorDict versions flatten or discard nested mappings.
    # Mirror the scalar contract so advantage computation has a stable fallback.
    merged.update(reward_info)
    return merged


def finish_trainer_wandb(trainer: Any, *, exit_code: int) -> None:
    """Flush W&B before Ray tears down the task-runner process."""
    tracking = getattr(trainer, "logger", None)
    backends = getattr(tracking, "logger", None)
    if not isinstance(backends, dict):
        return
    wandb_backend = backends.pop("wandb", None)
    if wandb_backend is not None:
        wandb_backend.finish(exit_code=exit_code)


try:
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopManager as _AgentLoopManager,
        AgentLoopWorker as _AgentLoopWorker,
    )
    from verl.trainer.ppo.v1.agent_loop_tq import (
        AgentLoopManagerTQ as _AgentLoopManagerTQ,
        AgentLoopWorkerTQ as _AgentLoopWorkerTQ,
    )
except ImportError:  # Keep local tests importable without veRL.
    _AgentLoopManager = None
    _AgentLoopWorker = None
    _AgentLoopManagerTQ = None
    _AgentLoopWorkerTQ = None


if _AgentLoopWorker is not None and _AgentLoopManager is not None:
    import ray

    @ray.remote
    class LIFPOAgentLoopWorker(_AgentLoopWorker):
        def _postprocess(
            self,
            inputs: list[Any],
            input_non_tensor_batch: dict[str, Any] | None = None,
            validate: bool = False,
        ) -> Any:
            for item in inputs:
                item.extra_fields["reward_extra_info"] = (
                    normalize_lifpo_reward_extra_info(
                        item.extra_fields.get("reward_extra_info"),
                        reward_score=getattr(item, "reward_score", 0.0),
                    )
                )
            return super()._postprocess(
                inputs,
                input_non_tensor_batch=input_non_tensor_batch,
                validate=validate,
            )

    class LIFPOAgentLoopManager(_AgentLoopManager):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.agent_loop_workers_class = LIFPOAgentLoopWorker
            super().__init__(*args, **kwargs)

else:

    class LIFPOAgentLoopManager:
        @classmethod
        def create(cls, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("LIFPOAgentLoopManager requires veRL.")


if _AgentLoopWorkerTQ is not None and _AgentLoopManagerTQ is not None:
    import ray

    @ray.remote
    class LIFPOAgentLoopWorkerTQ(_AgentLoopWorkerTQ.__ray_metadata__.modified_class):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.agent_loop_name = "lifpo_agent_loop"
            print(
                "LIFPO_AGENT_LOOP_SELECTION_ACTIVE=1 "
                f"agent_loop={self.agent_loop_name}",
                flush=True,
            )

        async def _run_agent_loop(self, *args: Any, **kwargs: Any) -> Any:
            # Shared dataset rows carry causal_micro_lab_agent_loop. Select the
            # method loop explicitly instead of relying on name remapping.
            kwargs["agent_name"] = self.agent_loop_name
            return await super()._run_agent_loop(*args, **kwargs)

        async def _agent_loop_postprocess(
            self,
            output: Any,
            validate: bool,
            **kwargs: Any,
        ) -> None:
            input_extra_fields = kwargs.pop("extra_fields", None)
            output.extra_fields = merge_lifpo_output_extra_fields(
                output.extra_fields,
                input_extra_fields,
            )
            # AgentLoopWorkerTQ calls field.update(kwargs). Supplying the same
            # merged value makes this safe even if that veRL version expects an
            # input `extra_fields` key instead of allowing it to be removed.
            kwargs["extra_fields"] = output.extra_fields
            # veRL derives the batch schema from the first rollout and indexes
            # every later rollout with those keys. Normalize failure/padding
            # paths before that unguarded batch assembly.
            output.extra_fields["reward_extra_info"] = (
                normalize_lifpo_reward_extra_info(
                    output.extra_fields.get("reward_extra_info"),
                    reward_score=getattr(output, "reward_score", 0.0),
                )
            )
            return await super()._agent_loop_postprocess(
                output,
                validate,
                **kwargs,
            )

    class LIFPOAgentLoopManagerTQ(_AgentLoopManagerTQ):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.agent_loop_workers_class = LIFPOAgentLoopWorkerTQ

else:

    class LIFPOAgentLoopManagerTQ:
        @classmethod
        def create(cls, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("LIFPOAgentLoopManagerTQ requires veRL.")


def _config_value(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _as_float(value: Any, *, field: str, index: int) -> float:
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{field} at rollout index {index} must be numeric, got {value!r}"
        ) from error


def _as_int_list(value: Any, *, field: str, index: int) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field} at rollout index {index} must be a token-id sequence")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{field} at rollout index {index} contains a non-integer token ID"
        ) from error


_MISSING = object()


def _field_get(value: Any, key: str, default: Any = None) -> Any:
    """Read mapping-like fields without invoking TensorClass sequence indexing."""
    getter = getattr(value, "get", None)
    if not callable(getter):
        return default
    try:
        return getter(key, default)
    except (KeyError, RuntimeError, TypeError):
        return default


def _field_keys(value: Any) -> list[str]:
    keys = getattr(value, "keys", None)
    if not callable(keys):
        return []
    try:
        return sorted(str(key) for key in keys())
    except (RuntimeError, TypeError):
        return []


def scatter_real_rows(
    template: Any,
    real_index_tensor: Any,
    real_values: Any,
) -> Any:
    output = real_values.new_zeros(template.shape)
    output.index_copy_(0, real_index_tensor, real_values)
    return output


def sanitize_validation_reward_extras(
    reward_extra_infos: dict[str, list[Any]],
) -> tuple[dict[str, list[Any]], dict[str, float]]:
    """Remove outcome identifiers and incomplete auxiliary series from val logs."""
    sanitized: dict[str, list[Any]] = {}
    missing_rates: dict[str, float] = {}
    for key, values in reward_extra_infos.items():
        if key in LIFPO_METADATA_KEYS:
            continue
        missing = sum(value is None for value in values)
        if not missing:
            sanitized[key] = values
            continue
        if key == "reward":
            raise ValueError("validation reward series contains missing values")
        missing_rates[key] = missing / max(1, len(values))
    return sanitized, missing_rates


def select_lifpo_metadata(
    extra_fields: list[dict[str, Any]],
    tags: list[dict[str, Any]],
) -> tuple[list[int], list[dict[str, Any]]]:
    """Extract numeric LIFPO metadata while excluding veRL padding rows."""
    if len(extra_fields) != len(tags):
        raise ValueError("extra_fields and tags must have equal length")

    real_indices: list[int] = []
    metadata: list[dict[str, Any]] = []
    for index, (item, tag) in enumerate(zip(extra_fields, tags, strict=True)):
        if bool(tag.get("is_padding", False)):
            continue
        reward_info = _field_get(item, "reward_extra_info", {})
        if not isinstance(reward_info, dict):
            reward_info = {}
        required = {
            "validity",
            "reward_valid_hypothesis",
            "behavior_hash_hi",
            "behavior_hash_lo",
        }
        # TransferQueue versions differ in whether nested dictionaries survive
        # TensorDict conversion. Prefer reward_extra_info, then mirrored scalars.
        normalized_reward_info: dict[str, Any] = {}
        for key in required | set(LIFPO_REWARD_EXTRA_DEFAULTS):
            value = reward_info.get(key, _MISSING)
            if value is _MISSING:
                value = _field_get(item, key, _MISSING)
            if value is not _MISSING:
                normalized_reward_info[key] = value
        reward_info = normalized_reward_info
        missing = sorted(key for key in required if reward_info.get(key) is None)
        if missing:
            raise KeyError(
                f"real rollout at index {index} is missing LIFPO metadata: "
                + ", ".join(missing)
                + f"; available extra_fields keys: {_field_keys(item)}"
            )

        valid = _as_float(reward_info["validity"], field="validity", index=index) > 0
        valid_reward = _as_float(
            reward_info["reward_valid_hypothesis"],
            field="reward_valid_hypothesis",
            index=index,
        )
        hash_hi = int(
            _as_float(
                reward_info["behavior_hash_hi"],
                field="behavior_hash_hi",
                index=index,
            )
        )
        hash_lo = int(
            _as_float(
                reward_info["behavior_hash_lo"],
                field="behavior_hash_lo",
                index=index,
            )
        )
        if valid and (hash_hi < 0 or hash_lo < 0):
            raise ValueError(f"valid rollout at index {index} has no behavior identity")
        if not valid and valid_reward != 0.0:
            raise ValueError(
                f"invalid rollout at index {index} has nonzero validity reward"
            )

        real_indices.append(index)
        metadata.append(
            {
                "valid": valid,
                "valid_reward": valid_reward,
                "behavior": (hash_hi, hash_lo) if valid else None,
                "latent_enabled": _as_float(
                    reward_info.get("latent_enabled", 0.0),
                    field="latent_enabled",
                    index=index,
                )
                > 0,
                "latent_id": int(
                    _as_float(
                        reward_info.get("latent_id", 0.0),
                        field="latent_id",
                        index=index,
                    )
                ),
                "latent_negative_id": int(
                    _as_float(
                        reward_info.get("latent_negative_id", 0.0),
                        field="latent_negative_id",
                        index=index,
                    )
                ),
                "answer_token_count": int(
                    _as_float(
                        reward_info.get("latent_answer_token_count", 0.0),
                        field="latent_answer_token_count",
                        index=index,
                    )
                ),
                "valid_mode_count": int(
                    _as_float(
                        reward_info.get("valid_mode_count", 0.0),
                        field="valid_mode_count",
                        index=index,
                    )
                ),
            }
        )

    if not real_indices:
        raise ValueError("LIFPO batch contains no real rollouts")
    return real_indices, metadata


def compute_inverse_frequency_advantages(
    *,
    token_level_rewards: Any,
    response_mask: Any,
    index: list[object],
    metadata: list[dict[str, Any]],
    epsilon: float,
) -> tuple[Any, Any, dict[str, float]]:
    """Apply group empirical inverse-propensity scaling, then standard GRPO."""
    import torch

    if not 0.0 < epsilon <= 1.0:
        raise ValueError("LIFPO epsilon must be in (0, 1]")
    if token_level_rewards.shape != response_mask.shape:
        raise ValueError("token_level_rewards and response_mask must match")
    if len(index) != token_level_rewards.shape[0] or len(metadata) != len(index):
        raise ValueError("index, metadata, and reward rows must have equal length")

    raw_scores = token_level_rewards.sum(dim=-1)
    frequency_scores = raw_scores.clone()
    advantage_scalars = torch.zeros_like(raw_scores)
    groups: dict[object, list[int]] = defaultdict(list)
    for row_index, group_id in enumerate(index):
        groups[group_id].append(row_index)

    valid_count = 0
    duplicate_valid_count = 0
    clipped_valid_count = 0
    singleton_valid_count = 0
    groups_without_valid = 0
    groups_with_multiple_outcomes = 0
    weights: list[float] = []
    probabilities: list[float] = []
    unique_outcomes_total = 0

    for group_indices in groups.values():
        outcome_counts = Counter(
            metadata[row_index]["behavior"]
            for row_index in group_indices
            if metadata[row_index]["valid"]
        )
        if not outcome_counts:
            groups_without_valid += 1
        if len(outcome_counts) >= 2:
            groups_with_multiple_outcomes += 1
        unique_outcomes_total += len(outcome_counts)
        duplicate_valid_count += sum(
            max(0, count - 1) for count in outcome_counts.values()
        )
        group_size = len(group_indices)

        for row_index in group_indices:
            item = metadata[row_index]
            if not item["valid"]:
                continue
            count = outcome_counts[item["behavior"]]
            probability = count / group_size
            denominator = max(probability, epsilon)
            weight = 1.0 / denominator
            validity_reward = float(item["valid_reward"])
            frequency_scores[row_index] = (
                raw_scores[row_index] - validity_reward + validity_reward * weight
            )
            valid_count += 1
            singleton_valid_count += int(count == 1)
            clipped_valid_count += int(probability < epsilon)
            weights.append(weight)
            probabilities.append(probability)

        group_scores = frequency_scores[group_indices]
        if len(group_indices) > 1:
            group_advantages = (group_scores - group_scores.mean()) / (
                group_scores.std(unbiased=True) + 1e-6
            )
            advantage_scalars[group_indices] = group_advantages

    advantages = advantage_scalars.unsqueeze(-1) * response_mask
    group_count = len(groups)
    total_rows = len(metadata)
    metrics = {
        "lifpo/epsilon": float(epsilon),
        "lifpo/validity_rate": valid_count / max(1, total_rows),
        "lifpo/raw_score_mean": float(raw_scores.mean().item()),
        "lifpo/scaled_score_mean": float(frequency_scores.mean().item()),
        "lifpo/scaled_score_max": float(frequency_scores.max().item()),
        "lifpo/weight_mean": sum(weights) / max(1, len(weights)),
        "lifpo/weight_max": max(weights, default=0.0),
        "lifpo/empirical_probability_mean": sum(probabilities)
        / max(1, len(probabilities)),
        "lifpo/clipped_valid_rate": clipped_valid_count / max(1, valid_count),
        "lifpo/singleton_valid_rate": singleton_valid_count / max(1, valid_count),
        "lifpo/duplicate_valid_rate": duplicate_valid_count / max(1, valid_count),
        "lifpo/unique_valid_outcomes_per_group": unique_outcomes_total
        / max(1, group_count),
        "lifpo/groups_without_valid_rate": groups_without_valid / max(1, group_count),
        "lifpo/groups_with_multiple_valid_outcomes_rate": (
            groups_with_multiple_outcomes / max(1, group_count)
        ),
        "lifpo/advantage_abs_mean": float(advantage_scalars.abs().mean().item()),
        "lifpo/advantage_std": float(advantage_scalars.std(unbiased=False).item()),
    }
    return advantages, advantages, metrics


def _contrastive_token_scores(
    assigned_log_probs: Any,
    negative_log_probs: Any,
) -> Any:
    """Return log(2 p_assigned / (p_assigned + p_negative)) stably."""
    import torch

    return (
        math.log(2.0)
        + assigned_log_probs
        - torch.logaddexp(assigned_log_probs, negative_log_probs)
    )


def compute_lifpo_advantages(
    *,
    token_level_rewards: Any,
    response_mask: Any,
    assigned_log_probs: Any,
    negative_log_probs: Any,
    index: list[object],
    metadata: list[dict[str, Any]],
    epsilon: float,
    counterfactual_alpha: float,
    counterfactual_clip: float,
    inverse_frequency_enabled: bool,
    latent_count: int,
    frequency_credit_mode: str = "replace",
    frequency_credit_max: float = 0.25,
    counterfactual_token_scope: str = "answer",
    counterfactual_reduction: str = "mean",
    counterfactual_valid_only: bool = True,
    metric_prefix: str = "lifpo",
) -> tuple[Any, Any, dict[str, float]]:
    """Combine validity, latent specificity, and optional empirical inverse-frequency credit."""
    import torch

    if not 0.0 < epsilon <= 1.0:
        raise ValueError("LIFPO epsilon must be in (0, 1]")
    if counterfactual_alpha < 0.0:
        raise ValueError("counterfactual alpha must be non-negative")
    if counterfactual_clip <= 0.0:
        raise ValueError("counterfactual clip must be positive")
    if counterfactual_token_scope not in {"answer", "full_response"}:
        raise ValueError(
            "counterfactual token scope must be 'answer' or 'full_response'"
        )
    if counterfactual_reduction not in {"mean", "sum"}:
        raise ValueError("counterfactual reduction must be 'mean' or 'sum'")
    if latent_count < 2:
        raise ValueError("latent_count must be at least two")
    if frequency_credit_mode not in {"replace", "bonus"}:
        raise ValueError("LIFPO reward mode must be 'replace' or 'bonus'")
    if frequency_credit_max < 0.0:
        raise ValueError("LIFPO bonus maximum must be non-negative")
    if not metric_prefix or "/" in metric_prefix:
        raise ValueError("metric_prefix must be one non-empty path component")
    shapes = {
        tuple(token_level_rewards.shape),
        tuple(response_mask.shape),
        tuple(assigned_log_probs.shape),
        tuple(negative_log_probs.shape),
    }
    if len(shapes) != 1:
        raise ValueError("reward, mask, and log-probability tensors must match")
    if len(index) != token_level_rewards.shape[0] or len(metadata) != len(index):
        raise ValueError("index, metadata, and reward rows must have equal length")

    raw_scores = token_level_rewards.sum(dim=-1)
    shaped_scores = raw_scores.clone()
    advantage_scalars = torch.zeros_like(raw_scores)
    groups: dict[object, list[int]] = defaultdict(list)
    for row_index, group_id in enumerate(index):
        groups[group_id].append(row_index)

    counterfactual_raw_values: list[float] = []
    counterfactual_clipped_values: list[float] = []
    counterfactual_contributions: list[float] = []
    assigned_answer_logps: list[float] = []
    negative_answer_logps: list[float] = []
    assigned_trajectory_logps: list[float] = []
    negative_trajectory_logps: list[float] = []
    answer_counterfactual_raw_values: list[float] = []
    weights: list[float] = []
    probabilities: list[float] = []
    frequency_credits: list[float] = []
    valid_count = 0
    duplicate_valid_count = 0
    singleton_valid_count = 0
    clipped_weight_count = 0
    groups_without_valid = 0
    groups_with_multiple_outcomes = 0
    groups_all_latents_present = 0
    valid_cross_latent_pairs = 0
    same_outcome_cross_latent_pairs = 0
    unique_outcomes_total = 0
    row_counterfactual: list[float | None] = [None] * len(metadata)
    row_weights: list[float | None] = [None] * len(metadata)

    for group_indices in groups.values():
        outcome_counts = Counter(
            metadata[row_index]["behavior"]
            for row_index in group_indices
            if metadata[row_index]["valid"]
        )
        latent_ids = {
            metadata[row_index]["latent_id"]
            for row_index in group_indices
            if metadata[row_index].get("latent_enabled", False)
        }
        expected_latents = set(range(1, latent_count + 1))
        groups_all_latents_present += int(
            bool(expected_latents) and latent_ids == expected_latents
        )
        if not outcome_counts:
            groups_without_valid += 1
        if len(outcome_counts) >= 2:
            groups_with_multiple_outcomes += 1
        unique_outcomes_total += len(outcome_counts)
        duplicate_valid_count += sum(
            max(0, count - 1) for count in outcome_counts.values()
        )
        group_size = len(group_indices)

        valid_rows = [
            row_index for row_index in group_indices if metadata[row_index]["valid"]
        ]
        for left_offset, left_index in enumerate(valid_rows):
            for right_index in valid_rows[left_offset + 1 :]:
                if (
                    metadata[left_index]["latent_id"]
                    == metadata[right_index]["latent_id"]
                ):
                    continue
                valid_cross_latent_pairs += 1
                same_outcome_cross_latent_pairs += int(
                    metadata[left_index]["behavior"]
                    == metadata[right_index]["behavior"]
                )

        for row_index in group_indices:
            item = metadata[row_index]
            is_valid = bool(item["valid"])
            weight = 1.0
            if is_valid:
                valid_count += 1
                count = outcome_counts[item["behavior"]]
                probability = count / group_size
                denominator = max(probability, epsilon)
                weight = 1.0 / denominator if inverse_frequency_enabled else 1.0
                singleton_valid_count += int(count == 1)
                clipped_weight_count += int(
                    inverse_frequency_enabled and probability < epsilon
                )
                weights.append(weight)
                probabilities.append(probability)
                row_weights[row_index] = weight

            active = torch.nonzero(
                response_mask[row_index] > 0,
                as_tuple=False,
            ).flatten()
            answer_count = min(int(item["answer_token_count"]), len(active))
            counterfactual_raw = 0.0
            answer_counterfactual_raw = 0.0
            assigned_mean = 0.0
            negative_mean = 0.0
            assigned_trajectory_mean = 0.0
            negative_trajectory_mean = 0.0
            counterfactual_eligible = is_valid or not counterfactual_valid_only
            if counterfactual_eligible and len(active) > 0:
                trajectory_assigned = assigned_log_probs[row_index, active]
                trajectory_negative = negative_log_probs[row_index, active]
                assigned_trajectory_mean = float(trajectory_assigned.mean().item())
                negative_trajectory_mean = float(trajectory_negative.mean().item())
                if counterfactual_token_scope == "answer":
                    selected_indices = (
                        active[-answer_count:] if answer_count > 0 else active[:0]
                    )
                else:
                    selected_indices = active
                selected_scores = _contrastive_token_scores(
                    assigned_log_probs[row_index, selected_indices],
                    negative_log_probs[row_index, selected_indices],
                )
                if len(selected_scores) > 0:
                    reduced = (
                        selected_scores.sum()
                        if counterfactual_reduction == "sum"
                        else selected_scores.mean()
                    )
                    counterfactual_raw = float(reduced.item())
            if counterfactual_eligible and answer_count > 0:
                answer_indices = active[-answer_count:]
                assigned = assigned_log_probs[row_index, answer_indices]
                negative = negative_log_probs[row_index, answer_indices]
                answer_counterfactual_raw = float(
                    _contrastive_token_scores(assigned, negative).mean().item()
                )
                assigned_mean = float(assigned.mean().item())
                negative_mean = float(negative.mean().item())
            counterfactual_clipped = max(
                -counterfactual_clip, min(counterfactual_clip, counterfactual_raw)
            )
            contribution = counterfactual_alpha * counterfactual_clipped
            frequency_adjustment = 0.0
            if is_valid:
                validity_reward = float(item["valid_reward"])
                if inverse_frequency_enabled and frequency_credit_mode == "bonus":
                    max_weight = 1.0 / epsilon
                    normalized_rarity = (
                        (weight - 1.0) / (max_weight - 1.0) if max_weight > 1.0 else 0.0
                    )
                    frequency_adjustment = (
                        validity_reward * frequency_credit_max * normalized_rarity
                    )
                    frequency_credits.append(frequency_adjustment)
                else:
                    frequency_adjustment = validity_reward * (weight - 1.0)
            shaped_scores[row_index] = (
                raw_scores[row_index] + frequency_adjustment + contribution
            )
            if counterfactual_eligible:
                counterfactual_raw_values.append(counterfactual_raw)
                counterfactual_clipped_values.append(counterfactual_clipped)
                counterfactual_contributions.append(contribution)
                answer_counterfactual_raw_values.append(answer_counterfactual_raw)
                assigned_answer_logps.append(assigned_mean)
                negative_answer_logps.append(negative_mean)
                assigned_trajectory_logps.append(assigned_trajectory_mean)
                negative_trajectory_logps.append(negative_trajectory_mean)
                row_counterfactual[row_index] = counterfactual_raw

        group_scores = shaped_scores[group_indices]
        if len(group_indices) > 1:
            group_advantages = (group_scores - group_scores.mean()) / (
                group_scores.std(unbiased=True) + 1e-6
            )
            advantage_scalars[group_indices] = group_advantages

    advantages = advantage_scalars.unsqueeze(-1) * response_mask
    group_count = len(groups)
    total_rows = len(metadata)

    def mean(values: list[float]) -> float:
        return sum(values) / max(1, len(values))

    counterfactual_tensor = torch.tensor(counterfactual_raw_values, dtype=torch.float32)
    configured_valid_rewards = [
        float(item["valid_reward"])
        for item in metadata
        if float(item["valid_reward"]) > 0.0
    ]
    valid_reward_max = max(configured_valid_rewards, default=0.0)
    metrics = {
        "lifpo/inverse_frequency_enabled": float(inverse_frequency_enabled),
        "lifpo/epsilon": float(epsilon),
        "lifpo/frequency_credit_mode_bonus": float(frequency_credit_mode == "bonus"),
        "lifpo/frequency_credit_bound": float(
            frequency_credit_max if inverse_frequency_enabled else 0.0
        ),
        "lifpo/frequency_credit_mean": mean(frequency_credits),
        "lifpo/frequency_credit_max_observed": max(frequency_credits, default=0.0),
        "lifpo/counterfactual_alpha": float(counterfactual_alpha),
        "lifpo/counterfactual_clip": float(counterfactual_clip),
        "lifpo/counterfactual_reward_bound": float(
            counterfactual_alpha * counterfactual_clip
        ),
        "lifpo/valid_reward_max": valid_reward_max,
        "lifpo/counterfactual_to_valid_reward_bound_ratio": (
            counterfactual_alpha * counterfactual_clip / valid_reward_max
            if valid_reward_max > 0.0
            else 0.0
        ),
        "lifpo/counterfactual_scope_full_response": float(
            counterfactual_token_scope == "full_response"
        ),
        "lifpo/counterfactual_reduction_sum": float(counterfactual_reduction == "sum"),
        "lifpo/counterfactual_valid_only": float(counterfactual_valid_only),
        "lifpo/validity_rate": valid_count / max(1, total_rows),
        "lifpo/raw_score_mean": float(raw_scores.mean().item()),
        "lifpo/shaped_score_mean": float(shaped_scores.mean().item()),
        "lifpo/shaped_score_max": float(shaped_scores.max().item()),
        "lifpo/counterfactual_raw_mean": mean(counterfactual_raw_values),
        "lifpo/counterfactual_raw_std": (
            float(counterfactual_tensor.std(unbiased=False).item())
            if counterfactual_raw_values
            else 0.0
        ),
        "lifpo/counterfactual_clipped_mean": mean(counterfactual_clipped_values),
        "lifpo/counterfactual_reward_mean": mean(counterfactual_contributions),
        "lifpo/counterfactual_reward_abs_mean": mean(
            [abs(value) for value in counterfactual_contributions]
        ),
        "lifpo/counterfactual_reward_max_abs": max(
            (abs(value) for value in counterfactual_contributions),
            default=0.0,
        ),
        "lifpo/counterfactual_clip_rate": sum(
            abs(raw) >= counterfactual_clip for raw in counterfactual_raw_values
        )
        / max(1, len(counterfactual_raw_values)),
        "lifpo/counterfactual_nonzero_rate": sum(
            abs(value) > 1e-8 for value in counterfactual_raw_values
        )
        / max(1, len(counterfactual_raw_values)),
        "lifpo/answer_counterfactual_raw_mean": mean(answer_counterfactual_raw_values),
        "lifpo/assigned_answer_logp_mean": mean(assigned_answer_logps),
        "lifpo/negative_answer_logp_mean": mean(negative_answer_logps),
        "lifpo/answer_logp_margin_mean": mean(
            [
                assigned - negative
                for assigned, negative in zip(
                    assigned_answer_logps,
                    negative_answer_logps,
                    strict=True,
                )
            ]
        ),
        "lifpo/assigned_trajectory_logp_mean": mean(assigned_trajectory_logps),
        "lifpo/negative_trajectory_logp_mean": mean(negative_trajectory_logps),
        "lifpo/trajectory_logp_margin_mean": mean(
            [
                assigned - negative
                for assigned, negative in zip(
                    assigned_trajectory_logps,
                    negative_trajectory_logps,
                    strict=True,
                )
            ]
        ),
        "lifpo/answer_tokens_mean": mean(
            [float(item["answer_token_count"]) for item in metadata]
        ),
        "lifpo/valid_answer_tokens_mean": mean(
            [float(item["answer_token_count"]) for item in metadata if item["valid"]]
        ),
        "lifpo/weight_mean": mean(weights),
        "lifpo/weight_max": max(weights, default=0.0),
        "lifpo/empirical_probability_mean": mean(probabilities),
        "lifpo/clipped_valid_rate": clipped_weight_count / max(1, valid_count),
        "lifpo/singleton_valid_rate": singleton_valid_count / max(1, valid_count),
        "lifpo/duplicate_valid_rate": duplicate_valid_count / max(1, valid_count),
        "lifpo/unique_valid_outcomes_per_group": unique_outcomes_total
        / max(1, group_count),
        "lifpo/groups_without_valid_rate": groups_without_valid / max(1, group_count),
        "lifpo/groups_with_multiple_valid_outcomes_rate": (
            groups_with_multiple_outcomes / max(1, group_count)
        ),
        "lifpo/groups_all_latents_present_rate": groups_all_latents_present
        / max(1, group_count),
        "lifpo/cross_latent_outcome_collision_rate": (
            same_outcome_cross_latent_pairs / max(1, valid_cross_latent_pairs)
        ),
        "lifpo/advantage_abs_mean": float(advantage_scalars.abs().mean().item()),
        "lifpo/advantage_std": float(advantage_scalars.std(unbiased=False).item()),
    }
    latent_values = sorted(
        {item["latent_id"] for item in metadata if item.get("latent_enabled", False)}
    )
    for latent_id in latent_values:
        latent_rows = [
            row_index
            for row_index, item in enumerate(metadata)
            if item["latent_id"] == latent_id
        ]
        metrics[f"lifpo/latent_{latent_id}/rows"] = float(len(latent_rows))
        metrics[f"lifpo/latent_{latent_id}/validity_rate"] = sum(
            metadata[row_index]["valid"] for row_index in latent_rows
        ) / max(1, len(latent_rows))
        valid_latent_rows = [
            row_index for row_index in latent_rows if metadata[row_index]["valid"]
        ]
        metrics[f"lifpo/latent_{latent_id}/unique_valid_outcomes"] = float(
            len({metadata[row_index]["behavior"] for row_index in valid_latent_rows})
        )
        metrics[f"lifpo/latent_{latent_id}/counterfactual_raw_mean"] = mean(
            [
                float(row_counterfactual[row_index])
                for row_index in latent_rows
                if row_counterfactual[row_index] is not None
            ]
        )
        metrics[f"lifpo/latent_{latent_id}/weight_mean"] = mean(
            [
                float(row_weights[row_index])
                for row_index in valid_latent_rows
                if row_weights[row_index] is not None
            ]
        )

    mode_counts = sorted(
        {
            int(item.get("valid_mode_count", 0))
            for item in metadata
            if int(item.get("valid_mode_count", 0)) > 0
        }
    )
    for mode_count in mode_counts:
        mode_rows = [
            row_index
            for row_index, item in enumerate(metadata)
            if int(item.get("valid_mode_count", 0)) == mode_count
        ]
        mode_group_ids = {index[row_index] for row_index in mode_rows}
        unique_by_group = []
        for group_id in mode_group_ids:
            unique_by_group.append(
                len(
                    {
                        metadata[row_index]["behavior"]
                        for row_index in mode_rows
                        if index[row_index] == group_id and metadata[row_index]["valid"]
                    }
                )
            )
        metrics[f"lifpo/M_{mode_count}/rows"] = float(len(mode_rows))
        metrics[f"lifpo/M_{mode_count}/validity_rate"] = sum(
            metadata[row_index]["valid"] for row_index in mode_rows
        ) / max(1, len(mode_rows))
        metrics[f"lifpo/M_{mode_count}/unique_valid_outcomes_per_group"] = mean(
            [float(value) for value in unique_by_group]
        )
    if metric_prefix != "lifpo":
        metrics = {
            (
                f"{metric_prefix}/{key.removeprefix('lifpo/')}"
                if key.startswith("lifpo/")
                else key
            ): value
            for key, value in metrics.items()
        }
    return advantages, advantages, metrics


class LIFPOTrainerMixin:
    """veRL v1 trainer extension implementing the LIFPO objective."""

    def _lifpo_config(self) -> Any:
        return self.config.algorithm.get("lifpo", {})

    def _latent_enabled(self) -> bool:
        return bool(_config_value(self._lifpo_config(), "latent_enabled", True))

    def _compute_old_log_prob(
        self,
        batch: Any,
        metrics: dict[str, Any],
    ) -> Any:
        batch = super()._compute_old_log_prob(batch, metrics)
        if not self._latent_enabled():
            return batch

        import torch
        from tensordict import TensorDict
        import transfer_queue as tq
        from transfer_queue import KVBatchMeta
        from verl.utils.tensordict_utils import get as get_tensordict_field
        from verl.workers.utils.padding import response_from_nested

        fields = ["prompts", "responses", "response_mask", "extra_fields"]
        source = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=fields,
        )
        prompt_rows = list(source["prompts"].unbind())
        response_rows = list(source["responses"].unbind())
        response_mask = source["response_mask"]
        response_mask_rows = list(response_mask.unbind())
        extra_rows = get_tensordict_field(source, "extra_fields", [])
        if not (
            len(prompt_rows)
            == len(response_rows)
            == len(response_mask_rows)
            == len(extra_rows)
            == len(batch)
        ):
            raise ValueError("counterfactual scorer received misaligned TQ rows")

        counterfactual_inputs: list[Any] = []
        counterfactual_positions: list[Any] = []
        counterfactual_loss_masks: list[Any] = []
        scored_tokens = 0
        for row_index, (
            prompt_row,
            response_row,
            response_mask_row,
            extra,
        ) in enumerate(
            zip(
                prompt_rows,
                response_rows,
                response_mask_rows,
                extra_rows,
                strict=True,
            )
        ):
            negative_prompt = (
                extra.get("latent_negative_prompt_ids")
                if isinstance(extra, dict)
                else None
            )
            if negative_prompt is None:
                if not bool(batch.tags[row_index].get("is_padding", False)):
                    raise KeyError(
                        "real latent rollout is missing latent_negative_prompt_ids"
                    )
                negative_prompt_ids = [int(item) for item in prompt_row.tolist()]
            else:
                negative_prompt_ids = _as_int_list(
                    negative_prompt,
                    field="latent_negative_prompt_ids",
                    index=row_index,
                )
            response_ids = [int(item) for item in response_row.tolist()]
            if len(response_ids) != len(response_mask_row):
                raise ValueError(
                    "counterfactual scorer received a response/mask length mismatch"
                )
            full_ids = negative_prompt_ids + response_ids
            counterfactual_inputs.append(torch.tensor(full_ids, dtype=torch.long))
            counterfactual_positions.append(
                torch.arange(len(full_ids), dtype=torch.long)
            )
            counterfactual_loss_masks.append(
                torch.cat(
                    (
                        response_mask_row.new_zeros(len(negative_prompt_ids)),
                        response_mask_row,
                    )
                )
            )
            scored_tokens += len(full_ids)

        nested_inputs = torch.nested.as_nested_tensor(
            counterfactual_inputs,
            layout=torch.jagged,
        )
        nested_positions = torch.nested.as_nested_tensor(
            counterfactual_positions,
            layout=torch.jagged,
        )
        nested_loss_masks = torch.nested.as_nested_tensor(
            counterfactual_loss_masks,
            layout=torch.jagged,
        )
        shadow_keys = [
            f"{key}__latent_negative_{self.global_steps}" for key in batch.keys
        ]
        shadow_tags = []
        for tag, input_row in zip(
            batch.tags,
            counterfactual_inputs,
            strict=True,
        ):
            shadow_tag = dict(tag)
            shadow_tag["seq_len"] = len(input_row)
            shadow_tags.append(shadow_tag)
        shadow_fields = TensorDict(
            {
                "input_ids": nested_inputs,
                "position_ids": nested_positions,
                "loss_mask": nested_loss_masks,
            },
            batch_size=len(batch),
        )
        tq.kv_batch_put(
            keys=shadow_keys,
            partition_id=batch.partition_id,
            tags=shadow_tags,
            fields=shadow_fields,
        )
        shadow_batch = KVBatchMeta(
            keys=shadow_keys,
            partition_id=batch.partition_id,
            tags=shadow_tags,
        )
        shadow_batch.extra_info.update(
            {
                "calculate_entropy": False,
                "compute_loss": False,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            }
        )

        started = time.monotonic()
        try:
            output = self.actor_rollout_wg.compute_log_prob(shadow_batch)
            if len(output) != len(shadow_batch):
                raise RuntimeError(
                    "counterfactual actor scoring returned the wrong batch size"
                )
            scored = tq.kv_batch_get(
                keys=shadow_keys,
                partition_id=batch.partition_id,
                select_fields=["log_probs"],
            )
            negative_log_probs = response_from_nested(
                scored["log_probs"],
                response_mask,
            )
            tq.kv_batch_put(
                keys=batch.keys,
                partition_id=batch.partition_id,
                fields=TensorDict(
                    {"latent_negative_log_probs": negative_log_probs},
                    batch_size=len(batch),
                ),
            )
        finally:
            tq.kv_clear(
                keys=shadow_keys,
                partition_id=batch.partition_id,
            )

        elapsed = time.monotonic() - started
        metrics.update(
            {
                "lifpo/counterfactual_scored_tokens": float(scored_tokens),
                "lifpo/counterfactual_score_seconds": elapsed,
                "lifpo/counterfactual_tokens_per_second": (
                    scored_tokens / max(1e-9, elapsed)
                ),
            }
        )
        return batch

    def _val_metrics_update(
        self,
        data_sources: list[Any],
        sample_uids: list[Any],
        reward_extra_infos_dict: dict[str, list[Any]],
        sample_turns: list[Any],
    ) -> dict[str, float]:
        sanitized, missing_rates = sanitize_validation_reward_extras(
            reward_extra_infos_dict
        )
        metrics = super()._val_metrics_update(
            data_sources,
            sample_uids,
            sanitized,
            sample_turns,
        )
        for key, rate in missing_rates.items():
            safe_key = key.replace("/", "_")
            metrics[f"val-aux/lifpo/missing_reward_extra/{safe_key}"] = rate
        return metrics

    def _compute_advantage(self, batch: Any, metrics: dict[str, Any]) -> Any:
        import numpy as np
        import torch
        from tensordict import TensorDict
        import transfer_queue as tq
        from verl.protocol import DataProto
        from verl.trainer.ppo.ray_trainer import apply_kl_penalty
        from verl.trainer.ppo.rollout_corr_helper import (
            compute_rollout_correction_and_add_to_batch,
        )
        from verl.utils.tensordict_utils import pop as pop_tensordict_field
        from verl.workers.utils.padding import response_to_nested

        fields = [
            "uid",
            "response_mask",
            "rm_scores",
            "rollout_log_probs",
            "old_log_probs",
            "ref_log_prob",
            "values",
            "extra_fields",
        ]
        if self._latent_enabled():
            fields.append("latent_negative_log_probs")
        raw = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=fields,
        )
        response_mask_nested = raw["response_mask"]
        extra_fields = pop_tensordict_field(raw, "extra_fields", [])
        tags = list(batch.tags)
        real_indices, frequency_metadata = select_lifpo_metadata(extra_fields, tags)
        data = DataProto(batch=raw.to_padded_tensor())
        data.batch["token_level_scores"] = data.batch["rm_scores"]
        uid = np.array(data.batch.pop("uid").tolist(), dtype=object)
        data.non_tensor_batch["uid"] = uid

        if self.config.algorithm.use_kl_in_reward:
            data, kl_metrics = apply_kl_penalty(
                data,
                kl_ctrl=self.kl_ctrl_in_reward,
                kl_penalty=self.config.algorithm.kl_penalty,
            )
            metrics.update(kl_metrics)
        else:
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]

        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass = rollout_corr_config and rollout_corr_config.get(
            "bypass_mode",
            False,
        )
        rollout_correction = (
            rollout_corr_config is not None
            and "rollout_log_probs" in data.batch
            and not bypass
        )
        if rollout_correction:
            data, correction_metrics = compute_rollout_correction_and_add_to_batch(
                data,
                rollout_corr_config,
            )
            metrics.update(correction_metrics)

        real_index_tensor = data.batch["response_mask"].new_tensor(
            real_indices,
            dtype=torch.long,
        )
        if self._latent_enabled():
            real_advantages, real_returns, frequency_metrics = compute_lifpo_advantages(
                token_level_rewards=data.batch["token_level_rewards"].index_select(
                    0, real_index_tensor
                ),
                response_mask=data.batch["response_mask"].index_select(
                    0,
                    real_index_tensor,
                ),
                assigned_log_probs=data.batch["old_log_probs"].index_select(
                    0,
                    real_index_tensor,
                ),
                negative_log_probs=data.batch["latent_negative_log_probs"].index_select(
                    0, real_index_tensor
                ),
                index=uid[real_indices].tolist(),
                metadata=frequency_metadata,
                epsilon=float(
                    _config_value(self._lifpo_config(), "frequency_epsilon", 0.2)
                ),
                counterfactual_alpha=float(
                    _config_value(self._lifpo_config(), "counterfactual_alpha", 0.5)
                ),
                counterfactual_clip=float(
                    _config_value(self._lifpo_config(), "counterfactual_clip", 1.0)
                ),
                counterfactual_token_scope=str(
                    _config_value(
                        self._lifpo_config(),
                        "counterfactual_token_scope",
                        "full_response",
                    )
                ),
                counterfactual_reduction=str(
                    _config_value(
                        self._lifpo_config(), "counterfactual_reduction", "sum"
                    )
                ),
                counterfactual_valid_only=bool(
                    _config_value(
                        self._lifpo_config(), "counterfactual_valid_only", True
                    )
                ),
                inverse_frequency_enabled=bool(
                    _config_value(
                        self._lifpo_config(), "inverse_frequency_enabled", True
                    )
                ),
                latent_count=int(
                    _config_value(self._lifpo_config(), "latent_count", 8)
                ),
                frequency_credit_mode=str(
                    _config_value(
                        self._lifpo_config(), "frequency_credit_mode", "bonus"
                    )
                ),
                frequency_credit_max=float(
                    _config_value(self._lifpo_config(), "frequency_credit_max", 0.5)
                ),
                metric_prefix="lifpo",
            )
        else:
            real_advantages, real_returns, frequency_metrics = (
                compute_inverse_frequency_advantages(
                    token_level_rewards=data.batch["token_level_rewards"].index_select(
                        0, real_index_tensor
                    ),
                    response_mask=data.batch["response_mask"].index_select(
                        0,
                        real_index_tensor,
                    ),
                    index=uid[real_indices].tolist(),
                    metadata=frequency_metadata,
                    epsilon=float(
                        _config_value(self._lifpo_config(), "frequency_epsilon", 0.2)
                    ),
                )
            )
        data.batch["advantages"] = scatter_real_rows(
            data.batch["response_mask"],
            real_index_tensor,
            real_advantages,
        )
        data.batch["returns"] = scatter_real_rows(
            data.batch["response_mask"],
            real_index_tensor,
            real_returns,
        )
        metrics.update(frequency_metrics)
        metrics.update(
            {
                "lifpo/batch_rows": float(len(tags)),
                "lifpo/real_rows": float(len(real_indices)),
                "lifpo/padding_rows": float(len(tags) - len(real_indices)),
                "lifpo/padding_rate": (len(tags) - len(real_indices))
                / max(1, len(tags)),
            }
        )

        output_fields = ["advantages", "returns"]
        if self.config.algorithm.use_kl_in_reward:
            output_fields.append("token_level_rewards")
        if rollout_correction:
            output_fields.append("response_mask")
        if "rollout_is_weights" in data.batch:
            output_fields.append("rollout_is_weights")
        output = {
            field: response_to_nested(
                data.batch[field],
                response_mask_nested,
            )
            for field in output_fields
        }
        return tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=TensorDict(output, batch_size=len(batch)),
        )


def build_lifpo_task_runner() -> Any:
    import ray

    @ray.remote
    class LIFPOTaskRunner:
        def __init__(self) -> None:
            self.config = None
            self.trainer = None
            self.agent_loop_manager = None

        def init_agent_loop_manager(self) -> None:
            self.agent_loop_manager = LIFPOAgentLoopManagerTQ.create(
                config=self.config,
                llm_client=self.trainer.get_llm_client(),
                teacher_client=self.trainer.get_teacher_client(),
                reward_loop_worker_handles=self.trainer.get_reward_handles(),
            )

        def run(self, config: Any) -> None:
            import transfer_queue as tq
            from verl.trainer.ppo.v1 import get_trainer_cls

            if not bool(config.trainer.use_v1):
                raise RuntimeError("LIFPOTaskRunner requires trainer.use_v1=True")
            base_cls = get_trainer_cls(config.trainer.v1.trainer_mode)
            trainer_cls = type(
                f"LIFPO{base_cls.__name__}",
                (LIFPOTrainerMixin, base_cls),
                {},
            )
            print(
                "LIFPO_CUSTOM_TASK_RUNNER_ACTIVE=1 "
                f"trainer_class={trainer_cls.__name__}",
                flush=True,
            )
            config.transfer_queue.enable = True
            self.config = config
            tq.init(config.transfer_queue)
            exit_code = 0
            try:
                self.trainer = trainer_cls(config=config)
                self.trainer.init()
                self.init_agent_loop_manager()
                self.trainer.fit(self.agent_loop_manager)
            except BaseException:
                exit_code = 1
                raise
            finally:
                try:
                    finish_trainer_wandb(self.trainer, exit_code=exit_code)
                finally:
                    tq.close()

    return LIFPOTaskRunner
