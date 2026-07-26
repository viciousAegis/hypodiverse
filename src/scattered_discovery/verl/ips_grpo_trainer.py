from __future__ import annotations

from collections import Counter, defaultdict
import math
import time
from typing import Any


IPS_METADATA_KEYS = {
    "ips_behavior_hash_hi",
    "ips_behavior_hash_lo",
    "cd_reward_payload",
    "cd_eval_payload",
    "cd_reward_payload_json",
    "cd_eval_payload_json",
}

IPS_REWARD_EXTRA_DEFAULTS = {
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
    "cd_probe_count": 0.0,
    "ips_behavior_hash_hi": -1.0,
    "ips_behavior_hash_lo": -1.0,
    "valid_mode_count": 0.0,
    "latent_enabled": 0.0,
    "latent_id": 0.0,
    "latent_negative_id": 0.0,
    "latent_answer_token_count": 0.0,
}


def normalize_ips_reward_extra_info(
    value: Any,
    *,
    reward_score: Any = 0.0,
) -> dict[str, float]:
    """Return the fixed scalar schema required by veRL batch postprocessing."""
    source = value if isinstance(value, dict) else {}
    defaults = dict(IPS_REWARD_EXTRA_DEFAULTS)
    defaults["terminal_reward"] = float(reward_score or 0.0)
    defaults["base_terminal_reward"] = float(reward_score or 0.0)
    return {key: float(source.get(key, default)) for key, default in defaults.items()}


try:
    from verl.trainer.ppo.v1.agent_loop_tq import (
        AgentLoopManagerTQ as _AgentLoopManagerTQ,
        AgentLoopWorkerTQ as _AgentLoopWorkerTQ,
    )
except ImportError:  # Keep local tests importable without veRL.
    _AgentLoopManagerTQ = None
    _AgentLoopWorkerTQ = None


if _AgentLoopWorkerTQ is not None and _AgentLoopManagerTQ is not None:
    import ray

    @ray.remote
    class IPSGRPOAgentLoopWorkerTQ(_AgentLoopWorkerTQ.__ray_metadata__.modified_class):
        async def _agent_loop_postprocess(
            self,
            output: Any,
            validate: bool,
            **kwargs: Any,
        ) -> None:
            input_extra_fields = kwargs.pop("extra_fields", None)
            if isinstance(input_extra_fields, dict):
                merged = dict(input_extra_fields)
                merged.update(output.extra_fields)
                output.extra_fields = merged
            # veRL derives the batch schema from the first rollout and indexes
            # every later rollout with those keys. Normalize failure/padding
            # paths before that unguarded batch assembly.
            output.extra_fields["reward_extra_info"] = normalize_ips_reward_extra_info(
                output.extra_fields.get("reward_extra_info"),
                reward_score=getattr(output, "reward_score", 0.0),
            )
            return await super()._agent_loop_postprocess(
                output,
                validate,
                **kwargs,
            )

    class IPSGRPOAgentLoopManagerTQ(_AgentLoopManagerTQ):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.agent_loop_workers_class = IPSGRPOAgentLoopWorkerTQ

else:

    class IPSGRPOAgentLoopManagerTQ:
        @classmethod
        def create(cls, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("IPSGRPOAgentLoopManagerTQ requires veRL.")


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
        if key in IPS_METADATA_KEYS:
            continue
        missing = sum(value is None for value in values)
        if not missing:
            sanitized[key] = values
            continue
        if key == "reward":
            raise ValueError("validation reward series contains missing values")
        missing_rates[key] = missing / max(1, len(values))
    return sanitized, missing_rates


def select_ips_metadata(
    extra_fields: list[dict[str, Any]],
    tags: list[dict[str, Any]],
) -> tuple[list[int], list[dict[str, Any]]]:
    """Extract numeric IPS metadata while excluding veRL padding rows."""
    if len(extra_fields) != len(tags):
        raise ValueError("extra_fields and tags must have equal length")

    real_indices: list[int] = []
    metadata: list[dict[str, Any]] = []
    for index, (item, tag) in enumerate(zip(extra_fields, tags, strict=True)):
        if bool(tag.get("is_padding", False)):
            continue
        reward_info = item.get("reward_extra_info", {})
        if not isinstance(reward_info, dict):
            raise TypeError(
                f"reward_extra_info at rollout index {index} must be a mapping"
            )
        required = {
            "validity",
            "reward_valid_hypothesis",
            "ips_behavior_hash_hi",
            "ips_behavior_hash_lo",
        }
        missing = sorted(required - set(reward_info))
        if missing:
            raise KeyError(
                f"real rollout at index {index} is missing IPS metadata: "
                + ", ".join(missing)
            )

        valid = _as_float(reward_info["validity"], field="validity", index=index) > 0
        valid_reward = _as_float(
            reward_info["reward_valid_hypothesis"],
            field="reward_valid_hypothesis",
            index=index,
        )
        hash_hi = int(
            _as_float(
                reward_info["ips_behavior_hash_hi"],
                field="ips_behavior_hash_hi",
                index=index,
            )
        )
        hash_lo = int(
            _as_float(
                reward_info["ips_behavior_hash_lo"],
                field="ips_behavior_hash_lo",
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
        raise ValueError("IPS-GRPO batch contains no real rollouts")
    return real_indices, metadata


def compute_ips_grpo_advantages(
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
        raise ValueError("IPS epsilon must be in (0, 1]")
    if token_level_rewards.shape != response_mask.shape:
        raise ValueError("token_level_rewards and response_mask must match")
    if len(index) != token_level_rewards.shape[0] or len(metadata) != len(index):
        raise ValueError("index, metadata, and reward rows must have equal length")

    raw_scores = token_level_rewards.sum(dim=-1)
    ips_scores = raw_scores.clone()
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
            ips_scores[row_index] = (
                raw_scores[row_index] - validity_reward + validity_reward * weight
            )
            valid_count += 1
            singleton_valid_count += int(count == 1)
            clipped_valid_count += int(probability < epsilon)
            weights.append(weight)
            probabilities.append(probability)

        group_scores = ips_scores[group_indices]
        if len(group_indices) > 1:
            group_advantages = (group_scores - group_scores.mean()) / (
                group_scores.std(unbiased=True) + 1e-6
            )
            advantage_scalars[group_indices] = group_advantages

    advantages = advantage_scalars.unsqueeze(-1) * response_mask
    group_count = len(groups)
    total_rows = len(metadata)
    metrics = {
        "ips_grpo/epsilon": float(epsilon),
        "ips_grpo/validity_rate": valid_count / max(1, total_rows),
        "ips_grpo/raw_score_mean": float(raw_scores.mean().item()),
        "ips_grpo/scaled_score_mean": float(ips_scores.mean().item()),
        "ips_grpo/scaled_score_max": float(ips_scores.max().item()),
        "ips_grpo/weight_mean": sum(weights) / max(1, len(weights)),
        "ips_grpo/weight_max": max(weights, default=0.0),
        "ips_grpo/empirical_probability_mean": sum(probabilities)
        / max(1, len(probabilities)),
        "ips_grpo/clipped_valid_rate": clipped_valid_count / max(1, valid_count),
        "ips_grpo/singleton_valid_rate": singleton_valid_count / max(1, valid_count),
        "ips_grpo/duplicate_valid_rate": duplicate_valid_count / max(1, valid_count),
        "ips_grpo/unique_valid_outcomes_per_group": unique_outcomes_total
        / max(1, group_count),
        "ips_grpo/groups_without_valid_rate": groups_without_valid
        / max(1, group_count),
        "ips_grpo/groups_with_multiple_valid_outcomes_rate": (
            groups_with_multiple_outcomes / max(1, group_count)
        ),
        "ips_grpo/advantage_abs_mean": float(advantage_scalars.abs().mean().item()),
        "ips_grpo/advantage_std": float(advantage_scalars.std(unbiased=False).item()),
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


def compute_latent_ips_grpo_advantages(
    *,
    token_level_rewards: Any,
    response_mask: Any,
    assigned_log_probs: Any,
    negative_log_probs: Any,
    index: list[object],
    metadata: list[dict[str, Any]],
    epsilon: float,
    mi_alpha: float,
    mi_clip: float,
    use_ips: bool,
    latent_count: int,
) -> tuple[Any, Any, dict[str, float]]:
    """Combine validity, latent specificity, and optional empirical IPS."""
    import torch

    if not 0.0 < epsilon <= 1.0:
        raise ValueError("IPS epsilon must be in (0, 1]")
    if mi_alpha < 0.0:
        raise ValueError("latent MI alpha must be non-negative")
    if mi_clip <= 0.0:
        raise ValueError("latent MI clip must be positive")
    if latent_count < 2:
        raise ValueError("latent_count must be at least two")
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

    mi_raw_values: list[float] = []
    mi_clipped_values: list[float] = []
    mi_contributions: list[float] = []
    assigned_answer_logps: list[float] = []
    negative_answer_logps: list[float] = []
    weights: list[float] = []
    probabilities: list[float] = []
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
    row_mi: list[float | None] = [None] * len(metadata)
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
            if not item["valid"]:
                continue
            valid_count += 1
            count = outcome_counts[item["behavior"]]
            probability = count / group_size
            denominator = max(probability, epsilon)
            weight = 1.0 / denominator if use_ips else 1.0
            singleton_valid_count += int(count == 1)
            clipped_weight_count += int(use_ips and probability < epsilon)
            weights.append(weight)
            probabilities.append(probability)
            row_weights[row_index] = weight

            active = torch.nonzero(
                response_mask[row_index] > 0,
                as_tuple=False,
            ).flatten()
            answer_count = min(int(item["answer_token_count"]), len(active))
            mi_raw = 0.0
            assigned_mean = 0.0
            negative_mean = 0.0
            if answer_count > 0:
                answer_indices = active[-answer_count:]
                assigned = assigned_log_probs[row_index, answer_indices]
                negative = negative_log_probs[row_index, answer_indices]
                mi_raw = float(
                    _contrastive_token_scores(assigned, negative).mean().item()
                )
                assigned_mean = float(assigned.mean().item())
                negative_mean = float(negative.mean().item())
            mi_clipped = max(-mi_clip, min(mi_clip, mi_raw))
            contribution = mi_alpha * mi_clipped
            validity_reward = float(item["valid_reward"])
            shaped_scores[row_index] = (
                raw_scores[row_index]
                - validity_reward
                + validity_reward * weight
                + contribution
            )
            mi_raw_values.append(mi_raw)
            mi_clipped_values.append(mi_clipped)
            mi_contributions.append(contribution)
            assigned_answer_logps.append(assigned_mean)
            negative_answer_logps.append(negative_mean)
            row_mi[row_index] = mi_raw

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

    mi_tensor = torch.tensor(mi_raw_values, dtype=torch.float32)
    metrics = {
        "latent_ips/use_ips": float(use_ips),
        "latent_ips/epsilon": float(epsilon),
        "latent_ips/mi_alpha": float(mi_alpha),
        "latent_ips/mi_clip": float(mi_clip),
        "latent_ips/validity_rate": valid_count / max(1, total_rows),
        "latent_ips/raw_score_mean": float(raw_scores.mean().item()),
        "latent_ips/shaped_score_mean": float(shaped_scores.mean().item()),
        "latent_ips/shaped_score_max": float(shaped_scores.max().item()),
        "latent_ips/mi_raw_mean": mean(mi_raw_values),
        "latent_ips/mi_raw_std": (
            float(mi_tensor.std(unbiased=False).item()) if mi_raw_values else 0.0
        ),
        "latent_ips/mi_clipped_mean": mean(mi_clipped_values),
        "latent_ips/mi_reward_mean": mean(mi_contributions),
        "latent_ips/mi_nonzero_rate": sum(abs(value) > 1e-8 for value in mi_raw_values)
        / max(1, len(mi_raw_values)),
        "latent_ips/assigned_answer_logp_mean": mean(assigned_answer_logps),
        "latent_ips/negative_answer_logp_mean": mean(negative_answer_logps),
        "latent_ips/answer_logp_margin_mean": mean(
            [
                assigned - negative
                for assigned, negative in zip(
                    assigned_answer_logps,
                    negative_answer_logps,
                    strict=True,
                )
            ]
        ),
        "latent_ips/answer_tokens_mean": mean(
            [float(item["answer_token_count"]) for item in metadata]
        ),
        "latent_ips/valid_answer_tokens_mean": mean(
            [float(item["answer_token_count"]) for item in metadata if item["valid"]]
        ),
        "latent_ips/weight_mean": mean(weights),
        "latent_ips/weight_max": max(weights, default=0.0),
        "latent_ips/empirical_probability_mean": mean(probabilities),
        "latent_ips/clipped_valid_rate": clipped_weight_count / max(1, valid_count),
        "latent_ips/singleton_valid_rate": singleton_valid_count / max(1, valid_count),
        "latent_ips/duplicate_valid_rate": duplicate_valid_count / max(1, valid_count),
        "latent_ips/unique_valid_outcomes_per_group": unique_outcomes_total
        / max(1, group_count),
        "latent_ips/groups_without_valid_rate": groups_without_valid
        / max(1, group_count),
        "latent_ips/groups_with_multiple_valid_outcomes_rate": (
            groups_with_multiple_outcomes / max(1, group_count)
        ),
        "latent_ips/groups_all_latents_present_rate": groups_all_latents_present
        / max(1, group_count),
        "latent_ips/cross_latent_outcome_collision_rate": (
            same_outcome_cross_latent_pairs / max(1, valid_cross_latent_pairs)
        ),
        "latent_ips/advantage_abs_mean": float(advantage_scalars.abs().mean().item()),
        "latent_ips/advantage_std": float(advantage_scalars.std(unbiased=False).item()),
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
        metrics[f"latent_ips/latent_{latent_id}/rows"] = float(len(latent_rows))
        metrics[f"latent_ips/latent_{latent_id}/validity_rate"] = sum(
            metadata[row_index]["valid"] for row_index in latent_rows
        ) / max(1, len(latent_rows))
        valid_latent_rows = [
            row_index for row_index in latent_rows if metadata[row_index]["valid"]
        ]
        metrics[f"latent_ips/latent_{latent_id}/unique_valid_outcomes"] = float(
            len({metadata[row_index]["behavior"] for row_index in valid_latent_rows})
        )
        metrics[f"latent_ips/latent_{latent_id}/mi_raw_mean"] = mean(
            [
                float(row_mi[row_index])
                for row_index in valid_latent_rows
                if row_mi[row_index] is not None
            ]
        )
        metrics[f"latent_ips/latent_{latent_id}/weight_mean"] = mean(
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
        metrics[f"latent_ips/M_{mode_count}/rows"] = float(len(mode_rows))
        metrics[f"latent_ips/M_{mode_count}/validity_rate"] = sum(
            metadata[row_index]["valid"] for row_index in mode_rows
        ) / max(1, len(mode_rows))
        metrics[f"latent_ips/M_{mode_count}/unique_valid_outcomes_per_group"] = mean(
            [float(value) for value in unique_by_group]
        )
    return advantages, advantages, metrics


class IPSGRPOTrainerMixin:
    """veRL v1 trainer extension implementing empirical IPS-GRPO."""

    def _ips_config(self) -> Any:
        return self.config.algorithm.get("ips_grpo", {})

    def _latent_enabled(self) -> bool:
        return bool(_config_value(self._ips_config(), "latent_enabled", False))

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
        extra_rows = list(source["extra_fields"])
        if not (
            len(prompt_rows) == len(response_rows) == len(extra_rows) == len(batch)
        ):
            raise ValueError("counterfactual scorer received misaligned TQ rows")

        counterfactual_inputs: list[Any] = []
        counterfactual_positions: list[Any] = []
        scored_tokens = 0
        for row_index, (prompt_row, response_row, extra) in enumerate(
            zip(prompt_rows, response_rows, extra_rows, strict=True)
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
            full_ids = negative_prompt_ids + response_ids
            counterfactual_inputs.append(torch.tensor(full_ids, dtype=torch.long))
            counterfactual_positions.append(
                torch.arange(len(full_ids), dtype=torch.long)
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
                "latent_ips/counterfactual_scored_tokens": float(scored_tokens),
                "latent_ips/counterfactual_score_seconds": elapsed,
                "latent_ips/counterfactual_tokens_per_second": (
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
            metrics[f"val-aux/ips_grpo/missing_reward_extra/{safe_key}"] = rate
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
        extra_fields = list(raw.pop("extra_fields"))
        tags = list(batch.tags)
        real_indices, ips_metadata = select_ips_metadata(extra_fields, tags)
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
            real_advantages, real_returns, ips_metrics = (
                compute_latent_ips_grpo_advantages(
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
                    negative_log_probs=data.batch[
                        "latent_negative_log_probs"
                    ].index_select(0, real_index_tensor),
                    index=uid[real_indices].tolist(),
                    metadata=ips_metadata,
                    epsilon=float(_config_value(self._ips_config(), "epsilon", 0.2)),
                    mi_alpha=float(
                        _config_value(self._ips_config(), "latent_mi_alpha", 0.1)
                    ),
                    mi_clip=float(
                        _config_value(self._ips_config(), "latent_mi_clip", 1.0)
                    ),
                    use_ips=bool(
                        _config_value(self._ips_config(), "latent_use_ips", True)
                    ),
                    latent_count=int(
                        _config_value(self._ips_config(), "latent_count", 8)
                    ),
                )
            )
        else:
            real_advantages, real_returns, ips_metrics = compute_ips_grpo_advantages(
                token_level_rewards=data.batch["token_level_rewards"].index_select(
                    0, real_index_tensor
                ),
                response_mask=data.batch["response_mask"].index_select(
                    0,
                    real_index_tensor,
                ),
                index=uid[real_indices].tolist(),
                metadata=ips_metadata,
                epsilon=float(_config_value(self._ips_config(), "epsilon", 0.2)),
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
        metrics.update(ips_metrics)
        metrics.update(
            {
                "ips_grpo/batch_rows": float(len(tags)),
                "ips_grpo/real_rows": float(len(real_indices)),
                "ips_grpo/padding_rows": float(len(tags) - len(real_indices)),
                "ips_grpo/padding_rate": (len(tags) - len(real_indices))
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


def build_ips_task_runner() -> Any:
    import ray

    @ray.remote
    class IPSGRPOTaskRunner:
        def __init__(self) -> None:
            self.config = None
            self.trainer = None
            self.agent_loop_manager = None

        def init_agent_loop_manager(self) -> None:
            self.agent_loop_manager = IPSGRPOAgentLoopManagerTQ.create(
                config=self.config,
                llm_client=self.trainer.get_llm_client(),
                teacher_client=self.trainer.get_teacher_client(),
                reward_loop_worker_handles=self.trainer.get_reward_handles(),
            )

        def run(self, config: Any) -> None:
            import transfer_queue as tq
            from verl.trainer.ppo.v1 import get_trainer_cls

            base_cls = get_trainer_cls(config.trainer.v1.trainer_mode)
            trainer_cls = type(
                f"IPSGRPO{base_cls.__name__}",
                (IPSGRPOTrainerMixin, base_cls),
                {},
            )
            config.transfer_queue.enable = True
            self.config = config
            tq.init(config.transfer_queue)
            try:
                self.trainer = trainer_cls(config=config)
                self.trainer.init()
                self.init_agent_loop_manager()
                self.trainer.fit(self.agent_loop_manager)
            finally:
                tq.close()

    return IPSGRPOTaskRunner
