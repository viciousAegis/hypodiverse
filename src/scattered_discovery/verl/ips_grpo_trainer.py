from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


IPS_METADATA_KEYS = {
    "ips_behavior_hash_hi",
    "ips_behavior_hash_lo",
    "cd_reward_payload",
    "cd_eval_payload",
    "cd_reward_payload_json",
    "cd_eval_payload_json",
}


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


class IPSGRPOTrainerMixin:
    """veRL v1 trainer extension implementing empirical IPS-GRPO."""

    def _ips_config(self) -> Any:
        return self.config.algorithm.get("ips_grpo", {})

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
        raw = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=fields,
        )
        response_mask_nested = raw["response_mask"]
        extra_fields = raw.pop("extra_fields").tolist()
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
        real_advantages, real_returns, ips_metrics = compute_ips_grpo_advantages(
            token_level_rewards=data.batch["token_level_rewards"].index_select(
                0,
                real_index_tensor,
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
            from verl.trainer.ppo.v1 import AgentLoopManagerTQ

            self.agent_loop_manager = AgentLoopManagerTQ.create(
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
