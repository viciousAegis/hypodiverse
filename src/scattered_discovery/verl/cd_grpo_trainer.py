from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

from scattered_discovery.envs.causal_micro_lab.consequence_diversity import (
    BehaviorArchive,
    DiversityCandidate,
    diversity_rewards,
)
from scattered_discovery.envs.causal_micro_lab.consequence_reward import (
    CandidateStatus,
)

ARCHIVE_FILENAME = "cd_grpo_archive.json"
CD_PAYLOAD_KEYS = {
    "cd_reward_payload",
    "cd_eval_payload",
    "cd_reward_payload_json",
    "cd_eval_payload_json",
}

try:
    from verl.trainer.ppo.v1.agent_loop_tq import (
        AgentLoopManagerTQ as _AgentLoopManagerTQ,
        AgentLoopWorkerTQ as _AgentLoopWorkerTQ,
    )
except ImportError:  # Keep local verifier/tests importable without veRL.
    _AgentLoopManagerTQ = None
    _AgentLoopWorkerTQ = None


if _AgentLoopWorkerTQ is not None and _AgentLoopManagerTQ is not None:
    import ray

    @ray.remote
    class CDGRPOAgentLoopWorkerTQ(
        _AgentLoopWorkerTQ.__ray_metadata__.modified_class
    ):
        async def _agent_loop_postprocess(
            self,
            output: Any,
            validate: bool,
            **kwargs: Any,
        ) -> None:
            # veRL's TQ adapter calls field.update(kwargs), and RLHFDataset
            # supplies an input extra_fields value. Remove that collision and
            # merge the input metadata explicitly so generated CD payloads
            # survive serialization.
            input_extra_fields = kwargs.pop("extra_fields", None)
            if isinstance(input_extra_fields, dict):
                merged = dict(input_extra_fields)
                merged.update(output.extra_fields)
                output.extra_fields = merged
            return await super()._agent_loop_postprocess(
                output,
                validate,
                **kwargs,
            )


    class CDGRPOAgentLoopManagerTQ(_AgentLoopManagerTQ):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.agent_loop_workers_class = CDGRPOAgentLoopWorkerTQ

else:

    class CDGRPOAgentLoopManagerTQ:
        @classmethod
        def create(cls, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("CDGRPOAgentLoopManagerTQ requires veRL.")


def _config_value(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _normalize_group(values: list[float], active: list[int]) -> list[float]:
    result = [0.0] * len(values)
    if len(active) < 2:
        return result
    selected = [values[index] for index in active]
    mean = sum(selected) / len(selected)
    variance = sum((value - mean) ** 2 for value in selected) / (len(selected) - 1)
    std = math.sqrt(variance)
    denominator = max(std, 1e-4)
    for index in active:
        result[index] = (values[index] - mean) / denominator
    return result


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def scatter_real_rows(
    template: Any,
    real_index_tensor: Any,
    real_values: Any,
) -> Any:
    """Scatter real-row values without inheriting an integer mask dtype."""
    output = real_values.new_zeros(template.shape)
    output.index_copy_(0, real_index_tensor, real_values)
    return output


def sanitize_validation_reward_extras(
    reward_extra_infos: dict[str, list[Any]],
) -> tuple[dict[str, list[Any]], dict[str, float]]:
    """Exclude incomplete auxiliary series before veRL's numeric aggregation."""
    sanitized: dict[str, list[Any]] = {}
    missing_rates: dict[str, float] = {}
    for key, values in reward_extra_infos.items():
        if key in CD_PAYLOAD_KEYS:
            continue
        missing = sum(value is None for value in values)
        if not missing:
            sanitized[key] = values
            continue
        if key == "reward":
            raise ValueError("validation reward series contains missing values")
        missing_rates[key] = missing / max(1, len(values))
    return sanitized, missing_rates


def _decode_cd_payload(
    value: Any,
    *,
    field_name: str,
    index: int,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} at rollout index {index} must be a mapping or JSON string"
        )
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{field_name} at rollout index {index} is invalid JSON"
        ) from error
    if not isinstance(decoded, dict):
        raise ValueError(
            f"{field_name} at rollout index {index} must decode to an object"
        )
    return decoded


def _payload_from_sources(
    item: dict[str, Any],
    reward_extra_info: dict[str, Any],
    *,
    field_name: str,
    index: int,
) -> dict[str, Any] | None:
    value = item.get(field_name)
    if value is None:
        value = reward_extra_info.get(field_name)
    if value is not None:
        return _decode_cd_payload(
            value,
            field_name=field_name,
            index=index,
        )

    json_field = f"{field_name}_json"
    value = item.get(json_field)
    if value is None:
        value = reward_extra_info.get(json_field)
    return _decode_cd_payload(
        value,
        field_name=json_field,
        index=index,
    )


def select_cd_payloads(
    extra_fields: list[dict[str, Any]],
    tags: list[dict[str, Any]],
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    """Select real rollout payloads while excluding veRL's synthetic padding."""
    if len(extra_fields) != len(tags):
        raise ValueError("extra_fields and tags must have equal length")

    real_indices: list[int] = []
    payloads: list[dict[str, Any]] = []
    eval_payloads: list[dict[str, Any]] = []
    for index, (item, tag) in enumerate(zip(extra_fields, tags, strict=True)):
        if bool(tag.get("is_padding", False)):
            continue
        reward_extra_info = item.get("reward_extra_info", {})
        if not isinstance(reward_extra_info, dict):
            reward_extra_info = {}
        payload = _payload_from_sources(
            item,
            reward_extra_info,
            field_name="cd_reward_payload",
            index=index,
        )
        if payload is None:
            diagnostic_keys = {
                "validity",
                "parse_valid",
                "response_length_cap_hit",
            }
            if diagnostic_keys.isdisjoint(reward_extra_info):
                raise KeyError(
                    f"real rollout at index {index} has no cd_reward_payload"
                )
            if float(reward_extra_info.get("validity", 0.0)) > 0.0:
                raise KeyError(
                    f"valid rollout at index {index} has no cd_reward_payload"
                )
            if float(reward_extra_info.get("response_length_cap_hit", 0.0)) > 0.0:
                status = CandidateStatus.TRUNCATED.value
            elif float(reward_extra_info.get("parse_valid", 0.0)) <= 0.0:
                status = CandidateStatus.PARSE_FAIL.value
            else:
                status = CandidateStatus.INVALID.value
            payload = {
                "status": status,
                "state_id": f"nonvalid-rollout-{index}",
                "consequence_signature": None,
                "behavior_key": None,
            }
        eval_payload = _payload_from_sources(
            item,
            reward_extra_info,
            field_name="cd_eval_payload",
            index=index,
        )
        if eval_payload is None:
            eval_payload = {
                "valid_mode_count": int(
                    reward_extra_info.get("valid_mode_count", 0)
                ),
                "separation_bucket": "unknown",
                "family_bucket": "unknown",
            }
        real_indices.append(index)
        payloads.append(payload)
        eval_payloads.append(eval_payload or {})

    if not real_indices:
        raise ValueError("CD-GRPO batch contains no real rollouts")
    return real_indices, payloads, eval_payloads


def _pairwise_distances(signatures: list[str]) -> list[float]:
    distances: list[float] = []
    for left_index, left in enumerate(signatures):
        for right in signatures[left_index + 1 :]:
            if len(left) != len(right):
                raise ValueError("consequence signatures must have equal length")
            distances.append(
                sum(a != b for a, b in zip(left, right, strict=True))
                / max(1, len(left))
            )
    return distances


def compute_cd_grpo_advantages(
    *,
    token_level_rewards: Any,
    response_mask: Any,
    index: list[object],
    payloads: list[dict[str, Any]],
    eval_payloads: list[dict[str, Any]] | None,
    config: Any,
    archive: BehaviorArchive,
) -> tuple[Any, Any, dict[str, float]]:
    import torch

    scores = token_level_rewards.sum(dim=-1)
    beta = float(_config_value(config, "beta", 0.3))
    beta_guard = bool(_config_value(config, "beta_guard", True))
    beta_guard_window = int(_config_value(config, "beta_guard_window", 50))
    variant = str(_config_value(config, "variant", "logdet"))
    ell = float(_config_value(config, "ell", 0.25))
    archive_enabled = bool(_config_value(config, "archive", True))

    groups: dict[object, list[int]] = defaultdict(list)
    for item_index, group_id in enumerate(index):
        groups[group_id].append(item_index)

    validity_rate = sum(
        payload.get("status") == CandidateStatus.VALID.value for payload in payloads
    ) / max(1, len(payloads))
    running_validity = archive.update_validity(
        validity_rate,
        window=max(1, beta_guard_window),
    )
    if (
        beta_guard
        and archive.max_running_validity - running_validity > 0.1
        and archive.beta_multiplier == 1.0
    ):
        archive.beta_multiplier = 0.5
    effective_beta = beta * archive.beta_multiplier

    advantage_scalars = torch.zeros_like(scores)
    diversity_raw_all: list[float] = []
    diversity_advantages_all: list[float] = []
    validity_advantages_all: list[float] = []
    pairwise_distances_all: list[float] = []
    unique_pairwise_distances_all: list[float] = []
    diagnostics_all = []
    eval_rows: list[dict[str, Any]] = []
    valid_candidates_for_archive: list[DiversityCandidate] = []
    groups_with_zero_valid = 0
    groups_with_one_valid = 0
    groups_with_two_plus_valid = 0
    groups_with_two_plus_unique = 0
    groups_with_nonzero_diversity = 0
    groups_all_truncated = 0
    archive_new_valid_completions = 0
    archive_new_unique_behaviors = 0
    archive_valid_completions = 0
    archive_unique_behaviors = 0
    for group_indices in groups.values():
        group_scores = scores[group_indices]
        if len(group_indices) == 1:
            validity_advantages = torch.zeros_like(group_scores)
        else:
            validity_advantages = (group_scores - group_scores.mean()) / (
                group_scores.std(unbiased=True) + 1e-6
            )

        candidates = [
            DiversityCandidate(
                state_id=str(payloads[item_index]["state_id"]),
                status=CandidateStatus(str(payloads[item_index]["status"])),
                consequence_signature=payloads[item_index].get("consequence_signature"),
                behavior_key=payloads[item_index].get("behavior_key"),
            )
            for item_index in group_indices
        ]
        raw_rewards, diagnostics = diversity_rewards(
            candidates,
            variant=variant,  # type: ignore[arg-type]
            ell=ell,
            archive=archive if archive_enabled else None,
            update_archive=False,
        )
        active = [
            local_index
            for local_index, candidate in enumerate(candidates)
            if candidate.valid
        ]
        normalized = _normalize_group(raw_rewards, active)
        valid_count = len(active)
        unique_valid_keys = {
            candidates[local_index].behavior_key or "" for local_index in active
        }
        if valid_count == 0:
            groups_with_zero_valid += 1
        elif valid_count == 1:
            groups_with_one_valid += 1
        else:
            groups_with_two_plus_valid += 1
        if len(unique_valid_keys) >= 2:
            groups_with_two_plus_unique += 1
        if any(abs(normalized[local_index]) > 1e-8 for local_index in active):
            groups_with_nonzero_diversity += 1
        if candidates and all(
            candidate.status is CandidateStatus.TRUNCATED for candidate in candidates
        ):
            groups_all_truncated += 1

        valid_signatures = [
            candidates[local_index].consequence_signature or ""
            for local_index in active
        ]
        pairwise_distances_all.extend(_pairwise_distances(valid_signatures))
        unique_signatures = sorted(set(valid_signatures))
        unique_pairwise_distances_all.extend(_pairwise_distances(unique_signatures))

        if archive_enabled:
            archive_valid_completions += valid_count
            archive_unique_behaviors += len(unique_valid_keys)
            archive_new_valid_completions += sum(
                archive.count(
                    candidates[local_index].state_id,
                    candidates[local_index].behavior_key or "",
                )
                == 0.0
                for local_index in active
            )
            archive_new_unique_behaviors += (
                sum(
                    archive.count(
                        candidates[active[0]].state_id,
                        behavior_key,
                    )
                    == 0.0
                    for behavior_key in unique_valid_keys
                )
                if active
                else 0
            )

        for local_index, item_index in enumerate(group_indices):
            validity_value = float(validity_advantages[local_index].item())
            diversity_value = float(normalized[local_index])
            advantage_scalars[item_index] = (
                validity_value + effective_beta * diversity_value
            )
            validity_advantages_all.append(validity_value)
            diversity_advantages_all.append(diversity_value)
        diversity_raw_all.extend(raw_rewards[local_index] for local_index in active)
        valid_candidates_for_archive.extend(
            candidate for candidate in candidates if candidate.valid
        )
        diagnostics_all.append(diagnostics)
        if eval_payloads:
            metadata = eval_payloads[group_indices[0]]
            valid_keys = [
                candidate.behavior_key for candidate in candidates if candidate.valid
            ]
            counts: dict[str, int] = defaultdict(int)
            for key in valid_keys:
                counts[key or ""] += 1
            valid_count = len(valid_keys)
            probabilities = (
                [count / valid_count for count in counts.values()]
                if valid_count
                else []
            )
            entropy = -sum(
                probability * math.log(probability)
                for probability in probabilities
                if probability > 0.0
            )
            available = int(metadata.get("valid_mode_count", 0))
            eval_rows.append(
                {
                    "coverage": len(counts) / available if available else 0.0,
                    "dominant_mode_mass": max(probabilities, default=0.0),
                    "effective_mode_count": math.exp(entropy) if probabilities else 0.0,
                    "M": available,
                    "separation_bucket": str(
                        metadata.get("separation_bucket", "unknown")
                    ),
                }
            )

    if archive_enabled:
        for candidate in valid_candidates_for_archive:
            archive.add(candidate.state_id, candidate.behavior_key or "")

    advantages = advantage_scalars.unsqueeze(-1) * response_mask
    unique_behaviors = sum(item.unique_behaviors for item in diagnostics_all)
    valid_completions = sum(item.valid_completions for item in diagnostics_all)
    duplicate_completions = sum(
        item.duplicate_valid_completions for item in diagnostics_all
    )
    group_count = len(diagnostics_all)
    diversity_contributions = [
        effective_beta * value for value in diversity_advantages_all
    ]
    metrics = {
        "cd_grpo/beta": effective_beta,
        "cd_grpo/validity_rate": validity_rate,
        "cd_grpo/running_validity_rate": running_validity,
        "cd_grpo/valid_completions_per_group": valid_completions / max(1, group_count),
        "cd_grpo/unique_behaviors_per_group": unique_behaviors / max(1, group_count),
        "cd_grpo/duplicate_valid_rate": duplicate_completions
        / max(1, valid_completions),
        "cd_grpo/groups_skipped_rate": sum(item.skipped for item in diagnostics_all)
        / max(1, group_count),
        "cd_grpo/groups_with_0_valid_rate": groups_with_zero_valid
        / max(1, group_count),
        "cd_grpo/groups_with_1_valid_rate": groups_with_one_valid / max(1, group_count),
        "cd_grpo/groups_with_2plus_valid_rate": groups_with_two_plus_valid
        / max(1, group_count),
        "cd_grpo/groups_with_2plus_unique_valid_rate": groups_with_two_plus_unique
        / max(1, group_count),
        "cd_grpo/diversity_signal_active_rate": groups_with_nonzero_diversity
        / max(1, group_count),
        "cd_grpo/all_truncated_group_rate": groups_all_truncated / max(1, group_count),
        "cd_grpo/pairwise_consequence_distance_mean": _mean(pairwise_distances_all),
        "cd_grpo/unique_pairwise_consequence_distance_mean": _mean(
            unique_pairwise_distances_all
        ),
        "cd_grpo/diversity_raw_mean": _mean(diversity_raw_all),
        "cd_grpo/validity_advantage_abs_mean": _mean(
            [abs(value) for value in validity_advantages_all]
        ),
        "cd_grpo/validity_advantage_std": _population_std(validity_advantages_all),
        "cd_grpo/diversity_advantage_abs_mean": _mean(
            [abs(value) for value in diversity_advantages_all]
        ),
        "cd_grpo/diversity_advantage_std": _population_std(diversity_advantages_all),
        "cd_grpo/diversity_contribution_abs_mean": _mean(
            [abs(value) for value in diversity_contributions]
        ),
        "cd_grpo/archive_size": float(len(archive.counts)) if archive_enabled else 0.0,
        "cd_grpo/archive_new_valid_completion_rate": (
            archive_new_valid_completions / max(1, archive_valid_completions)
            if archive_enabled
            else 0.0
        ),
        "cd_grpo/archive_new_unique_behavior_rate": (
            archive_new_unique_behaviors / max(1, archive_unique_behaviors)
            if archive_enabled
            else 0.0
        ),
        "cd_grpo/archive_scale_mean": sum(
            item.mean_archive_scale for item in diagnostics_all
        )
        / max(1, group_count),
    }
    if eval_rows:
        for metric_name in (
            "coverage",
            "dominant_mode_mass",
            "effective_mode_count",
        ):
            metrics[f"cd_grpo/{metric_name}"] = sum(
                float(row[metric_name]) for row in eval_rows
            ) / len(eval_rows)
        for field, prefix in (
            ("M", "M"),
            ("separation_bucket", "separation"),
        ):
            labels = sorted({str(row[field]) for row in eval_rows})
            for label in labels:
                selected = [row for row in eval_rows if str(row[field]) == label]
                metrics[f"cd_grpo/{prefix}_{label}/groups"] = float(len(selected))
                metrics[f"cd_grpo/{prefix}_{label}/coverage"] = sum(
                    float(row["coverage"]) for row in selected
                ) / len(selected)
    return advantages, advantages, metrics


class CDGRPOTrainerMixin:
    """veRL v1 trainer extension with checkpointed consequence diversity."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.cd_grpo_archive = BehaviorArchive()
        super().__init__(*args, **kwargs)

    def _cd_config(self) -> Any:
        return self.config.algorithm.get("cd_grpo", {})

    def on_step_begin(self) -> None:
        super().on_step_begin()
        if not hasattr(self, "steps_per_epoch"):
            return
        epoch = max(0, (self.global_steps - 1) // max(1, self.steps_per_epoch))
        if epoch > self.cd_grpo_archive.last_epoch:
            gamma = float(_config_value(self._cd_config(), "gamma", 0.7))
            self.cd_grpo_archive.decay(gamma)
            self.cd_grpo_archive.last_epoch = epoch

    def _load_checkpoint(self) -> None:
        super()._load_checkpoint()
        if not getattr(self, "global_steps", 0):
            return
        path = (
            Path(self.config.trainer.default_local_dir)
            / f"global_step_{self.global_steps}"
            / ARCHIVE_FILENAME
        )
        if path.exists():
            self.cd_grpo_archive = BehaviorArchive.load(path)

    def _save_checkpoint(self) -> Any:
        result = super()._save_checkpoint()
        path = (
            Path(self.config.trainer.default_local_dir)
            / f"global_step_{self.global_steps}"
            / ARCHIVE_FILENAME
        )
        self.cd_grpo_archive.save(path)
        return result

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
            metrics[f"val-aux/cd_grpo/missing_reward_extra/{safe_key}"] = rate
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
        real_indices, payloads, eval_payloads = select_cd_payloads(
            extra_fields,
            tags,
        )
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

        rollout_corr_config = self.config.algorithm.get(
            "rollout_correction",
            None,
        )
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
        real_advantages, real_returns, cd_metrics = compute_cd_grpo_advantages(
            token_level_rewards=data.batch["token_level_rewards"].index_select(
                0,
                real_index_tensor,
            ),
            response_mask=data.batch["response_mask"].index_select(
                0,
                real_index_tensor,
            ),
            index=uid[real_indices].tolist(),
            payloads=payloads,
            eval_payloads=eval_payloads,
            config=self._cd_config(),
            archive=self.cd_grpo_archive,
        )
        advantages = scatter_real_rows(
            data.batch["response_mask"],
            real_index_tensor,
            real_advantages,
        )
        returns = scatter_real_rows(
            data.batch["response_mask"],
            real_index_tensor,
            real_returns,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        metrics.update(cd_metrics)
        metrics.update(
            {
                "cd_grpo/batch_rows": float(len(tags)),
                "cd_grpo/real_rows": float(len(real_indices)),
                "cd_grpo/padding_rows": float(len(tags) - len(real_indices)),
                "cd_grpo/padding_rate": (len(tags) - len(real_indices))
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


def build_cd_task_runner() -> Any:
    import ray

    @ray.remote
    class CDGRPOTaskRunner:
        def __init__(self) -> None:
            self.config = None
            self.trainer = None
            self.agent_loop_manager = None

        def init_agent_loop_manager(self) -> None:
            from verl.trainer.ppo.v1 import AgentLoopManagerTQ
            from verl.utils.import_utils import load_class_from_fqn

            manager_fqn = self.config.actor_rollout_ref.rollout.get(
                "agent",
                {},
            ).get("agent_loop_manager_class")
            manager_cls = (
                load_class_from_fqn(manager_fqn, "AgentLoopManager")
                if manager_fqn
                else AgentLoopManagerTQ
            )
            self.agent_loop_manager = manager_cls.create(
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
                f"CDGRPO{base_cls.__name__}",
                (CDGRPOTrainerMixin, base_cls),
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

    return CDGRPOTaskRunner
