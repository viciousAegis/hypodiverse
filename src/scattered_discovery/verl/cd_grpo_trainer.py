from __future__ import annotations

from collections import defaultdict
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
        payload.get("status") == CandidateStatus.VALID.value
        for payload in payloads
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
    diagnostics_all = []
    eval_rows: list[dict[str, Any]] = []
    valid_candidates_for_archive: list[DiversityCandidate] = []
    for group_indices in groups.values():
        group_scores = scores[group_indices]
        if len(group_indices) == 1:
            validity_advantages = torch.zeros_like(group_scores)
        else:
            validity_advantages = (
                group_scores - group_scores.mean()
            ) / (group_scores.std(unbiased=True) + 1e-6)

        candidates = [
            DiversityCandidate(
                state_id=str(payloads[item_index]["state_id"]),
                status=CandidateStatus(str(payloads[item_index]["status"])),
                consequence_signature=payloads[item_index].get(
                    "consequence_signature"
                ),
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
        for local_index, item_index in enumerate(group_indices):
            advantage_scalars[item_index] = (
                validity_advantages[local_index]
                + effective_beta * normalized[local_index]
            )
        diversity_raw_all.extend(raw_rewards[local_index] for local_index in active)
        valid_candidates_for_archive.extend(
            candidate for candidate in candidates if candidate.valid
        )
        diagnostics_all.append(diagnostics)
        if eval_payloads:
            metadata = eval_payloads[group_indices[0]]
            valid_keys = [
                candidate.behavior_key
                for candidate in candidates
                if candidate.valid
            ]
            counts: dict[str, int] = defaultdict(int)
            for key in valid_keys:
                counts[key or ""] += 1
            valid_count = len(valid_keys)
            probabilities = [
                count / valid_count
                for count in counts.values()
            ] if valid_count else []
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
                    "effective_mode_count": math.exp(entropy)
                    if probabilities
                    else 0.0,
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
    metrics = {
        "cd_grpo/beta": effective_beta,
        "cd_grpo/validity_rate": validity_rate,
        "cd_grpo/running_validity_rate": running_validity,
        "cd_grpo/unique_behaviors_per_group": unique_behaviors
        / max(1, len(diagnostics_all)),
        "cd_grpo/duplicate_valid_rate": duplicate_completions
        / max(1, valid_completions),
        "cd_grpo/groups_skipped_rate": sum(item.skipped for item in diagnostics_all)
        / max(1, len(diagnostics_all)),
        "cd_grpo/diversity_raw_mean": sum(diversity_raw_all)
        / max(1, len(diversity_raw_all)),
        "cd_grpo/archive_size": float(len(archive.counts))
        if archive_enabled
        else 0.0,
        "cd_grpo/archive_scale_mean": sum(
            item.mean_archive_scale for item in diagnostics_all
        )
        / max(1, len(diagnostics_all)),
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
                selected = [
                    row for row in eval_rows if str(row[field]) == label
                ]
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

    def _compute_advantage(self, batch: Any, metrics: dict[str, Any]) -> Any:
        import numpy as np
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
        payloads = [
            item["cd_reward_payload"]
            for item in extra_fields
        ]
        eval_payloads = [
            item.get("cd_eval_payload", {})
            for item in extra_fields
        ]
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

        advantages, returns, cd_metrics = compute_cd_grpo_advantages(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=uid.tolist(),
            payloads=payloads,
            eval_payloads=eval_payloads,
            config=self._cd_config(),
            archive=self.cd_grpo_archive,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        metrics.update(cd_metrics)

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
