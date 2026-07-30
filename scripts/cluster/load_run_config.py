from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path
from typing import Any

import yaml


KEY_MAP = {
    "dataset_config": "DATASET_CONFIG",
    "prepare_datasets": "PREPARE_DATASETS",
    "discovery_algo": "DISCOVERY_ALGO",
    "model_id": "MODEL_ID",
    "model_path": "MODEL_PATH",
    "model_basename": "MODEL_BASENAME",
    "download_model": "DOWNLOAD_MODEL",
    "attn_implementation": "ATTN_IMPLEMENTATION",
    "use_remove_padding": "USE_REMOVE_PADDING",
    "train_file": "TRAIN_FILE",
    "val_file": "VAL_FILE",
    "val_files": "VAL_FILES",
    "n_gpus_per_node": "NGPUS_PER_NODE",
    "train_batch_size": "TRAIN_BATCH_SIZE",
    "ppo_mini_batch_size": "PPO_MINI_BATCH_SIZE",
    "rollout_n": "ROLLOUT_N",
    "rollout_gpu_mem_util": "ROLLOUT_GPU_MEM_UTIL",
    "rollout_tp": "ROLLOUT_TP",
    "sglang_attention_backend": "SGLANG_ATTENTION_BACKEND",
    "actor_micro_batch_size_per_gpu": "ACTOR_MICRO_BATCH_SIZE_PER_GPU",
    "rollout_log_prob_micro_batch_size_per_gpu": "ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU",
    "rollout_calculate_log_probs": "ROLLOUT_CALCULATE_LOG_PROBS",
    "ref_log_prob_micro_batch_size_per_gpu": "REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU",
    "actor_lr": "ACTOR_LR",
    "kl_loss_coef": "KL_LOSS_COEF",
    "entropy_coeff": "ENTROPY_COEFF",
    "total_epochs": "TOTAL_EPOCHS",
    "save_freq": "SAVE_FREQ",
    "test_freq": "TEST_FREQ",
    "max_actor_ckpt_to_keep": "MAX_ACTOR_CKPT_TO_KEEP",
    "max_critic_ckpt_to_keep": "MAX_CRITIC_CKPT_TO_KEEP",
    "max_prompt_length": "MAX_PROMPT_LENGTH",
    "max_response_length": "MAX_RESPONSE_LENGTH",
    "default_agent_loop": "DEFAULT_AGENT_LOOP",
    "agent_loop_config_path": "AGENT_LOOP_CONFIG_PATH",
    "trainer_entrypoint": "TRAINER_ENTRYPOINT",
    "trainer_use_v1": "TRAINER_USE_V1",
    "total_training_steps": "TOTAL_TRAINING_STEPS",
    "causal_micro_lab_agent_name": "CML_AGENT_NAME",
    "causal_micro_lab_generate_dataset_if_missing": "CML_GENERATE_DATASET_IF_MISSING",
    "causal_micro_lab_rebuild_dataset": "CML_REBUILD_DATASET",
    "causal_micro_lab_dataset_preset": "CML_DATASET_PRESET",
    "causal_micro_lab_dataset_output_dir": "CML_DATASET_OUTPUT_DIR",
    "causal_micro_lab_dataset_seed": "CML_DATASET_SEED",
    "causal_micro_lab_train_states_per_count": "CML_TRAIN_STATES_PER_COUNT",
    "causal_micro_lab_val_states_per_count": "CML_VAL_STATES_PER_COUNT",
    "causal_micro_lab_test_states_per_count": "CML_TEST_STATES_PER_COUNT",
    "causal_micro_lab_train_max_rows": "CML_TRAIN_MAX_ROWS",
    "causal_micro_lab_val_max_rows": "CML_VAL_MAX_ROWS",
    "causal_micro_lab_test_max_rows": "CML_TEST_MAX_ROWS",
    "causal_micro_lab_eval_preset": "CML_EVAL_PRESET",
    "causal_micro_lab_eval_output_dir": "CML_EVAL_OUTPUT_DIR",
    "causal_micro_lab_eval_seed": "CML_EVAL_SEED",
    "causal_micro_lab_target_counts": "CML_TARGET_COUNTS",
    "causal_micro_lab_progress_every": "CML_PROGRESS_EVERY",
    "causal_micro_lab_nonempty_output_reward": "CML_NONEMPTY_OUTPUT_REWARD",
    "causal_micro_lab_rule_marker_reward": "CML_RULE_MARKER_REWARD",
    "causal_micro_lab_parse_valid_reward": "CML_PARSE_VALID_REWARD",
    "causal_micro_lab_syntax_valid_reward": "CML_SYNTAX_VALID_REWARD",
    "causal_micro_lab_evidence_consistent_reward": "CML_EVIDENCE_CONSISTENT_REWARD",
    "causal_micro_lab_valid_hypothesis_reward": "CML_VALID_HYPOTHESIS_REWARD",
    "causal_micro_lab_output_mode": "CML_OUTPUT_MODE",
    "causal_micro_lab_answer_count": "CML_ANSWER_COUNT",
    "causal_micro_lab_multi_answer_format_reward": "CML_MULTI_ANSWER_FORMAT_REWARD",
    "causal_micro_lab_multi_answer_accuracy_reward": "CML_MULTI_ANSWER_ACCURACY_REWARD",
    "causal_micro_lab_multi_answer_accuracy_mode": "CML_MULTI_ANSWER_ACCURACY_MODE",
    "causal_micro_lab_length_penalty_start": "CAUSAL_MICRO_LAB_LENGTH_PENALTY_START",
    "causal_micro_lab_length_penalty_max": "CAUSAL_MICRO_LAB_LENGTH_PENALTY_MAX",
    "causal_micro_lab_mask_truncated": "CAUSAL_MICRO_LAB_MASK_TRUNCATED",
    "cd_grpo_variant": "CD_GRPO_VARIANT",
    "cd_grpo_archive": "CD_GRPO_ARCHIVE",
    "cd_grpo_beta": "CD_GRPO_BETA",
    "cd_grpo_beta_guard": "CD_GRPO_BETA_GUARD",
    "cd_grpo_beta_guard_window": "CD_GRPO_BETA_GUARD_WINDOW",
    "cd_grpo_ell": "CD_GRPO_ELL",
    "cd_grpo_gamma": "CD_GRPO_GAMMA",
    "cd_grpo_probe_fraction": "CD_GRPO_PROBE_FRACTION",
    "cd_grpo_length_penalty_start": "CD_GRPO_LENGTH_PENALTY_START",
    "ips_grpo_epsilon": "IPS_GRPO_EPSILON",
    "ips_grpo_probe_fraction": "IPS_GRPO_PROBE_FRACTION",
    "ips_grpo_length_penalty_start": "IPS_GRPO_LENGTH_PENALTY_START",
    "ips_grpo_latent_enabled": "IPS_GRPO_LATENT_ENABLED",
    "ips_grpo_latent_count": "IPS_GRPO_LATENT_COUNT",
    "ips_grpo_latent_negative_offset": "IPS_GRPO_LATENT_NEGATIVE_OFFSET",
    "ips_grpo_latent_mi_alpha": "IPS_GRPO_LATENT_MI_ALPHA",
    "ips_grpo_latent_mi_clip": "IPS_GRPO_LATENT_MI_CLIP",
    "ips_grpo_latent_mi_token_scope": "IPS_GRPO_LATENT_MI_TOKEN_SCOPE",
    "ips_grpo_latent_mi_reduction": "IPS_GRPO_LATENT_MI_REDUCTION",
    "ips_grpo_latent_mi_valid_only": "IPS_GRPO_LATENT_MI_VALID_ONLY",
    "ips_grpo_latent_use_ips": "IPS_GRPO_LATENT_USE_IPS",
    "ips_grpo_latent_ips_reward_mode": "IPS_GRPO_LATENT_IPS_REWARD_MODE",
    "ips_grpo_latent_ips_bonus_max": "IPS_GRPO_LATENT_IPS_BONUS_MAX",
    "max_steps": "MAX_STEPS",
    "max_consecutive_invalid": "MAX_CONSECUTIVE_INVALID",
    "experiment_name": "EXPERIMENT_NAME",
    "experiment_name_prefix": "EXPERIMENT_NAME_PREFIX",
    "eval_file": "EVAL_FILE",
    "output_dir": "OUTPUT_DIR",
    "run_name": "RUN_NAME",
    "max_examples": "MAX_EXAMPLES",
    "rollouts_per_spec": "ROLLOUTS_PER_SPEC",
    "prefix_ks": "PREFIX_KS",
    "eval_workers": "EVAL_WORKERS",
    "eval_shard_index": "EVAL_SHARD_INDEX",
    "eval_num_shards": "EVAL_NUM_SHARDS",
    "temperature": "TEMPERATURE",
    "top_p": "TOP_P",
    "think": "THINK",
    "thinking_fallback": "THINKING_FALLBACK",
    "fallback_max_response_length": "FALLBACK_MAX_RESPONSE_LENGTH",
    "fallback_temperature": "FALLBACK_TEMPERATURE",
    "latent_count": "LATENT_COUNT",
    "build_eval_report": "BUILD_EVAL_REPORT",
    "base_url": "BASE_URL",
    "request_timeout_s": "REQUEST_TIMEOUT_S",
    "wandb_project": "WANDB_PROJECT",
    "serve_model": "SERVE_MODEL",
    "sglang_port": "SGLANG_PORT",
    "sglang_host": "SGLANG_HOST",
    "sglang_tp": "SGLANG_TP",
    "sglang_mem_fraction_static": "SGLANG_MEM_FRACTION_STATIC",
    "sglang_extra_args": "SGLANG_EXTRA_ARGS",
    "transcripts": "TRANSCRIPTS",
}


def shell_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, list):
        return "[" + ",".join(str(item) for item in value) + "]"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit("Run config must be a YAML mapping.")

    unknown = sorted(set(data) - set(KEY_MAP))
    if unknown:
        raise SystemExit(f"Unknown run config field(s): {', '.join(unknown)}")

    for key, value in data.items():
        env_key = KEY_MAP[key]
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", env_key):
            raise SystemExit(f"Invalid env key: {env_key}")
        quoted = shlex.quote(shell_value(value))
        print(f'if [[ -z "${{{env_key}:-}}" ]]; then export {env_key}={quoted}; fi')


if __name__ == "__main__":
    main()
