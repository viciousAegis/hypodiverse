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
    "total_epochs": "TOTAL_EPOCHS",
    "save_freq": "SAVE_FREQ",
    "test_freq": "TEST_FREQ",
    "max_prompt_length": "MAX_PROMPT_LENGTH",
    "max_response_length": "MAX_RESPONSE_LENGTH",
    "default_agent_loop": "DEFAULT_AGENT_LOOP",
    "causal_micro_lab_generate_dataset_if_missing": "CML_GENERATE_DATASET_IF_MISSING",
    "causal_micro_lab_dataset_preset": "CML_DATASET_PRESET",
    "causal_micro_lab_dataset_output_dir": "CML_DATASET_OUTPUT_DIR",
    "causal_micro_lab_dataset_seed": "CML_DATASET_SEED",
    "causal_micro_lab_eval_preset": "CML_EVAL_PRESET",
    "causal_micro_lab_eval_output_dir": "CML_EVAL_OUTPUT_DIR",
    "causal_micro_lab_eval_seed": "CML_EVAL_SEED",
    "causal_micro_lab_target_counts": "CML_TARGET_COUNTS",
    "causal_micro_lab_progress_every": "CML_PROGRESS_EVERY",
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
