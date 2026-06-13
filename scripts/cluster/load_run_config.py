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
    "experiment_name": "EXPERIMENT_NAME",
    "experiment_name_prefix": "EXPERIMENT_NAME_PREFIX",
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
