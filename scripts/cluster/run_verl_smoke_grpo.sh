#!/usr/bin/env bash
set -xeuo pipefail

# Cheap end-to-end cluster check: dataset generation, SGLang rollout, GRPO update,
# checkpointing, validation, and W&B logging.

if [[ -f scripts/env.sh ]]; then
  set +x
  # shellcheck disable=SC1091
  source scripts/env.sh
  set -x
fi

DATASET_CONFIG="${DATASET_CONFIG:-configs/verl/datasets/scattered_smoke.yaml}"
if [[ "${PREPARE_DATASETS:-1}" == "1" ]]; then
  DATASET_CONFIG="$DATASET_CONFIG" scripts/cluster/prepare_verl_datasets.sh
fi

export DISCOVERY_ALGO="${DISCOVERY_ALGO:-grpo}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
export TRAIN_FILE="${TRAIN_FILE:-data/verl/scattered_causal_smoke_train.parquet}"
export VAL_FILE="${VAL_FILE:-data/verl/scattered_causal_smoke_val.parquet}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export SAVE_FREQ="${SAVE_FREQ:-4}"
export TEST_FREQ="${TEST_FREQ:-2}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-scattered_smoke_qwen3_4b_$(date +%Y%m%d_%H%M)}"

scripts/cluster/run_verl_discovery_grpo.sh "$@"
