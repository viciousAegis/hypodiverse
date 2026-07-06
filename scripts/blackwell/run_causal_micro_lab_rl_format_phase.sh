#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

export CML_DATASET_PRESET="${CML_DATASET_PRESET:-trainable}"
export CML_DATASET_OUTPUT_DIR="${CML_DATASET_OUTPUT_DIR:-$DATA_ROOT/causal_micro_lab/trainable}"
export TRAIN_FILE="${TRAIN_FILE:-$CML_DATASET_OUTPUT_DIR/verl_train.jsonl}"
export VAL_FILE="${VAL_FILE:-$CML_DATASET_OUTPUT_DIR/verl_val.jsonl}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.50}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
export CAUSAL_MICRO_LAB_DISABLE_THINKING="${CAUSAL_MICRO_LAB_DISABLE_THINKING:-1}"
export CAUSAL_MICRO_LAB_DENSE_REWARD="${CAUSAL_MICRO_LAB_DENSE_REWARD:-1}"

export SAVE_FREQ="${SAVE_FREQ:-50}"
export TEST_FREQ="${TEST_FREQ:-50}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export EXPERIMENT_NAME_PREFIX="${EXPERIMENT_NAME_PREFIX:-causal_micro_lab_blackwell_nothink_warmup_qwen3_4b_r4}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${EXPERIMENT_NAME_PREFIX}_$(date +%Y%m%d_%H%M)}"
export RL_LOG_DIR="${RL_LOG_DIR:-$ARTIFACT_ROOT/causal_micro_lab_rl_format/logs}"

exec "$SCRIPT_DIR/run_causal_micro_lab_rl_trainable.sh" "$@"
