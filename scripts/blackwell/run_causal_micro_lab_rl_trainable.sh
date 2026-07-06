#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

export RUN_CONFIG="${RUN_CONFIG:-configs/verl/runs/causal_micro_lab_blackwell_smoke.yaml}"
export CML_DATASET_PRESET="${CML_DATASET_PRESET:-trainable}"
export CML_DATASET_OUTPUT_DIR="${CML_DATASET_OUTPUT_DIR:-$DATA_ROOT/causal_micro_lab/trainable}"
export CML_TARGET_COUNTS="${CML_TARGET_COUNTS:-4,8,16}"
export CML_PROGRESS_EVERY="${CML_PROGRESS_EVERY:-512}"

export TRAIN_FILE="${TRAIN_FILE:-$CML_DATASET_OUTPUT_DIR/verl_train.jsonl}"
export VAL_FILE="${VAL_FILE:-$CML_DATASET_OUTPUT_DIR/verl_val.jsonl}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.50}"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
export ACTOR_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-1}"
export ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export SAVE_FREQ="${SAVE_FREQ:-200}"
export TEST_FREQ="${TEST_FREQ:-200}"
export EXPERIMENT_NAME_PREFIX="${EXPERIMENT_NAME_PREFIX:-causal_micro_lab_blackwell_trainable_qwen3_4b_r4}"
export RL_LOG_DIR="${RL_LOG_DIR:-$ARTIFACT_ROOT/causal_micro_lab_rl_trainable/logs}"

exec "$SCRIPT_DIR/run_causal_micro_lab_rl_smoke.sh" "$@"
