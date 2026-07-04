#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
cd "$REPO_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Missing venv at $VENV_DIR. Run: bash scripts/blackwell/setup_env.sh" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
if [[ -f scripts/cluster/prepend_venv_cuda_libs.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/cluster/prepend_venv_cuda_libs.sh
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-scattered-discovery}"
export PROJECT_NAME="${PROJECT_NAME:-$WANDB_PROJECT}"
export PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"

export RUN_CONFIG="${RUN_CONFIG:-configs/verl/runs/causal_micro_lab_blackwell_smoke.yaml}"
export CML_DATASET_PRESET="${CML_DATASET_PRESET:-smoke}"
export CML_DATASET_OUTPUT_DIR="${CML_DATASET_OUTPUT_DIR:-$DATA_ROOT/causal_micro_lab/smoke}"
export CML_EVAL_OUTPUT_DIR="${CML_EVAL_OUTPUT_DIR:-}"
export CML_TARGET_COUNTS="${CML_TARGET_COUNTS:-4,8,16}"
export CML_PROGRESS_EVERY="${CML_PROGRESS_EVERY:-16}"

export TRAIN_FILE="${TRAIN_FILE:-$CML_DATASET_OUTPUT_DIR/verl_train.jsonl}"
export VAL_FILE="${VAL_FILE:-$CML_DATASET_OUTPUT_DIR/verl_val.jsonl}"

if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  scripts/cluster/prepare_causal_micro_lab_dataset.sh
fi

export NNODES="${NNODES:-1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export ROLLOUT_N="${ROLLOUT_N:-2}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.55}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}"
export ACTOR_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-1}"
export ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export SAVE_FREQ="${SAVE_FREQ:-1}"
export TEST_FREQ="${TEST_FREQ:-1}"
export DEFAULT_AGENT_LOOP="${DEFAULT_AGENT_LOOP:-causal_micro_lab_agent_loop}"
export EXPERIMENT_NAME_PREFIX="${EXPERIMENT_NAME_PREFIX:-causal_micro_lab_blackwell_smoke_qwen3_4b_r2}"

LOG_DIR="${RL_LOG_DIR:-$ARTIFACT_ROOT/causal_micro_lab_rl_smoke/logs}"
mkdir -p "$LOG_DIR"
RUN_LOG_FILE="${RUN_LOG_FILE:-$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$RUN_LOG_FILE") 2>&1

echo "RL smoke log: $RUN_LOG_FILE"
echo "Run config: $RUN_CONFIG"
echo "Train file: $TRAIN_FILE"
echo "Val file: $VAL_FILE"
echo "GPU: $CUDA_VISIBLE_DEVICES"

scripts/cluster/run_verl_pilot_grpo.sh "$@"
