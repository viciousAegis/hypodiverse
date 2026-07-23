#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

export RUN_CONFIG="${RUN_CONFIG:-configs/verl/runs/causal_micro_lab_blackwell_multi_answer_k4.yaml}"
export CML_DATASET_OUTPUT_DIR="${CML_DATASET_OUTPUT_DIR:-$DATA_ROOT/causal_micro_lab/trainable_multi_k4}"
export TRAIN_FILE="${TRAIN_FILE:-$CML_DATASET_OUTPUT_DIR/verl_train.jsonl}"
export VAL_FILE="${VAL_FILE:-$CML_DATASET_OUTPUT_DIR/verl_val.jsonl}"
export RL_LOG_DIR="${RL_LOG_DIR:-$ARTIFACT_ROOT/causal_micro_lab_rl_multi_answer_k4/logs}"

exec "$SCRIPT_DIR/run_causal_micro_lab_rl_trainable.sh" "$@"
