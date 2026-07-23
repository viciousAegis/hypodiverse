#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
cd "$REPO_DIR"

run_config="configs/verl/runs/causal_micro_lab_blackwell_cd_grpo_smoke.yaml"
if [[ "${1:-}" == "--config" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "--config requires a YAML path" >&2
    exit 2
  fi
  run_config="$2"
  shift 2
fi

export RUN_CONFIG="$run_config"
export CML_DATASET_OUTPUT_DIR="$DATA_ROOT/causal_micro_lab/cd_grpo"
export TRAIN_FILE="$CML_DATASET_OUTPUT_DIR/verl_train.jsonl"
export VAL_FILE="$CML_DATASET_OUTPUT_DIR/verl_val.jsonl"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export RL_LOG_DIR="${RL_LOG_DIR:-$ARTIFACT_ROOT/causal_micro_lab_cd_grpo/logs}"

"$VENV_DIR/bin/python" scripts/cluster/check_cd_grpo_verl.py

exec "$SCRIPT_DIR/run_causal_micro_lab_rl_smoke.sh" "$@"
