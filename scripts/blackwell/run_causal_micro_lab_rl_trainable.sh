#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

export RUN_CONFIG="${RUN_CONFIG:-configs/verl/runs/causal_micro_lab_blackwell_trainable.yaml}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export RL_LOG_DIR="${RL_LOG_DIR:-$ARTIFACT_ROOT/causal_micro_lab_rl_trainable/logs}"

exec "$SCRIPT_DIR/run_causal_micro_lab_rl_smoke.sh" "$@"
