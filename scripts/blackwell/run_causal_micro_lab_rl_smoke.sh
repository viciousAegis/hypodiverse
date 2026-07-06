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
export NNODES="${NNODES:-1}"

LOG_DIR="${RL_LOG_DIR:-$ARTIFACT_ROOT/causal_micro_lab_rl_smoke/logs}"
mkdir -p "$LOG_DIR"
RUN_LOG_FILE="${RUN_LOG_FILE:-$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$RUN_LOG_FILE") 2>&1

echo "RL smoke log: $RUN_LOG_FILE"
echo "Run config: $RUN_CONFIG"
echo "GPU: $CUDA_VISIBLE_DEVICES"

DEFAULT_HYDRA_ARGS=()
has_attention_backend=0
has_calculate_log_probs=0
for arg in "$@"; do
  case "$arg" in
    *actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=*)
      has_attention_backend=1
      ;;
    *actor_rollout_ref.rollout.calculate_log_probs=*)
      has_calculate_log_probs=1
      ;;
  esac
done

if [[ "$has_attention_backend" == "0" ]]; then
  DEFAULT_HYDRA_ARGS+=(
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer
  )
fi
if [[ "$has_calculate_log_probs" == "0" ]]; then
  DEFAULT_HYDRA_ARGS+=(
    actor_rollout_ref.rollout.calculate_log_probs=False
  )
fi

scripts/cluster/run_verl_pilot_grpo.sh "${DEFAULT_HYDRA_ARGS[@]}" "$@"
