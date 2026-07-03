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
export EVAL_CONFIG="${EVAL_CONFIG:-configs/verl/eval/causal_micro_lab_test_k16.yaml}"
export WANDB_PROJECT="${WANDB_PROJECT:-scattered-discovery}"
export EVAL_NUM_SHARDS="${EVAL_NUM_SHARDS:-1}"
export EVAL_SHARD_INDEX="${EVAL_SHARD_INDEX:-0}"
export SGLANG_PORT="${SGLANG_PORT:-30000}"
export SGLANG_TP="${SGLANG_TP:-1}"

# RTX PRO 6000 Blackwell has 96GB VRAM. Leave some headroom for CUDA graphs,
# kernels, NCCL, and driver allocations while allowing a large KV cache.
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.86}"
export EVAL_WORKERS="${EVAL_WORKERS:-112}"
export REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-2400}"

scripts/cluster/run_causal_micro_lab_eval_openai.sh "$@"
