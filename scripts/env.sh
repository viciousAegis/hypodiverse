#!/usr/bin/env bash
# Shared local/cluster environment defaults. Source this file; do not execute it.

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  _SD_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  _SD_DEFAULT_REPO_DIR="$(cd "$_SD_SCRIPT_DIR/.." && pwd)"
else
  _SD_DEFAULT_REPO_DIR="$PWD"
fi

export REPO_DIR="${REPO_DIR:-$_SD_DEFAULT_REPO_DIR}"

_SD_XTRACE=0
case "$-" in
  *x*)
    _SD_XTRACE=1
    set +x
    ;;
esac

_SD_ENV_FILE="${ENV_FILE:-$REPO_DIR/.env}"
if [[ -f "$_SD_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$_SD_ENV_FILE"
  set +a
fi

if [[ "$_SD_XTRACE" == "1" ]]; then
  set -x
fi

export CACHE_ROOT="${CACHE_ROOT:-$REPO_DIR/.cache}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO_DIR/artifacts}"
export DATA_ROOT="${DATA_ROOT:-$REPO_DIR/data}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$REPO_DIR/checkpoints}"
export MODEL_ROOT="${MODEL_ROOT:-$CACHE_ROOT/models}"
export VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_DIR/.uv-cache}"
export UV_TOOL_DIR="${UV_TOOL_DIR:-$REPO_DIR/.uv-tools}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-$HF_HOME/assets}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export RAY_TMPDIR="${RAY_TMPDIR:-$CACHE_ROOT/ray}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CACHE_ROOT/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$CACHE_ROOT/nv/ComputeCache}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-$CACHE_ROOT/vllm}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_ROOT/vllm}"
export SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-$CACHE_ROOT/sglang}"
export WANDB_DIR="${WANDB_DIR:-$REPO_DIR/.wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_DIR/cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$WANDB_DIR/config}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$WANDB_DIR/data}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$ARTIFACT_ROOT/wandb}"
export TMPDIR="${TMPDIR:-$CACHE_ROOT/tmp}"

mkdir -p \
  "$CACHE_ROOT" \
  "$ARTIFACT_ROOT" \
  "$DATA_ROOT" \
  "$CHECKPOINT_ROOT" \
  "$MODEL_ROOT" \
  "$UV_CACHE_DIR" \
  "$UV_TOOL_DIR" \
  "$XDG_CACHE_HOME" \
  "$PIP_CACHE_DIR" \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$HF_ASSETS_CACHE" \
  "$HF_XET_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$RAY_TMPDIR" \
  "$TORCH_HOME" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TRITON_CACHE_DIR" \
  "$CUDA_CACHE_PATH" \
  "$VLLM_CONFIG_ROOT" \
  "$VLLM_CACHE_ROOT" \
  "$SGLANG_CACHE_DIR" \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" \
  "$WANDB_CONFIG_DIR" \
  "$WANDB_DATA_DIR" \
  "$WANDB_ARTIFACT_DIR" \
  "$TMPDIR"

case ":${PYTHONPATH:-}:" in
  *":$REPO_DIR/src:"*) ;;
  *) export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

unset _SD_SCRIPT_DIR _SD_DEFAULT_REPO_DIR _SD_ENV_FILE _SD_XTRACE
