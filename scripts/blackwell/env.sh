#!/usr/bin/env bash
# Source this on the Blackwell workstation before running setup/eval/training.

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  _BW_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  _BW_REPO_DIR="$(cd "$_BW_SCRIPT_DIR/../.." && pwd)"
else
  _BW_REPO_DIR="$PWD"
fi

export REPO_DIR="${REPO_DIR:-$_BW_REPO_DIR}"
export CSRID="${CSRID:-${USER:?set CSRID or USER}}"
export BLACKWELL_SCRATCH_BASE="${BLACKWELL_SCRATCH_BASE:-/scratch/$CSRID}"
export BLACKWELL_RUN_ROOT="${BLACKWELL_RUN_ROOT:-$BLACKWELL_SCRATCH_BASE/open-discovery}"

export CACHE_ROOT="${CACHE_ROOT:-$BLACKWELL_RUN_ROOT/.cache}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$BLACKWELL_RUN_ROOT/artifacts}"
export DATA_ROOT="${DATA_ROOT:-$BLACKWELL_RUN_ROOT/data}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$BLACKWELL_RUN_ROOT/checkpoints}"
export MODEL_ROOT="${MODEL_ROOT:-$BLACKWELL_RUN_ROOT/models}"
export VENV_DIR="${VENV_DIR:-$BLACKWELL_RUN_ROOT/.venv}"
export WANDB_DIR="${WANDB_DIR:-$BLACKWELL_RUN_ROOT/.wandb}"
export TMPDIR="${TMPDIR:-$BLACKWELL_RUN_ROOT/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-$BLACKWELL_RUN_ROOT/ray-tmp}"

mkdir -p "$BLACKWELL_RUN_ROOT"

# shellcheck disable=SC1091
source "$REPO_DIR/scripts/env.sh"

unset _BW_SCRIPT_DIR _BW_REPO_DIR
