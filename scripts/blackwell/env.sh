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
if [[ -z "${BLACKWELL_SCRATCH_BASE:-}" ]]; then
  if [[ -d "/scratch/$CSRID" || -w /scratch ]]; then
    export BLACKWELL_SCRATCH_BASE="/scratch/$CSRID"
  else
    export BLACKWELL_SCRATCH_BASE="${HOME:-/homes/$CSRID}"
  fi
fi
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

if [[ ! -d "$BLACKWELL_SCRATCH_BASE" && ! -w "$(dirname "$BLACKWELL_SCRATCH_BASE")" ]]; then
  cat >&2 <<EOF
Cannot create Blackwell scratch directory:
  $BLACKWELL_SCRATCH_BASE

The parent directory is not writable by this user:
  $(dirname "$BLACKWELL_SCRATCH_BASE")

Ask the machine admin to create/chown it, for example:
  sudo mkdir -p "$BLACKWELL_SCRATCH_BASE"
  sudo chown "$CSRID":"$CSRID" "$BLACKWELL_SCRATCH_BASE"

Or point BLACKWELL_SCRATCH_BASE at an existing writable scratch directory:
  export BLACKWELL_SCRATCH_BASE=/path/to/writable/scratch/$CSRID
EOF
  return 1 2>/dev/null || exit 1
fi

mkdir -p "$BLACKWELL_RUN_ROOT"

# shellcheck disable=SC1091
source "$REPO_DIR/scripts/env.sh"

unset _BW_SCRIPT_DIR _BW_REPO_DIR
