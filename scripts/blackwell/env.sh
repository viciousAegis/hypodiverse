#!/usr/bin/env bash
# Source this on the Blackwell workstation before running setup/eval/training.

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  _BW_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  _BW_REPO_DIR="$(cd "$_BW_SCRIPT_DIR/../.." && pwd)"
else
  _BW_REPO_DIR="$PWD"
fi

export REPO_DIR="${REPO_DIR:-$_BW_REPO_DIR}"
export BLACKWELL_USER="${BLACKWELL_USER:-${USER:?set USER or BLACKWELL_USER}}"
export CSRID="${CSRID:-$BLACKWELL_USER}"
export BLACKWELL_SCRATCH_BASE="${BLACKWELL_SCRATCH_BASE:-/scratch/$BLACKWELL_USER}"
export BLACKWELL_RUN_ROOT="${BLACKWELL_RUN_ROOT:-$BLACKWELL_SCRATCH_BASE/open-discovery}"

export CACHE_ROOT="${CACHE_ROOT:-$BLACKWELL_RUN_ROOT/.cache}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$BLACKWELL_RUN_ROOT/artifacts}"
export DATA_ROOT="${DATA_ROOT:-$BLACKWELL_RUN_ROOT/data}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$BLACKWELL_RUN_ROOT/checkpoints}"
export MODEL_ROOT="${MODEL_ROOT:-$BLACKWELL_RUN_ROOT/models}"
export VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
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

_bw_require_scratch_path() {
  local name="$1"
  local value="$2"
  case "$value" in
    "$BLACKWELL_SCRATCH_BASE"/*) ;;
    *)
      cat >&2 <<EOF
Blackwell path $name is not under scratch:
  $value

Expected it under:
  $BLACKWELL_SCRATCH_BASE

Unset the variable or point it at scratch before running this script.
EOF
      return 1
      ;;
  esac
}

_bw_require_scratch_path BLACKWELL_RUN_ROOT "$BLACKWELL_RUN_ROOT"
_bw_require_scratch_path CACHE_ROOT "$CACHE_ROOT"
_bw_require_scratch_path ARTIFACT_ROOT "$ARTIFACT_ROOT"
_bw_require_scratch_path DATA_ROOT "$DATA_ROOT"
_bw_require_scratch_path CHECKPOINT_ROOT "$CHECKPOINT_ROOT"
_bw_require_scratch_path MODEL_ROOT "$MODEL_ROOT"
_bw_require_scratch_path WANDB_DIR "$WANDB_DIR"
_bw_require_scratch_path TMPDIR "$TMPDIR"
_bw_require_scratch_path RAY_TMPDIR "$RAY_TMPDIR"

unset _BW_SCRIPT_DIR _BW_REPO_DIR
unset -f _bw_require_scratch_path
