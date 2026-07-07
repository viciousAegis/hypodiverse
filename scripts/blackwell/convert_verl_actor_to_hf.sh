#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Convert a veRL FSDP actor checkpoint to a Hugging Face serving directory.

Required:
  --run NAME    veRL experiment directory name under checkpoints/scattered-discovery/
  --step N      numeric global step to convert

Optional:
  --checkpoint-root DIR  checkpoint project root
  --out DIR              output Hugging Face model directory

Example:
  bash scripts/blackwell/convert_verl_actor_to_hf.sh \
    --run causal_micro_lab_blackwell_validity_format02_r8_from_warmup_gs4_think_20260706_1547 \
    --step 100
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

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

RUN_ARG="${RUN:-}"
STEP_ARG="${STEP:-}"
CKPT_ROOT_ARG="${CKPT_ROOT:-}"
OUT_ARG="${OUT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --run)
      RUN_ARG="${2:?--run requires a value}"
      shift 2
      ;;
    --step)
      STEP_ARG="${2:?--step requires a value}"
      shift 2
      ;;
    --checkpoint-root)
      CKPT_ROOT_ARG="${2:?--checkpoint-root requires a value}"
      shift 2
      ;;
    --out)
      OUT_ARG="${2:?--out requires a value}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$RUN_ARG" || -z "$STEP_ARG" ]]; then
  usage >&2
  echo >&2
  echo "ERROR: pass --run and --step." >&2
  exit 2
fi

RUN="$RUN_ARG"
STEP="$STEP_ARG"
CKPT_ROOT="${CKPT_ROOT_ARG:-$CHECKPOINT_ROOT/scattered-discovery}"
STEP_DIR="$CKPT_ROOT/$RUN/global_step_$STEP"
ACTOR_DIR="$STEP_DIR/actor"
OUT="${OUT_ARG:-$MODEL_ROOT/${RUN}_global_step_${STEP}_hf}"

if [[ ! -d "$STEP_DIR" ]]; then
  echo "Missing checkpoint step directory: $STEP_DIR" >&2
  exit 1
fi
if [[ ! -d "$ACTOR_DIR" ]]; then
  echo "Missing actor checkpoint directory: $ACTOR_DIR" >&2
  echo >&2
  echo "This step is probably bookkeeping-only. Available files:" >&2
  find "$STEP_DIR" -maxdepth 2 -type f | sort >&2
  exit 1
fi
if ! compgen -G "$ACTOR_DIR/model_world_size_*_rank_*.pt" >/dev/null; then
  echo "No FSDP model shards found in: $ACTOR_DIR" >&2
  echo >&2
  echo "This step is probably not a real model checkpoint. Available files:" >&2
  find "$ACTOR_DIR" -maxdepth 2 -type f | sort >&2
  exit 1
fi
if [[ ! -f "$ACTOR_DIR/huggingface/config.json" ]]; then
  echo "Missing Hugging Face metadata: $ACTOR_DIR/huggingface/config.json" >&2
  echo "Cannot merge this checkpoint without the actor/huggingface metadata." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

echo "Converting veRL actor checkpoint to Hugging Face model"
echo "  run:      $RUN"
echo "  step:     $STEP"
echo "  actor:    $ACTOR_DIR"
echo "  output:   $OUT"

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$ACTOR_DIR" \
  --target_dir "$OUT"

echo
echo "Converted model files:"
find "$OUT" -maxdepth 1 -type f | sort
