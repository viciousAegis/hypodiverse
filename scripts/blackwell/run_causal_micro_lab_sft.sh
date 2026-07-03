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

if [[ "${SFT_DISABLE_LORA:-0}" != "1" ]]; then
  "$VENV_DIR/bin/python" - <<'PY'
from __future__ import annotations

import importlib.metadata
import sys

from packaging.version import Version

try:
    torchao_version = importlib.metadata.version("torchao")
except importlib.metadata.PackageNotFoundError:
    raise SystemExit(0)

if Version(torchao_version) < Version("0.16.0"):
    print(
        "Incompatible torchao for PEFT LoRA SFT: "
        f"torchao=={torchao_version}, but PEFT requires torchao>=0.16.0.",
        file=sys.stderr,
    )
    print(
        'Fix inside the Blackwell repo with: source scripts/blackwell/env.sh && '
        'source "$VENV_DIR/bin/activate" && uv pip install -U "torchao>=0.16.0"',
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-scattered-discovery}"

export CML_DATASET_OUTPUT_DIR="${CML_DATASET_OUTPUT_DIR:-$DATA_ROOT/causal_micro_lab/pilot}"
export CML_EVAL_OUTPUT_DIR="${CML_EVAL_OUTPUT_DIR:-$DATA_ROOT/causal_micro_lab/canonical_eval}"
export CML_DATASET_PRESET="${CML_DATASET_PRESET:-pilot}"
export CML_EVAL_PRESET="${CML_EVAL_PRESET:-canonical_eval}"
export CML_TARGET_COUNTS="${CML_TARGET_COUNTS:-4,8,16}"

if [[ ! -f "$CML_DATASET_OUTPUT_DIR/sft_train.jsonl" || ! -f "$CML_DATASET_OUTPUT_DIR/sft_val.jsonl" ]]; then
  scripts/cluster/prepare_causal_micro_lab_dataset.sh
fi

# shellcheck disable=SC1091
source scripts/cluster/resolve_model_path.sh

RUN_NAME="${RUN_NAME:-causal_micro_lab_sft_qwen3_4b_lora}"
OUTPUT_DIR="${SFT_OUTPUT_DIR:-$CHECKPOINT_ROOT/causal_micro_lab_sft/$RUN_NAME}"
TRAIN_FILE="${TRAIN_FILE:-$CML_DATASET_OUTPUT_DIR/sft_train.jsonl}"
VAL_FILE="${VAL_FILE:-$CML_DATASET_OUTPUT_DIR/sft_val.jsonl}"

mkdir -p "$OUTPUT_DIR/logs"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/logs/run_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "SFT log: $LOG_FILE"
echo "Run name: $RUN_NAME"
echo "Model: $MODEL_PATH"
echo "Train file: $TRAIN_FILE"
echo "Val file: $VAL_FILE"
echo "Output dir: $OUTPUT_DIR"

ARGS=(
  --model "$MODEL_PATH"
  --train-file "$TRAIN_FILE"
  --val-file "$VAL_FILE"
  --output-dir "$OUTPUT_DIR"
  --run-name "$RUN_NAME"
  --max-length "${SFT_MAX_LENGTH:-3072}"
  --epochs "${SFT_EPOCHS:-1}"
  --learning-rate "${SFT_LR:-2e-4}"
  --per-device-train-batch-size "${SFT_BATCH_SIZE:-2}"
  --per-device-eval-batch-size "${SFT_EVAL_BATCH_SIZE:-2}"
  --gradient-accumulation-steps "${SFT_GRAD_ACCUM:-8}"
  --logging-steps "${SFT_LOGGING_STEPS:-10}"
  --eval-steps "${SFT_EVAL_STEPS:-100}"
  --save-steps "${SFT_SAVE_STEPS:-100}"
  --save-total-limit "${SFT_SAVE_TOTAL_LIMIT:-3}"
  --max-steps "${SFT_MAX_STEPS:--1}"
  --lora-r "${SFT_LORA_R:-32}"
  --lora-alpha "${SFT_LORA_ALPHA:-64}"
  --lora-dropout "${SFT_LORA_DROPOUT:-0.05}"
)

if [[ -n "${WANDB_PROJECT:-}" ]]; then
  ARGS+=(--wandb-project "$WANDB_PROJECT")
fi
if [[ "${SFT_ENABLE_THINKING_TEMPLATE:-0}" == "1" ]]; then
  ARGS+=(--enable-thinking-template)
fi
if [[ "${SFT_DISABLE_LORA:-0}" == "1" ]]; then
  ARGS+=(--disable-lora)
fi
if [[ "${SFT_SAVE_MERGED:-1}" == "0" ]]; then
  ARGS+=(--no-save-merged)
fi
if [[ -n "${SFT_RESUME_FROM_CHECKPOINT:-}" ]]; then
  ARGS+=(--resume-from-checkpoint "$SFT_RESUME_FROM_CHECKPOINT")
fi

"$VENV_DIR/bin/python" scripts/train_causal_micro_lab_sft.py "${ARGS[@]}" "$@"

echo "SFT output: $OUTPUT_DIR/final"
echo "Merged serving model: $OUTPUT_DIR/merged"
echo "SFT log: $LOG_FILE"
