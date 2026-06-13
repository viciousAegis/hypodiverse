#!/usr/bin/env bash
set -euo pipefail

# Source this after the run YAML has been loaded. It keeps HF downloads in the
# repo cache while still allowing MODEL_PATH to point at an existing checkpoint.

if [[ -z "${MODEL_ROOT:-}" && -f scripts/env.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/env.sh
fi

MODEL_PATH_WAS_SET=0
if [[ -n "${MODEL_PATH:-}" ]]; then
  MODEL_PATH_WAS_SET=1
fi

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
MODEL_BASENAME="${MODEL_BASENAME:-${MODEL_ID##*/}}"

if [[ "$MODEL_PATH_WAS_SET" == "0" ]]; then
  MODEL_PATH="$MODEL_ROOT/$MODEL_BASENAME"
fi

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  if [[ "$MODEL_PATH_WAS_SET" == "0" && "${DOWNLOAD_MODEL:-1}" == "1" ]]; then
    mkdir -p "$MODEL_PATH"
    if command -v hf >/dev/null 2>&1; then
      hf download "$MODEL_ID" --local-dir "$MODEL_PATH"
    elif command -v huggingface-cli >/dev/null 2>&1; then
      huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_PATH"
    else
      echo "Neither hf nor huggingface-cli is available for model download." >&2
      exit 1
    fi
  elif [[ "$MODEL_PATH_WAS_SET" == "0" ]]; then
    MODEL_PATH="$MODEL_ID"
  fi
fi

export MODEL_ID MODEL_PATH
