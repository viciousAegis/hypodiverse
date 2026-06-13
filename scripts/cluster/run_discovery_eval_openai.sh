#!/usr/bin/env bash
set -xeuo pipefail

# Evaluate a base model or trained checkpoint served by SGLang/vLLM/OpenAI-compatible API.
# Start the server separately, then run this script from the project root.

if [[ -f scripts/env.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/env.sh
fi

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export WANDB_DIR="${WANDB_DIR:-$PWD/.wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PWD/.wandb/cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$PWD/.wandb/config}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
EVAL_FILE=${EVAL_FILE:-data/verl/hypospace_causal_eval.parquet}
BASE_URL=${BASE_URL:-http://127.0.0.1:30000/v1}
OUTPUT_DIR=${OUTPUT_DIR:-results/envspec_eval}
RUN_NAME=${RUN_NAME:-eval_$(date +%Y%m%d_%H%M)}
MAX_EXAMPLES=${MAX_EXAMPLES:-}
ROLLOUTS_PER_SPEC=${ROLLOUTS_PER_SPEC:-1}
MAX_STEPS=${MAX_STEPS:-}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
WANDB_PROJECT=${WANDB_PROJECT:-}

ARGS=(
  --input "$EVAL_FILE"
  --output-dir "$OUTPUT_DIR"
  --run-name "$RUN_NAME"
  --provider openai-compatible
  --model "$MODEL_PATH"
  --base-url "$BASE_URL"
  --rollouts-per-spec "$ROLLOUTS_PER_SPEC"
  --num-predict "$MAX_RESPONSE_LENGTH"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
)

if [[ -n "$MAX_EXAMPLES" ]]; then
  ARGS+=(--max-examples "$MAX_EXAMPLES")
fi
if [[ -n "$MAX_STEPS" ]]; then
  ARGS+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "$WANDB_PROJECT" ]]; then
  ARGS+=(--wandb-project "$WANDB_PROJECT" --wandb-run-name "$RUN_NAME")
fi

python3 -m scattered_discovery.eval.envspecs "${ARGS[@]}" "$@"
