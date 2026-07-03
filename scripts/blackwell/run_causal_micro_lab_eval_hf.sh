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

export CML_DATASET_OUTPUT_DIR="${CML_DATASET_OUTPUT_DIR:-$DATA_ROOT/causal_micro_lab/pilot}"
export CML_EVAL_OUTPUT_DIR="${CML_EVAL_OUTPUT_DIR:-$DATA_ROOT/causal_micro_lab/canonical_eval}"
export EVAL_FILE="${EVAL_FILE:-$CML_EVAL_OUTPUT_DIR/verl_test.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ARTIFACT_ROOT/causal_micro_lab_eval}"

if [[ "${CML_GENERATE_DATASET_IF_MISSING:-1}" == "1" && ! -f "$EVAL_FILE" ]]; then
  scripts/cluster/prepare_causal_micro_lab_dataset.sh
fi

# shellcheck disable=SC1091
source scripts/cluster/resolve_model_path.sh

RUN_NAME="${RUN_NAME:-causal_micro_lab_eval_hf_$(date +%Y%m%d_%H%M)}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
ROLLOUTS_PER_SPEC="${ROLLOUTS_PER_SPEC:-16}"
PREFIX_KS="${PREFIX_KS:-4,8,16}"
EVAL_WORKERS="${EVAL_WORKERS:-1}"
EVAL_SHARD_INDEX="${EVAL_SHARD_INDEX:-0}"
EVAL_NUM_SHARDS="${EVAL_NUM_SHARDS:-1}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-2400}"
THINK="${THINK:-true}"

LOG_DIR="${EVAL_LOG_DIR:-$OUTPUT_DIR/$RUN_NAME/logs}"
mkdir -p "$LOG_DIR"
RUN_LOG_FILE="${RUN_LOG_FILE:-$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$RUN_LOG_FILE") 2>&1

echo "Run log: $RUN_LOG_FILE"
echo "Run name: $RUN_NAME"
echo "Provider: transformers"
echo "Model: $MODEL_PATH"
echo "Eval file: $EVAL_FILE"
echo "Workers: $EVAL_WORKERS"

ARGS=(
  --input "$EVAL_FILE"
  --output-dir "$OUTPUT_DIR"
  --run-name "$RUN_NAME"
  --provider transformers
  --model "$MODEL_PATH"
  --rollouts-per-state "$ROLLOUTS_PER_SPEC"
  --workers "$EVAL_WORKERS"
  --shard-index "$EVAL_SHARD_INDEX"
  --num-shards "$EVAL_NUM_SHARDS"
  --num-predict "$MAX_RESPONSE_LENGTH"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --request-timeout-s "$REQUEST_TIMEOUT_S"
  --think "$THINK"
)

if [[ -n "$PREFIX_KS" ]]; then
  ARGS+=(--prefix-ks "$PREFIX_KS")
fi
if [[ -n "${MAX_EXAMPLES:-}" ]]; then
  ARGS+=(--max-examples "$MAX_EXAMPLES")
fi
if [[ "${TRANSCRIPTS:-0}" == "1" ]]; then
  ARGS+=(--transcripts)
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  ARGS+=(--wandb-project "$WANDB_PROJECT" --wandb-run-name "$RUN_NAME")
fi

"$VENV_DIR/bin/python" -m scattered_discovery.envs.causal_micro_lab.eval "${ARGS[@]}" "$@"

echo "Eval output: $OUTPUT_DIR/$RUN_NAME"
echo "Latest per-sample summary: $OUTPUT_DIR/$RUN_NAME/latest/summary.json"
echo "Latest grouped set summary: $OUTPUT_DIR/$RUN_NAME/latest/set_summary.json"
echo "Run log: $RUN_LOG_FILE"
