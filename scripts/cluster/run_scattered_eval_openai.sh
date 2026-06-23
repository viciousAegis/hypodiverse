#!/usr/bin/env bash
set -xeuo pipefail

# Run a scattered-discovery calibration eval against an OpenAI-compatible server.
# If SERVE_MODEL=1, this script starts an SGLang server for MODEL_PATH first.

if [[ -f scripts/env.sh ]]; then
  set +x
  # shellcheck disable=SC1091
  source scripts/env.sh
  set -x
fi

export CUDA_HOME="${CLUSTER_CUDA_HOME:-/usr/local/software/cuda/12.1}"
export CUDA_PATH="$CUDA_HOME"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

EVAL_CONFIG="${EVAL_CONFIG:-configs/verl/eval/scattered_smoke_base_single.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python
fi
CONFIG_EXPORTS="$("$PYTHON_BIN" scripts/cluster/load_run_config.py "$EVAL_CONFIG")"
eval "$CONFIG_EXPORTS"

DATASET_CONFIG="${DATASET_CONFIG:?missing DATASET_CONFIG}"
if [[ "${PREPARE_DATASETS:-1}" == "1" ]]; then
  DATASET_CONFIG="$DATASET_CONFIG" scripts/cluster/prepare_verl_datasets.sh
fi

# shellcheck disable=SC1091
source scripts/cluster/resolve_model_path.sh

SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
BASE_URL="${BASE_URL:-http://${SGLANG_HOST}:${SGLANG_PORT}/v1}"
RUN_NAME="${RUN_NAME:-scattered_eval_$(date +%Y%m%d_%H%M)}"
OUTPUT_DIR="${OUTPUT_DIR:-results/envspec_eval}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
ROLLOUTS_PER_SPEC="${ROLLOUTS_PER_SPEC:-1}"
EVAL_WORKERS="${EVAL_WORKERS:-1}"
EVAL_SHARD_INDEX="${EVAL_SHARD_INDEX:-0}"
EVAL_NUM_SHARDS="${EVAL_NUM_SHARDS:-1}"
MAX_CONSECUTIVE_INVALID="${MAX_CONSECUTIVE_INVALID:-2}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-600}"
TRANSCRIPTS="${TRANSCRIPTS:-0}"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_for_server() {
  "$PYTHON_BIN" - <<'PY'
import os
import sys
import time
import urllib.error
import urllib.request

base_url = os.environ["BASE_URL"].rstrip("/")
deadline = time.monotonic() + int(os.environ.get("SERVER_START_TIMEOUT_S", "900"))
last_error = None
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(base_url + "/models", timeout=5) as response:
            if 200 <= response.status < 500:
                print(f"Server ready: {base_url}")
                raise SystemExit(0)
    except Exception as exc:  # noqa: BLE001
        last_error = exc
    time.sleep(5)
print(f"Timed out waiting for server at {base_url}: {last_error}", file=sys.stderr)
raise SystemExit(1)
PY
}

if [[ "${SERVE_MODEL:-1}" == "1" ]]; then
  SGLANG_TP="${SGLANG_TP:-1}"
  SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.55}"
  read -r -a SGLANG_EXTRA <<< "${SGLANG_EXTRA_ARGS:-}"
  echo "Starting SGLang with CUDA_HOME=$CUDA_HOME"
  echo "FlashInfer cache: ${FLASHINFER_CACHE_DIR:-unset}"
  command -v nvcc || true
  nvcc --version || true
  "$PYTHON_BIN" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$SGLANG_HOST" \
    --port "$SGLANG_PORT" \
    --tp-size "$SGLANG_TP" \
    --mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
    "${SGLANG_EXTRA[@]}" &
  SERVER_PID=$!
  export BASE_URL SERVER_START_TIMEOUT_S
  wait_for_server
fi

ARGS=(
  --input "${EVAL_FILE:?missing EVAL_FILE}"
  --output-dir "$OUTPUT_DIR"
  --run-name "$RUN_NAME"
  --provider openai-compatible
  --model "$MODEL_PATH"
  --base-url "$BASE_URL"
  --rollouts-per-spec "$ROLLOUTS_PER_SPEC"
  --workers "$EVAL_WORKERS"
  --shard-index "$EVAL_SHARD_INDEX"
  --num-shards "$EVAL_NUM_SHARDS"
  --max-consecutive-invalid "$MAX_CONSECUTIVE_INVALID"
  --num-predict "$MAX_RESPONSE_LENGTH"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --request-timeout-s "$REQUEST_TIMEOUT_S"
)

if [[ -n "${MAX_EXAMPLES:-}" ]]; then
  ARGS+=(--max-examples "$MAX_EXAMPLES")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  ARGS+=(--max-steps "$MAX_STEPS")
fi
if [[ "${TRANSCRIPTS:-0}" == "1" ]]; then
  ARGS+=(--transcripts)
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  ARGS+=(--wandb-project "$WANDB_PROJECT" --wandb-run-name "$RUN_NAME")
fi

"$PYTHON_BIN" -m scattered_discovery.eval.envspecs "${ARGS[@]}" "$@"

echo "Eval output: $OUTPUT_DIR/$RUN_NAME"
echo "Latest aggregate summary: $OUTPUT_DIR/$RUN_NAME/latest/summary.json"
echo "Latest stratified summary: $OUTPUT_DIR/$RUN_NAME/latest/summary_by_task.json"
