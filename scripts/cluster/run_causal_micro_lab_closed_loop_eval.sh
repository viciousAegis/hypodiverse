#!/usr/bin/env bash
set -euo pipefail
set -x

if [[ -f scripts/env.sh ]]; then
  set +x
  # shellcheck disable=SC1091
  source scripts/env.sh
  set -x
fi

if [[ -n "${CLUSTER_CUDA_HOME:-}" ]]; then
  export CUDA_HOME="$CLUSTER_CUDA_HOME"
elif [[ -d /usr/local/cuda ]]; then
  export CUDA_HOME=/usr/local/cuda
elif [[ -d /usr/local/software/cuda/12.1 ]]; then
  export CUDA_HOME=/usr/local/software/cuda/12.1
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export CUDA_PATH="$CUDA_HOME"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

EVAL_CONFIG="${EVAL_CONFIG:-configs/verl/eval/hypodiverse_closed_loop.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python
fi
CONFIG_EXPORTS="$("$PYTHON_BIN" scripts/cluster/load_run_config.py "$EVAL_CONFIG")"
eval "$CONFIG_EXPORTS"

"$PYTHON_BIN" scripts/build_causal_micro_lab_closed_loop_states.py \
  --output-dir "$(dirname "${EVAL_FILE:?missing EVAL_FILE}")" \
  --verify-only

# shellcheck disable=SC1091
source scripts/cluster/resolve_model_path.sh

SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
BASE_URL="${BASE_URL:-http://${SGLANG_HOST}:${SGLANG_PORT}/v1}"
RUN_NAME="${RUN_NAME:-causal_micro_lab_closed_loop_$(date +%Y%m%d_%H%M)}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/causal_micro_lab_closed_loop}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-6000}"
ROLLOUTS_PER_SPEC="${ROLLOUTS_PER_SPEC:-8}"
MAX_STEPS="${MAX_STEPS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-128}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-1800}"
THINK="${THINK:-true}"
THINKING_FALLBACK="${THINKING_FALLBACK:-1}"
FALLBACK_MAX_RESPONSE_LENGTH="${FALLBACK_MAX_RESPONSE_LENGTH:-256}"
FALLBACK_TEMPERATURE="${FALLBACK_TEMPERATURE:-0.0}"
LATENT_COUNT="${LATENT_COUNT:-0}"
INITIAL_MODE_COUNTS="${INITIAL_MODE_COUNTS:-16,32}"
TRAJECTORIES_PER_COUNT="${TRAJECTORIES_PER_COUNT:-64}"
CLOSED_LOOP_RESUME="${CLOSED_LOOP_RESUME:-1}"
DEDUPLICATE_PLANNER_MODES="${DEDUPLICATE_PLANNER_MODES:-0}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
REFERENCE_SAMPLES="${REFERENCE_SAMPLES:-32}"
PROGRESS_INTERVAL_S="${PROGRESS_INTERVAL_S:-60}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"

LOG_DIR="${EVAL_LOG_DIR:-$OUTPUT_DIR/$RUN_NAME/logs}"
mkdir -p "$LOG_DIR"
RUN_LOG_FILE="${RUN_LOG_FILE:-$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$RUN_LOG_FILE") 2>&1
echo "Run log: $RUN_LOG_FILE"
echo "Run name: $RUN_NAME"
echo "Eval config: $EVAL_CONFIG"

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
import time
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
raise SystemExit(f"Timed out waiting for {base_url}: {last_error}")
PY
}

if [[ "${SERVE_MODEL:-1}" == "1" ]]; then
  SGLANG_TP="${SGLANG_TP:-1}"
  SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.82}"
  read -r -a SGLANG_EXTRA <<< "${SGLANG_EXTRA_ARGS:-}"
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
  --input "$EVAL_FILE"
  --output-dir "$OUTPUT_DIR"
  --run-name "$RUN_NAME"
  --provider openai-compatible
  --model "$MODEL_PATH"
  --base-url "$BASE_URL"
  --k "$ROLLOUTS_PER_SPEC"
  --max-steps "$MAX_STEPS"
  --initial-mode-counts "$INITIAL_MODE_COUNTS"
  --trajectories-per-count "$TRAJECTORIES_PER_COUNT"
  --workers "$EVAL_WORKERS"
  --num-predict "$MAX_RESPONSE_LENGTH"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --request-timeout-s "$REQUEST_TIMEOUT_S"
  --think "$THINK"
  --fallback-num-predict "$FALLBACK_MAX_RESPONSE_LENGTH"
  --fallback-temperature "$FALLBACK_TEMPERATURE"
  --latent-count "$LATENT_COUNT"
  --bootstrap-samples "$BOOTSTRAP_SAMPLES"
  --reference-samples "$REFERENCE_SAMPLES"
  --progress-interval-s "$PROGRESS_INTERVAL_S"
)
if [[ "$THINKING_FALLBACK" == "1" ]]; then
  ARGS+=(--thinking-fallback)
fi
if [[ "$CLOSED_LOOP_RESUME" == "1" ]]; then
  ARGS+=(--resume)
fi
if [[ "$DEDUPLICATE_PLANNER_MODES" == "1" ]]; then
  ARGS+=(--deduplicate-planner-modes)
fi
if [[ -n "$WANDB_PROJECT" ]]; then
  ARGS+=(--wandb-project "$WANDB_PROJECT" --wandb-run-name "$RUN_NAME")
fi

"$PYTHON_BIN" scripts/run_causal_micro_lab_closed_loop_eval.py "${ARGS[@]}" "$@"

echo "Trace: $OUTPUT_DIR/$RUN_NAME/trace.jsonl"
echo "Summary: $OUTPUT_DIR/$RUN_NAME/summary.json"
echo "Curves: $OUTPUT_DIR/$RUN_NAME/curves.csv"
echo "Curves by M: $OUTPUT_DIR/$RUN_NAME/curves_by_initial_M.csv"
echo "Run log: $RUN_LOG_FILE"
