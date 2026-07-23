#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EVAL_CONFIG="${EVAL_CONFIG:-configs/verl/eval/causal_micro_lab_multi_answer_k4_base.yaml}"
export CML_EVAL_OUTPUT_DIR="${CML_EVAL_OUTPUT_DIR:-}"
export THINK="${THINK:-true}"

exec "$SCRIPT_DIR/run_causal_micro_lab_eval.sh" "$@"
