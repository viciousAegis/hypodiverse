#!/usr/bin/env bash
set -euo pipefail

# Build causal micro-lab frozen rows on the cluster when they are not already
# present. This is intentionally separate from scattered-causal dataset prep:
# causal micro-lab rows are fully enumerable and generated locally from code.

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python
fi

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

dataset_output_dir="${CML_DATASET_OUTPUT_DIR:-data/causal_micro_lab/pilot}"
dataset_preset="${CML_DATASET_PRESET:-pilot}"
dataset_seed="${CML_DATASET_SEED:-1}"
eval_output_dir="${CML_EVAL_OUTPUT_DIR:-}"
eval_preset="${CML_EVAL_PRESET:-canonical_eval}"
eval_seed="${CML_EVAL_SEED:-1}"
target_counts="${CML_TARGET_COUNTS:-4,8,16}"
progress_every="${CML_PROGRESS_EVERY:-128}"

build_dataset() {
  local label="$1"
  local preset="$2"
  local output_dir="$3"
  local seed="$4"

  if [[ -z "$output_dir" ]]; then
    return 0
  fi

  if [[ -f "$output_dir/verl_train.jsonl" && -f "$output_dir/verl_val.jsonl" && -f "$output_dir/verl_test.jsonl" ]]; then
    echo "[causal-micro-lab] $label dataset already exists at $output_dir"
    return 0
  fi

  echo "[causal-micro-lab] building $label dataset at $output_dir (preset=$preset seed=$seed)"
  "$PYTHON_BIN" -c '
import sys
from scattered_discovery.envs.causal_micro_lab.cli import build_split_dataset_main

sys.argv = [
    "causal-micro-lab-build-split-dataset",
    "--preset", sys.argv[1],
    "--output-dir", sys.argv[2],
    "--seed", sys.argv[3],
    "--progress-every", sys.argv[4],
    "--target-counts", sys.argv[5],
]
build_split_dataset_main()
' "$preset" "$output_dir" "$seed" "$progress_every" "$target_counts"
}

build_dataset "train/pilot" "$dataset_preset" "$dataset_output_dir" "$dataset_seed"

if [[ -n "$eval_output_dir" && "$eval_output_dir" != "$dataset_output_dir" ]]; then
  build_dataset "canonical eval" "$eval_preset" "$eval_output_dir" "$eval_seed"
fi
