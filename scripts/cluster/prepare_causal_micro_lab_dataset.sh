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
rebuild_dataset="${CML_REBUILD_DATASET:-0}"

dataset_args=()
if [[ -n "${CML_TRAIN_STATES_PER_COUNT:-}" ]]; then
  dataset_args+=(--train-states-per-count "$CML_TRAIN_STATES_PER_COUNT")
fi
if [[ -n "${CML_VAL_STATES_PER_COUNT:-}" ]]; then
  dataset_args+=(--val-states-per-count "$CML_VAL_STATES_PER_COUNT")
fi
if [[ -n "${CML_TEST_STATES_PER_COUNT:-}" ]]; then
  dataset_args+=(--test-states-per-count "$CML_TEST_STATES_PER_COUNT")
fi
if [[ -n "${CML_TRAIN_MAX_ROWS:-}" ]]; then
  dataset_args+=(--train-max-rows "$CML_TRAIN_MAX_ROWS")
fi
if [[ -n "${CML_VAL_MAX_ROWS:-}" ]]; then
  dataset_args+=(--val-max-rows "$CML_VAL_MAX_ROWS")
fi
if [[ -n "${CML_TEST_MAX_ROWS:-}" ]]; then
  dataset_args+=(--test-max-rows "$CML_TEST_MAX_ROWS")
fi

reward_args=()
if [[ -n "${CML_NONEMPTY_OUTPUT_REWARD:-}" ]]; then
  reward_args+=(--nonempty-output-reward "$CML_NONEMPTY_OUTPUT_REWARD")
fi
if [[ -n "${CML_RULE_MARKER_REWARD:-}" ]]; then
  reward_args+=(--rule-marker-reward "$CML_RULE_MARKER_REWARD")
fi
if [[ -n "${CML_PARSE_VALID_REWARD:-}" ]]; then
  reward_args+=(--parse-valid-reward "$CML_PARSE_VALID_REWARD")
fi
if [[ -n "${CML_SYNTAX_VALID_REWARD:-}" ]]; then
  reward_args+=(--syntax-valid-reward "$CML_SYNTAX_VALID_REWARD")
fi
if [[ -n "${CML_EVIDENCE_CONSISTENT_REWARD:-}" ]]; then
  reward_args+=(--evidence-consistent-reward "$CML_EVIDENCE_CONSISTENT_REWARD")
fi
if [[ -n "${CML_VALID_HYPOTHESIS_REWARD:-}" ]]; then
  reward_args+=(--valid-hypothesis-reward "$CML_VALID_HYPOTHESIS_REWARD")
fi
if [[ -n "${CAUSAL_MICRO_LAB_LENGTH_PENALTY_START:-}" ]]; then
  reward_args+=(--length-penalty-start "$CAUSAL_MICRO_LAB_LENGTH_PENALTY_START")
fi
if [[ -n "${CAUSAL_MICRO_LAB_LENGTH_PENALTY_MAX:-}" ]]; then
  reward_args+=(--length-penalty-max "$CAUSAL_MICRO_LAB_LENGTH_PENALTY_MAX")
fi
if [[ -n "${CAUSAL_MICRO_LAB_MASK_TRUNCATED:-}" ]]; then
  reward_args+=(--mask-truncated "$CAUSAL_MICRO_LAB_MASK_TRUNCATED")
fi

build_dataset() {
  local label="$1"
  local preset="$2"
  local output_dir="$3"
  local seed="$4"

  if [[ -z "$output_dir" ]]; then
    return 0
  fi

  local needs_rebuild="$rebuild_dataset"
  if [[ "$needs_rebuild" != "1" && ( "${#reward_args[@]}" -gt 0 || "${#dataset_args[@]}" -gt 0 ) ]]; then
    if [[ ! -f "$output_dir/manifest.jsonl" ]]; then
      needs_rebuild=1
    elif ! "$PYTHON_BIN" -c '
import json
import os
import sys
from pathlib import Path

def maybe_float(name):
    raw = os.environ.get(name)
    return None if raw in (None, "") else float(raw)

def maybe_bool(name):
    raw = os.environ.get(name)
    return None if raw in (None, "") else raw == "1"

def maybe_int(name):
    raw = os.environ.get(name)
    return None if raw in (None, "") else int(raw)

expected_counts = {
    key: value
    for key, value in {
        "train": maybe_int("CML_TRAIN_STATES_PER_COUNT"),
        "val": maybe_int("CML_VAL_STATES_PER_COUNT"),
        "test": maybe_int("CML_TEST_STATES_PER_COUNT"),
    }.items()
    if value is not None
}
expected_caps = {
    key: value
    for key, value in {
        "train": maybe_int("CML_TRAIN_MAX_ROWS"),
        "val": maybe_int("CML_VAL_MAX_ROWS"),
        "test": maybe_int("CML_TEST_MAX_ROWS"),
    }.items()
    if value is not None
}
expected_task = {
    key: value
    for key, value in {
        "nonempty_output_reward": maybe_float("CML_NONEMPTY_OUTPUT_REWARD"),
        "rule_marker_reward": maybe_float("CML_RULE_MARKER_REWARD"),
        "parse_valid_reward": maybe_float("CML_PARSE_VALID_REWARD"),
        "syntax_valid_reward": maybe_float("CML_SYNTAX_VALID_REWARD"),
        "evidence_consistent_reward": maybe_float("CML_EVIDENCE_CONSISTENT_REWARD"),
        "valid_hypothesis_reward": maybe_float("CML_VALID_HYPOTHESIS_REWARD"),
    }.items()
    if value is not None
}
expected_agent = {
    key: value
    for key, value in {
        "length_penalty_start": maybe_float("CAUSAL_MICRO_LAB_LENGTH_PENALTY_START"),
        "length_penalty_max": maybe_float("CAUSAL_MICRO_LAB_LENGTH_PENALTY_MAX"),
        "mask_truncated": maybe_bool("CAUSAL_MICRO_LAB_MASK_TRUNCATED"),
    }.items()
    if value is not None
}

manifest = Path(sys.argv[1])
rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
latest = rows[-1] if rows else {}
actual_counts = latest.get("states_per_count", {})
actual_caps = latest.get("max_rows_per_split", {})
ok = (
    all(actual_counts.get(key) == value for key, value in expected_counts.items())
    and all(actual_caps.get(key) == value for key, value in expected_caps.items())
    and latest.get("verl_task_overrides", {}) == expected_task
    and latest.get("verl_agent_overrides", {}) == expected_agent
)
raise SystemExit(0 if ok else 1)
' "$output_dir/manifest.jsonl"; then
      needs_rebuild=1
    fi
  fi

  if [[ "$needs_rebuild" != "1" && -f "$output_dir/verl_train.jsonl" && -f "$output_dir/verl_val.jsonl" && -f "$output_dir/verl_test.jsonl" ]]; then
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
    *sys.argv[6:],
]
build_split_dataset_main()
' "$preset" "$output_dir" "$seed" "$progress_every" "$target_counts" "${dataset_args[@]}" "${reward_args[@]}"
}

build_dataset "train/pilot" "$dataset_preset" "$dataset_output_dir" "$dataset_seed"

if [[ -n "$eval_output_dir" && "$eval_output_dir" != "$dataset_output_dir" ]]; then
  build_dataset "canonical eval" "$eval_preset" "$eval_output_dir" "$eval_seed"
fi
