#!/usr/bin/env bash
set -xeuo pipefail

# First meaningful learning run. This targets about 256 actor optimizer updates
# with the default 4-GPU settings.

if [[ -f scripts/env.sh ]]; then
  set +x
  # shellcheck disable=SC1091
  source scripts/env.sh
  set -x
fi

RUN_CONFIG="${RUN_CONFIG:-configs/verl/runs/scattered_pilot.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python
fi
CONFIG_EXPORTS="$("$PYTHON_BIN" scripts/cluster/load_run_config.py "$RUN_CONFIG")"
eval "$CONFIG_EXPORTS"

if [[ "${PREPARE_DATASETS:-1}" == "1" ]]; then
  DATASET_CONFIG="${DATASET_CONFIG:?missing DATASET_CONFIG}"
  DATASET_CONFIG="$DATASET_CONFIG" scripts/cluster/prepare_verl_datasets.sh
fi

if [[ "${CML_GENERATE_DATASET_IF_MISSING:-0}" == "1" ]]; then
  # The preparation script validates the manifest as well as file existence.
  # Always invoke it so changed reward/agent settings cannot silently reuse
  # stale frozen rows; matching datasets return immediately.
  scripts/cluster/prepare_causal_micro_lab_dataset.sh
fi

# shellcheck disable=SC1091
source scripts/cluster/resolve_model_path.sh

EXPERIMENT_NAME_PREFIX="${EXPERIMENT_NAME_PREFIX:-scattered_pilot_qwen3_4b}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${EXPERIMENT_NAME_PREFIX}_$(date +%Y%m%d_%H%M)}"

scripts/cluster/run_verl_discovery_grpo.sh "$@"
