#!/usr/bin/env bash
set -xeuo pipefail

# Cheap end-to-end cluster check: dataset generation, SGLang rollout, GRPO update,
# checkpointing, validation, and W&B logging.

if [[ -f scripts/env.sh ]]; then
  set +x
  # shellcheck disable=SC1091
  source scripts/env.sh
  set -x
fi

RUN_CONFIG="${RUN_CONFIG:-configs/verl/runs/scattered_smoke.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python
fi
CONFIG_EXPORTS="$("$PYTHON_BIN" scripts/cluster/load_run_config.py "$RUN_CONFIG")"
eval "$CONFIG_EXPORTS"

DATASET_CONFIG="${DATASET_CONFIG:?missing DATASET_CONFIG}"
if [[ "${PREPARE_DATASETS:-1}" == "1" ]]; then
  DATASET_CONFIG="$DATASET_CONFIG" scripts/cluster/prepare_verl_datasets.sh
fi

# shellcheck disable=SC1091
source scripts/cluster/resolve_model_path.sh

EXPERIMENT_NAME_PREFIX="${EXPERIMENT_NAME_PREFIX:-scattered_smoke_qwen3_4b}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${EXPERIMENT_NAME_PREFIX}_$(date +%Y%m%d_%H%M)}"

scripts/cluster/run_verl_discovery_grpo.sh "$@"
