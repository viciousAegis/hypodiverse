#!/usr/bin/env bash
set -xeuo pipefail

# Run from the project root. Requires `uv sync --extra verl` if writing Parquet.

if [[ -f scripts/env.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/env.sh
fi

DATASET_CONFIG="${DATASET_CONFIG:-configs/verl/datasets/all_envs.yaml}"

if python -c "import scattered_discovery, pandas, pyarrow" >/dev/null 2>&1; then
  python -m scattered_discovery.verl.make_dataset --config "$DATASET_CONFIG"
else
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  uv run --extra verl scattered-discovery-make-dataset --config "$DATASET_CONFIG"
fi
