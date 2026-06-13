#!/usr/bin/env bash
set -xeuo pipefail

# Run from the project root. Requires `uv sync --extra verl` if writing Parquet.

export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"

DATASET_CONFIG="${DATASET_CONFIG:-configs/verl/datasets/all_envs.yaml}"

uv run --extra verl scattered-discovery-make-dataset --config "$DATASET_CONFIG"
