#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
cd "$REPO_DIR"

echo "Repo: $REPO_DIR"
echo "Blackwell scratch root: $BLACKWELL_RUN_ROOT"
echo "Venv: $VENV_DIR"
echo "Data root: $DATA_ROOT"
echo "Model root: $MODEL_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "Missing uv on PATH. Install uv first, then rerun this script." >&2
  exit 1
fi

bash scripts/cluster/bootstrap_python_env.sh

echo "Environment ok."
echo "Activate with: source $VENV_DIR/bin/activate"
