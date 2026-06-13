#!/usr/bin/env bash
set -euo pipefail

# Create the project uv environment if needed, then verify the training stack.
# Full veRL/SGLang CUDA installs are intentionally not hidden here; if the stack
# is absent, fail before launching Ray/veRL with a clear package list.

if [[ -f "${REPO_DIR:-$PWD}/scripts/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR:-$PWD}/scripts/env.sh"
elif [[ -f scripts/env.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/env.sh
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  if [[ "${BOOTSTRAP_UV_ENV:-1}" != "1" ]]; then
    echo "Missing venv at $VENV_DIR and BOOTSTRAP_UV_ENV is not 1." >&2
    exit 1
  fi
  if ! command -v uv >/dev/null 2>&1; then
    echo "Missing uv. Install uv or create $VENV_DIR before submitting." >&2
    exit 1
  fi
  uv sync --extra verl
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python - <<'PY'
import importlib.util
import os
import sys

backend = os.environ.get("INFER_BACKEND", "sglang")
required = [
    "scattered_discovery",
    "pandas",
    "pyarrow",
    "wandb",
    "verl",
    "torch",
    "ray",
]
if backend == "sglang":
    required.append("sglang")
elif backend == "vllm":
    required.append("vllm")

missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("Python environment is missing required training packages:", file=sys.stderr)
    for name in missing:
        print(f"  - {name}", file=sys.stderr)
    print("", file=sys.stderr)
    print(
        "The bootstrap created/checked the project uv environment, but the full "
        "veRL rollout stack must already be installed in that environment. "
        "Install veRL with the desired SGLang/vLLM backend in VENV_DIR, then "
        "resubmit the job.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
