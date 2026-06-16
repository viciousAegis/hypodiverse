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

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
if [[ -x "$VENV_DIR/bin/python" ]]; then
  VENV_PYTHON_VERSION="$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$VENV_PYTHON_VERSION" != "$PYTHON_VERSION" ]]; then
    echo "Existing venv uses Python $VENV_PYTHON_VERSION, but PYTHON_VERSION=$PYTHON_VERSION." >&2
    echo "Run: RECREATE_VENV=1 bash scripts/cluster/install_verl_stack.sh" >&2
    exit 1
  fi
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
  export UV_PYTHON_PREFERENCE="${UV_PYTHON_PREFERENCE:-only-managed}"
  uv python install "$PYTHON_VERSION"
  uv sync --python "$PYTHON_VERSION" --extra verl
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
if [[ -f "${REPO_DIR:-$PWD}/scripts/cluster/prepend_venv_cuda_libs.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR:-$PWD}/scripts/cluster/prepend_venv_cuda_libs.sh"
elif [[ -f scripts/cluster/prepend_venv_cuda_libs.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/cluster/prepend_venv_cuda_libs.sh
fi

python - <<'PY'
import pathlib
import sys
import sysconfig

header = pathlib.Path(sysconfig.get_paths()["include"]) / "Python.h"
if not header.exists():
    print(f"Missing Python development header: {header}", file=sys.stderr)
    print(
        "SGLang/Triton needs Python.h at runtime for JIT helper compilation. "
        "Rebuild the venv with: UV_PYTHON_PREFERENCE=only-managed "
        "RECREATE_VENV=1 bash scripts/cluster/install_verl_stack.sh",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(f"Python.h ok: {header}")
PY

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
    required.append("sglang.srt")
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
