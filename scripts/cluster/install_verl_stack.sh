#!/usr/bin/env bash
set -euo pipefail
set -x

# One-time cluster install for the CUDA training stack. Run from the repo root.
# This intentionally lives outside pyproject dependencies because torch/SGLang/
# veRL versions are CUDA- and cluster-specific.

if [[ -f /etc/profile.d/modules.sh && -d /usr/local/software/cuda/12.1 ]]; then
  # shellcheck disable=SC1091
  . /etc/profile.d/modules.sh
  module purge
  module load rhel8/default-amp
  module load gcc/11
  module load cuda/12.1
fi

if [[ -f scripts/env.sh ]]; then
  set +x
  # shellcheck disable=SC1091
  source scripts/env.sh
  set -x
fi

if [[ -n "${CLUSTER_CUDA_HOME:-}" ]]; then
  export CUDA_HOME="$CLUSTER_CUDA_HOME"
elif [[ -d /usr/local/cuda ]]; then
  export CUDA_HOME=/usr/local/cuda
elif [[ -d /usr/local/software/cuda/12.1 ]]; then
  export CUDA_HOME=/usr/local/software/cuda/12.1
else
  echo "Could not find CUDA. Set CLUSTER_CUDA_HOME to the CUDA install path." >&2
  exit 1
fi
export CUDA_PATH="$CUDA_HOME"
export PATH="${LOCAL_BIN_DIR:-$HOME/.local/bin}:$CUDA_HOME/bin:$PATH"
GCC_LIBSTDCPP_DIR="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBSTDCPP_DIR:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH CUDA_INC_PATH

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not on PATH. Install uv or add it to PATH before running this script." >&2
  exit 1
fi

export UV_PYTHON_PREFERENCE="${UV_PYTHON_PREFERENCE:-only-managed}"

retry_cmd() {
  local attempt=1
  local attempts="${UV_INSTALL_RETRIES:-3}"
  local delay_s="${UV_INSTALL_RETRY_DELAY_S:-20}"
  while true; do
    if "$@"; then
      return 0
    fi
    if [[ "$attempt" -ge "$attempts" ]]; then
      return 1
    fi
    echo "Command failed; retrying in $((delay_s * attempt))s ($attempt/$attempts): $*" >&2
    sleep "$((delay_s * attempt))"
    attempt="$((attempt + 1))"
  done
}

check_python_headers() {
  "$VENV_DIR/bin/python" - <<'PY'
import pathlib
import sys
import sysconfig

header = pathlib.Path(sysconfig.get_paths()["include"]) / "Python.h"
if not header.exists():
    print(f"Missing Python development header: {header}", file=sys.stderr)
    print(
        "Rebuild the venv with uv's managed Python: "
        "UV_PYTHON_PREFERENCE=only-managed RECREATE_VENV=1 "
        "bash scripts/cluster/install_verl_stack.sh",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(f"Python.h ok: {header}")
PY
}

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
if [[ "${RECREATE_VENV:-0}" == "1" && -d "$VENV_DIR" ]]; then
  rm -rf "$VENV_DIR"
fi

if [[ -x "$VENV_DIR/bin/python" ]]; then
  VENV_PYTHON_VERSION="$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$VENV_PYTHON_VERSION" != "$PYTHON_VERSION" ]]; then
    echo "Existing venv uses Python $VENV_PYTHON_VERSION, but PYTHON_VERSION=$PYTHON_VERSION." >&2
    echo "Set RECREATE_VENV=1 to rebuild $VENV_DIR, or remove it manually." >&2
    exit 1
  fi
fi

retry_cmd uv python install "$PYTHON_VERSION"
retry_cmd uv sync --python "$PYTHON_VERSION" --extra verl

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
check_python_headers
# shellcheck disable=SC1091
source scripts/cluster/prepend_venv_cuda_libs.sh

retry_cmd uv pip install -U "ray[data,train,tune,serve]"

VERL_SRC="${VERL_SRC:-$CACHE_ROOT/src/verl}"
if [[ ! -d "$VERL_SRC/.git" ]]; then
  mkdir -p "$(dirname "$VERL_SRC")"
  git clone https://github.com/verl-project/verl.git "$VERL_SRC"
else
  git -C "$VERL_SRC" pull --ff-only
fi

retry_cmd uv pip install -e "$VERL_SRC[sglang]"

SGLANG_SPEC="${SGLANG_SPEC:-sglang==0.5.8}"
retry_cmd uv pip install "$SGLANG_SPEC"
retry_cmd uv pip install "pyarrow>=16.0.0,<21.0.0"
retry_cmd uv pip install -U "torchao>=0.16.0"

# New installs may add nvidia/*/lib directories.
# shellcheck disable=SC1091
source scripts/cluster/prepend_venv_cuda_libs.sh

FLASH_ATTN_SPEC="${FLASH_ATTN_SPEC:-flash-attn}"
if [[ "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
  retry_cmd uv pip install -U packaging psutil ninja
  retry_cmd env MAX_JOBS="${MAX_JOBS:-4}" uv pip install --no-build-isolation "$FLASH_ATTN_SPEC"
fi

retry_cmd uv pip install "datasets==2.20.0" "fsspec==2024.5.0"

python - <<'PY'
import os
import importlib.util
import torch
import pyarrow
import datasets

for name in ["verl", "ray", "sglang", "torch", "datasets", "pyarrow"]:
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"missing {name}")
if importlib.util.find_spec("sglang.srt") is None:
    raise SystemExit("missing sglang.srt runtime module")
if os.environ.get("INSTALL_FLASH_ATTN") == "1" and importlib.util.find_spec("flash_attn") is None:
    raise SystemExit("missing flash_attn")

print("torch", torch.__version__)
print("pyarrow", pyarrow.__version__)
print("datasets", datasets.__version__)
print("torch cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device count", torch.cuda.device_count())
    print("cuda device 0", torch.cuda.get_device_name(0))
PY
