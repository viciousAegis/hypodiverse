#!/usr/bin/env bash
set -euo pipefail
set -x

# One-time cluster install for the CUDA training stack. Run from the repo root.
# This intentionally lives outside pyproject dependencies because torch/SGLang/
# veRL versions are CUDA- and cluster-specific.

if [[ -f /etc/profile.d/modules.sh ]]; then
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

export CUDA_HOME="${CUDA_HOME:-/usr/local/software/cuda/12.1}"
export CUDA_PATH="$CUDA_HOME"
export PATH="/home/as3727/.local/bin:$CUDA_HOME/bin:$PATH"
GCC_LIBSTDCPP_DIR="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBSTDCPP_DIR:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH CUDA_INC_PATH

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not on PATH. Install uv or add it to PATH before running this script." >&2
  exit 1
fi

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
if [[ -x "$VENV_DIR/bin/python" ]]; then
  VENV_PYTHON_VERSION="$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$VENV_PYTHON_VERSION" != "$PYTHON_VERSION" ]]; then
    if [[ "${RECREATE_VENV:-0}" == "1" ]]; then
      rm -rf "$VENV_DIR"
    else
      echo "Existing venv uses Python $VENV_PYTHON_VERSION, but PYTHON_VERSION=$PYTHON_VERSION." >&2
      echo "Set RECREATE_VENV=1 to rebuild $VENV_DIR, or remove it manually." >&2
      exit 1
    fi
  fi
fi

uv sync --python "$PYTHON_VERSION" --extra verl

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1}"

if [[ "${INSTALL_TORCH:-1}" == "1" ]]; then
  # shellcheck disable=SC2086
  uv pip install --index-url "$PYTORCH_INDEX_URL" $TORCH_SPEC
fi

uv pip install -U "ray[data,train,tune,serve]"

VERL_SRC="${VERL_SRC:-$CACHE_ROOT/src/verl}"
if [[ ! -d "$VERL_SRC/.git" ]]; then
  mkdir -p "$(dirname "$VERL_SRC")"
  git clone https://github.com/verl-project/verl.git "$VERL_SRC"
else
  git -C "$VERL_SRC" pull --ff-only
fi

uv pip install -e "$VERL_SRC[sglang]"

python - <<'PY'
import importlib.util
import torch

for name in ["verl", "ray", "sglang", "torch"]:
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"missing {name}")

print("torch", torch.__version__)
print("torch cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device count", torch.cuda.device_count())
    print("cuda device 0", torch.cuda.get_device_name(0))
PY
