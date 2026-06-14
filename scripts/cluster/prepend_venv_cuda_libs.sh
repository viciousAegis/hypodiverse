#!/usr/bin/env bash
set -euo pipefail

# PyTorch CUDA wheels ship their own CUDA runtime libraries under site-packages.
# Put those first so Ray/SGLang subprocesses do not pick up an older module
# libcudart before importing torch.

if [[ -z "${VENV_DIR:-}" || ! -x "$VENV_DIR/bin/python" ]]; then
  return 0 2>/dev/null || exit 0
fi

mapfile -t _SD_VENV_CUDA_LIBS < <("$VENV_DIR/bin/python" - <<'PY'
from pathlib import Path
import sysconfig

roots = []
for key in ("platlib", "purelib"):
    value = sysconfig.get_paths().get(key)
    if value and value not in roots:
        roots.append(value)

dirs = []
for root_text in roots:
    root = Path(root_text)
    nvidia_root = root / "nvidia"
    preferred = [
        nvidia_root / "cuda_runtime" / "lib",
        root / "torch" / "lib",
    ]
    for path in preferred:
        if path.is_dir() and path not in dirs:
            dirs.append(path)
    if nvidia_root.is_dir():
        for path in sorted(nvidia_root.glob("*/lib")):
            if path.is_dir() and path not in dirs:
                dirs.append(path)

for path in dirs:
    print(path)
PY
)

if [[ "${#_SD_VENV_CUDA_LIBS[@]}" -eq 0 ]]; then
  unset _SD_VENV_CUDA_LIBS
  return 0 2>/dev/null || exit 0
fi

_SD_LD_PREFIX=""
for _SD_LIB_DIR in "${_SD_VENV_CUDA_LIBS[@]}"; do
  case ":${_SD_LD_PREFIX}:${LD_LIBRARY_PATH:-}:" in
    *":$_SD_LIB_DIR:"*) ;;
    *) _SD_LD_PREFIX="${_SD_LD_PREFIX:+$_SD_LD_PREFIX:}$_SD_LIB_DIR" ;;
  esac
done

if [[ -n "$_SD_LD_PREFIX" ]]; then
  export LD_LIBRARY_PATH="$_SD_LD_PREFIX${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

unset _SD_VENV_CUDA_LIBS _SD_LD_PREFIX _SD_LIB_DIR
