#!/bin/bash
#SBATCH -J sd-eval-shard
#SBATCH -A MLMI-as3727-SL2-GPU
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --mail-type=NONE
#SBATCH -p ampere

set -euo pipefail
set -x

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load gcc/11
module load cuda/12.1

SBATCH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$SBATCH_SCRIPT_DIR/../.." && pwd)}}"
cd "$BOOTSTRAP_REPO_DIR"
mkdir -p logs

set +x
# shellcheck disable=SC1091
source scripts/env.sh
set -x
cd "$REPO_DIR"

export CUDA_HOME="${CLUSTER_CUDA_HOME:-/usr/local/software/cuda/12.1}"
export CUDA_PATH="$CUDA_HOME"
export PATH="${LOCAL_BIN_DIR:-$HOME/.local/bin}:$CUDA_HOME/bin:$PATH"
GCC_LIBSTDCPP_DIR="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBSTDCPP_DIR:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH CUDA_INC_PATH

bash scripts/cluster/bootstrap_python_env.sh
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
# shellcheck disable=SC1091
source scripts/cluster/prepend_venv_cuda_libs.sh

export WANDB_PROJECT="${WANDB_PROJECT:-scattered-discovery}"
export EVAL_CONFIG="${EVAL_CONFIG:-configs/verl/eval/scattered_pilot_base_pass4_util16.yaml}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"

CONFIG_EXPORTS="$("$PYTHON_BIN" scripts/cluster/load_run_config.py "$EVAL_CONFIG")"
eval "$CONFIG_EXPORTS"

DATASET_CONFIG="${DATASET_CONFIG:?missing DATASET_CONFIG}"
if [[ "${PREPARE_DATASETS:-1}" == "1" ]]; then
  DATASET_CONFIG="$DATASET_CONFIG" scripts/cluster/prepare_verl_datasets.sh
fi

BASE_RUN_NAME="${RUN_NAME:-scattered_eval_$(date +%Y%m%d_%H%M)}"
BASE_PORT="${SGLANG_PORT:-30000}"
EVAL_NUM_SHARDS="${EVAL_NUM_SHARDS:-${SLURM_GPUS_ON_NODE:-4}}"
if ! [[ "$EVAL_NUM_SHARDS" =~ ^[0-9]+$ ]]; then
  EVAL_NUM_SHARDS=4
fi
ORIGINAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
if [[ -n "$ORIGINAL_CUDA_VISIBLE_DEVICES" ]]; then
  IFS=',' read -r -a VISIBLE_DEVICES <<< "$ORIGINAL_CUDA_VISIBLE_DEVICES"
else
  VISIBLE_DEVICES=()
  for ((device = 0; device < EVAL_NUM_SHARDS; device++)); do
    VISIBLE_DEVICES+=("$device")
  done
fi
if (( EVAL_NUM_SHARDS > ${#VISIBLE_DEVICES[@]} )); then
  echo "EVAL_NUM_SHARDS=$EVAL_NUM_SHARDS but only ${#VISIBLE_DEVICES[@]} CUDA devices are visible: ${ORIGINAL_CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 1
fi

PIDS=()
for ((shard = 0; shard < EVAL_NUM_SHARDS; shard++)); do
  port=$((BASE_PORT + shard))
  (
    export CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICES[$shard]}"
    export EVAL_CONFIG
    export PREPARE_DATASETS=0
    export EVAL_NUM_SHARDS
    export EVAL_SHARD_INDEX="$shard"
    export SGLANG_PORT="$port"
    export SGLANG_TP=1
    export RUN_NAME="${BASE_RUN_NAME}_shard${shard}of${EVAL_NUM_SHARDS}"
    unset BASE_URL
    scripts/cluster/run_scattered_eval_openai.sh
  ) &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
