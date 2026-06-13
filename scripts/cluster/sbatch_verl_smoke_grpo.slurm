#!/bin/bash
#SBATCH -J sd-grpo-smoke
#SBATCH -A MLMI-as3727-SL2-GPU
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE
#SBATCH -p ampere

set -euo pipefail
set -x

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load gcc/11
module load cuda/12.1

REPO_DIR="${REPO_DIR:-/home/as3727/rds/hpc-work/thesis}"
cd "$REPO_DIR"
mkdir -p logs

set +x
# shellcheck disable=SC1091
source scripts/env.sh
set -x
cd "$REPO_DIR"

export CUDA_HOME="${CUDA_HOME:-/usr/local/software/cuda/12.1}"
export CUDA_PATH="$CUDA_HOME"
export PATH="/home/as3727/.local/bin:$CUDA_HOME/bin:$PATH"
GCC_LIBSTDCPP_DIR="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBSTDCPP_DIR:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH CUDA_INC_PATH

bash scripts/cluster/bootstrap_python_env.sh
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
MODEL_BASENAME="${MODEL_BASENAME:-${MODEL_ID##*/}}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/$MODEL_BASENAME}"
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  if [[ "${DOWNLOAD_MODEL:-1}" == "1" ]]; then
    mkdir -p "$MODEL_PATH"
    if command -v hf >/dev/null 2>&1; then
      hf download "$MODEL_ID" --local-dir "$MODEL_PATH"
    elif command -v huggingface-cli >/dev/null 2>&1; then
      huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_PATH"
    else
      echo "Neither hf nor huggingface-cli is available for model download." >&2
      exit 1
    fi
  else
    MODEL_PATH="$MODEL_ID"
  fi
fi
export MODEL_PATH

export WANDB_PROJECT="${WANDB_PROJECT:-scattered-discovery}"
export WANDB_NAME="${WANDB_NAME:-sd-grpo-smoke-${SLURM_JOB_ID:-manual}}"
export PROJECT_NAME="${PROJECT_NAME:-$WANDB_PROJECT}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$WANDB_NAME}"

export NNODES="${SLURM_NNODES:-1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-2}}"
export ROLLOUT_TP="${ROLLOUT_TP:-${TP:-1}}"

scripts/cluster/run_verl_smoke_grpo.sh "$@"
