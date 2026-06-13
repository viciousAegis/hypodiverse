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

SBATCH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$SBATCH_SCRIPT_DIR/../.." && pwd)}}"
cd "$BOOTSTRAP_REPO_DIR"
mkdir -p logs

set +x
# shellcheck disable=SC1091
source scripts/env.sh
set -x
cd "$REPO_DIR"

export CUDA_HOME="${CUDA_HOME:-/usr/local/software/cuda/12.1}"
export CUDA_PATH="$CUDA_HOME"
export PATH="${LOCAL_BIN_DIR:-$HOME/.local/bin}:$CUDA_HOME/bin:$PATH"
GCC_LIBSTDCPP_DIR="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBSTDCPP_DIR:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH CUDA_INC_PATH

bash scripts/cluster/bootstrap_python_env.sh
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

export WANDB_PROJECT="${WANDB_PROJECT:-scattered-discovery}"
export PROJECT_NAME="${PROJECT_NAME:-$WANDB_PROJECT}"

export NNODES="${SLURM_NNODES:-1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-2}}"
export ROLLOUT_TP="${ROLLOUT_TP:-${TP:-1}}"

scripts/cluster/run_verl_smoke_grpo.sh "$@"
