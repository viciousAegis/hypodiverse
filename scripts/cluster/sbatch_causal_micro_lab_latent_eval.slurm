#!/bin/bash
#SBATCH -J cml-latent-eval
#SBATCH -A MLMI-as3727-SL2-GPU
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -p ampere

set -euo pipefail
set -x

if [[ "$#" -lt 1 ]]; then
  echo "Usage: sbatch $0 MERGED_HF_MODEL_PATH [RUN_NAME] [eval args ...]" >&2
  exit 2
fi
MODEL_PATH="$1"
shift
RUN_NAME="${1:-causal_micro_lab_final_k16_latent}"
if [[ "$#" -gt 0 ]]; then
  shift
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Merged Hugging Face model not found at: $MODEL_PATH" >&2
  exit 1
fi

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

export MODEL_PATH RUN_NAME
export EVAL_CONFIG="configs/verl/eval/causal_micro_lab_final_k16_latent.yaml"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export SGLANG_PORT="${SGLANG_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"

scripts/cluster/run_causal_micro_lab_eval_openai.sh "$@"
