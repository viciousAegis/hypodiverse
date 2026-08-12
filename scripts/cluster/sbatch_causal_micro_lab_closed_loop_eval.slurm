#!/bin/bash
#SBATCH -J hypodiverse-loop
#SBATCH -A MLMI-as3727-SL2-GPU
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -p ampere

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: sbatch $0 {base|grpo|lifpo}" >&2
  exit 2
fi

METHOD="$1"
case "$METHOD" in
  base)
    export MODEL_ID="Qwen/Qwen3-4B"
    export MODEL_BASENAME="Qwen3-4B"
    export RUN_NAME="hypodiverse_closed_loop_base"
    export LATENT_COUNT=0
    ;;
  grpo)
    export MODEL_ID="viciousa3gis/hypodiverse-grpo"
    export MODEL_BASENAME="hypodiverse-grpo"
    export RUN_NAME="hypodiverse_closed_loop_grpo"
    export LATENT_COUNT=0
    ;;
  lifpo)
    export MODEL_ID="viciousa3gis/hypodiverse-lifpo"
    export MODEL_BASENAME="hypodiverse-lifpo"
    export RUN_NAME="hypodiverse_closed_loop_lifpo"
    export LATENT_COUNT=8
    ;;
  *)
    echo "METHOD must be base, grpo, or lifpo; got: $METHOD" >&2
    exit 2
    ;;
esac

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load gcc/11
module load cuda/12.1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}}"
cd "$REPO_DIR"
mkdir -p logs

set +x
# shellcheck disable=SC1091
source scripts/env.sh
set -x

export EVAL_CONFIG="configs/verl/eval/hypodiverse_closed_loop.yaml"
export SGLANG_PORT="${SGLANG_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"

bash scripts/cluster/bootstrap_python_env.sh
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
# shellcheck disable=SC1091
source scripts/cluster/prepend_venv_cuda_libs.sh

exec bash scripts/cluster/run_causal_micro_lab_closed_loop_eval.sh
