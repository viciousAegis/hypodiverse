#!/bin/bash
#SBATCH -J sd-eval-2gpu-util64
#SBATCH -A MLMI-as3727-SL2-GPU
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE
#SBATCH -p ampere

set -euo pipefail
set -x

export EVAL_CONFIG="${EVAL_CONFIG:-configs/verl/eval/scattered_pilot_base_pass4_util64_2gpu.yaml}"
export EVAL_NUM_SHARDS="${EVAL_NUM_SHARDS:-2}"
export EVAL_WORKERS="${EVAL_WORKERS:-32}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/sbatch_scattered_eval_sharded.slurm" "$@"
