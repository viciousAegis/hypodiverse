#!/bin/bash
#SBATCH -J cml-v3-base
#SBATCH -A MLMI-as3727-SL2-GPU
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -p ampere

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-$REPO_ROOT}"

export EVAL_CONFIG="configs/verl/eval/causal_micro_lab_final_k16_base_v3.yaml"
export RUN_NAME="${RUN_NAME:-causal_micro_lab_final_v3_k16_qwen3_4b_base}"

bash "$SCRIPT_DIR/sbatch_causal_micro_lab_final_eval.slurm" "$@"
