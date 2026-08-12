#!/bin/bash
#SBATCH -J hypodiverse-eval
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

if [[ "$#" -ne 1 ]]; then
  echo "Usage: sbatch $0 {base|grpo|lifpo}" >&2
  exit 2
fi

METHOD="$1"
case "$METHOD" in
  base)
    export MODEL_ID="Qwen/Qwen3-4B"
    export MODEL_BASENAME="Qwen3-4B"
    export EVAL_CONFIG="configs/verl/eval/hypodiverse_base.yaml"
    export RUN_NAME="hypodiverse_base_final"
    ;;
  grpo)
    export MODEL_ID="viciousa3gis/hypodiverse-grpo"
    export MODEL_BASENAME="hypodiverse-grpo"
    export EVAL_CONFIG="configs/verl/eval/hypodiverse_grpo.yaml"
    export RUN_NAME="hypodiverse_grpo_final"
    ;;
  lifpo)
    export MODEL_ID="viciousa3gis/hypodiverse-lifpo"
    export MODEL_BASENAME="hypodiverse-lifpo"
    export EVAL_CONFIG="configs/verl/eval/hypodiverse_lifpo.yaml"
    export RUN_NAME="hypodiverse_lifpo_final"
    ;;
  *)
    echo "METHOD must be base, grpo, or lifpo; got: $METHOD" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/sbatch_causal_micro_lab_final_eval.slurm"
