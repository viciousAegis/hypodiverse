#!/bin/bash
#SBATCH -J cml-v3-ckpt
#SBATCH -A MLMI-as3727-SL2-GPU
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -p ampere

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-$REPO_ROOT}"

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  cat >&2 <<'EOF'
Usage:
  sbatch scripts/cluster/sbatch_causal_micro_lab_v3_checkpoint_eval.slurm \
    TRAINING_RUN {standard|latent} [CHECKPOINT_STEP]

The latest complete actor checkpoint is selected when CHECKPOINT_STEP is
omitted. FSDP shards are merged once and then reused.
EOF
  exit 2
fi

if [[ "$2" == "latent" ]]; then
  export EVAL_CONFIG="configs/verl/eval/causal_micro_lab_final_k16_latent_v3.yaml"
else
  export EVAL_CONFIG="configs/verl/eval/causal_micro_lab_final_k16_base_v3.yaml"
fi
export EVAL_LABEL="final_v3_k16"

bash "$SCRIPT_DIR/sbatch_causal_micro_lab_checkpoint_eval.slurm" "$@"
