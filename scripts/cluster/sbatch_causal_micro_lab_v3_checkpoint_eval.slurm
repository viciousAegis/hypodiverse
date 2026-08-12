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
    TRAINING_RUN {standard|lifpo|latent} [CHECKPOINT_STEP]

The latest complete actor checkpoint is selected when CHECKPOINT_STEP is
omitted. FSDP shards are merged once and then reused. The 'latent' mode is a
deprecated alias for 'lifpo'.
EOF
  exit 2
fi

TRAINING_RUN="$1"
PUBLIC_MODE="$2"
CHECKPOINT_STEP="${3:-}"
case "$PUBLIC_MODE" in
  standard)
    INTERNAL_MODE="standard"
    export EVAL_CONFIG="configs/verl/eval/causal_micro_lab_final_k16_base_v3.yaml"
    export EVAL_LABEL="final_v3_k16"
    ;;
  lifpo)
    INTERNAL_MODE="latent"
    export EVAL_CONFIG="configs/verl/eval/causal_micro_lab_final_k16_lifpo_v3.yaml"
    export EVAL_LABEL="final_v3_k16_lifpo"
    ;;
  latent)
    echo "EVAL_MODE 'latent' is deprecated; use 'lifpo'." >&2
    INTERNAL_MODE="latent"
    export EVAL_CONFIG="configs/verl/eval/causal_micro_lab_final_k16_lifpo_v3.yaml"
    export EVAL_LABEL="final_v3_k16_lifpo"
    ;;
  *)
    echo "EVAL_MODE must be 'standard', 'lifpo', or 'latent', got: $PUBLIC_MODE" >&2
    exit 2
    ;;
esac

ARGS=("$TRAINING_RUN" "$INTERNAL_MODE")
if [[ -n "$CHECKPOINT_STEP" ]]; then
  ARGS+=("$CHECKPOINT_STEP")
fi
bash "$SCRIPT_DIR/sbatch_causal_micro_lab_checkpoint_eval.slurm" "${ARGS[@]}"
