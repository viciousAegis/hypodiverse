#!/bin/bash
#SBATCH -J cml-ckpt-eval
#SBATCH -A MLMI-as3727-SL2-GPU
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -p ampere

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/cluster/sbatch_causal_micro_lab_checkpoint_eval.slurm \
    TRAINING_RUN {standard|latent}

The launcher selects the latest complete actor checkpoint, merges its FSDP
shards into Hugging Face format, and evaluates K=16 on final_v1. K=4,8,12
metrics are derived from the same completion bank.
EOF
}

if [[ "$#" -ne 2 ]]; then
  usage >&2
  exit 2
fi

TRAINING_RUN="$1"
EVAL_MODE="$2"
case "$EVAL_MODE" in
  standard|latent) ;;
  *)
    echo "EVAL_MODE must be 'standard' or 'latent', got: $EVAL_MODE" >&2
    exit 2
    ;;
esac

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load gcc/11
module load cuda/12.1

SBATCH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$SBATCH_SCRIPT_DIR/../.." && pwd)}}"
cd "$BOOTSTRAP_REPO_DIR"
mkdir -p logs

# shellcheck disable=SC1091
source scripts/env.sh
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

RUN_ROOT="$CHECKPOINT_ROOT/scattered-discovery/$TRAINING_RUN"
if [[ ! -d "$RUN_ROOT" ]]; then
  echo "Training run not found: $RUN_ROOT" >&2
  exit 1
fi

STEP=""
TRACKER="$RUN_ROOT/latest_checkpointed_iteration.txt"
if [[ -f "$TRACKER" ]]; then
  TRACKED_STEP="$(tr -d '[:space:]' < "$TRACKER")"
  if [[ "$TRACKED_STEP" =~ ^[0-9]+$ ]]; then
    TRACKED_ACTOR="$RUN_ROOT/global_step_$TRACKED_STEP/actor"
    if compgen -G "$TRACKED_ACTOR/model_world_size_*_rank_*.pt" >/dev/null; then
      STEP="$TRACKED_STEP"
    fi
  fi
fi

if [[ -z "$STEP" ]]; then
  while IFS= read -r candidate; do
    actor_dir="$candidate/actor"
    if compgen -G "$actor_dir/model_world_size_*_rank_*.pt" >/dev/null; then
      STEP="${candidate##*/global_step_}"
      break
    fi
  done < <(
    find "$RUN_ROOT" -maxdepth 1 -type d -name 'global_step_*' -print \
      | sort -Vr
  )
fi

if [[ -z "$STEP" ]]; then
  echo "No complete actor checkpoint found under: $RUN_ROOT" >&2
  exit 1
fi

ACTOR_DIR="$RUN_ROOT/global_step_$STEP/actor"
if [[ ! -f "$ACTOR_DIR/huggingface/config.json" ]]; then
  echo "Checkpoint lacks Hugging Face metadata: $ACTOR_DIR/huggingface/config.json" >&2
  exit 1
fi

MODEL_PATH="$MODEL_ROOT/eval_checkpoints/${TRAINING_RUN}_global_step_${STEP}_hf"
if [[ ! -f "$MODEL_PATH/config.json" ]] || {
  [[ ! -f "$MODEL_PATH/model.safetensors" ]] &&
    [[ ! -f "$MODEL_PATH/model.safetensors.index.json" ]]
}; then
  mkdir -p "$(dirname "$MODEL_PATH")"
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$ACTOR_DIR" \
    --target_dir "$MODEL_PATH"
else
  echo "Reusing merged model: $MODEL_PATH"
fi

RUN_NAME="${TRAINING_RUN}_global_step_${STEP}_final_k16"
if [[ "$EVAL_MODE" == "latent" ]]; then
  EVAL_CONFIG="configs/verl/eval/causal_micro_lab_final_k16_latent.yaml"
  LATENT_COUNT=8
else
  EVAL_CONFIG="configs/verl/eval/causal_micro_lab_final_k16_base.yaml"
  LATENT_COUNT=0
fi

export MODEL_PATH RUN_NAME EVAL_CONFIG LATENT_COUNT
export DOWNLOAD_MODEL=0
export PYTHON_BIN=python
export SGLANG_PORT="$((20000 + SLURM_JOB_ID % 20000))"

echo "Training run: $TRAINING_RUN"
echo "Checkpoint:   global_step_$STEP"
echo "Eval mode:    $EVAL_MODE"
echo "Merged model: $MODEL_PATH"
echo "W&B run:      $RUN_NAME"

scripts/cluster/run_causal_micro_lab_eval_openai.sh
