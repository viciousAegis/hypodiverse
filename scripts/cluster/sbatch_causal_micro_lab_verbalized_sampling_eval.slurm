#!/bin/bash
#SBATCH -J cml-vs-k4
#SBATCH -A MLMI-as3727-SL2-GPU
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=01:15:00
#SBATCH --mail-type=END,FAIL
#SBATCH -p ampere

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/cluster/sbatch_causal_micro_lab_verbalized_sampling_eval.slurm \
    TRAINING_RUN_OR_MODEL [K] [RUN_NAME] [extra evaluator arguments...]

Examples:
  # Resolve the latest complete checkpoint, merging it only if necessary.
  sbatch scripts/cluster/sbatch_causal_micro_lab_verbalized_sampling_eval.slurm \
    causal_micro_lab_cluster_validity_grpo_r1 4

  # Evaluate K=8 from the same checkpoint.
  sbatch scripts/cluster/sbatch_causal_micro_lab_verbalized_sampling_eval.slurm \
    causal_micro_lab_cluster_validity_grpo_r1 8

  # Evaluate a Hugging Face base model.
  sbatch scripts/cluster/sbatch_causal_micro_lab_verbalized_sampling_eval.slurm \
    Qwen/Qwen3-4B causal_micro_lab_vs_k4_base

  # An already merged local model is also accepted.
  sbatch scripts/cluster/sbatch_causal_micro_lab_verbalized_sampling_eval.slurm \
    /path/to/merged_hf causal_micro_lab_vs_k4_validity_grpo
EOF
}

if [[ "$#" -lt 1 ]]; then
  usage >&2
  exit 2
fi

MODEL_SOURCE="$1"
shift
VS_K=4
if [[ "${1:-}" == "4" || "${1:-}" == "8" ]]; then
  VS_K="$1"
  shift
fi
REQUESTED_RUN_NAME="${1:-}"
if [[ "$#" -gt 0 ]]; then
  shift
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

RUN_ROOT="$CHECKPOINT_ROOT/scattered-discovery/$MODEL_SOURCE"
if [[ -d "$RUN_ROOT" ]]; then
  TRAINING_RUN="$MODEL_SOURCE"
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

  export MODEL_PATH
  export DOWNLOAD_MODEL=0
  MODEL_LABEL="${TRAINING_RUN}_global_step_${STEP}"
elif [[ -f "$MODEL_SOURCE/config.json" ]]; then
  export MODEL_PATH="$MODEL_SOURCE"
  export DOWNLOAD_MODEL=0
  MODEL_LABEL="$(basename "$MODEL_SOURCE")"
else
  unset MODEL_PATH
  export MODEL_ID="$MODEL_SOURCE"
  export DOWNLOAD_MODEL=1
  MODEL_LABEL="${MODEL_SOURCE##*/}"
fi

if [[ -n "$REQUESTED_RUN_NAME" ]]; then
  export RUN_NAME="$REQUESTED_RUN_NAME"
else
  SAFE_MODEL_LABEL="${MODEL_LABEL//[^A-Za-z0-9_.-]/_}"
  export RUN_NAME="causal_micro_lab_vs_k${VS_K}_${SAFE_MODEL_LABEL}"
fi

export EVAL_CONFIG="configs/verl/eval/causal_micro_lab_verbalized_sampling_k${VS_K}.yaml"
export PYTHON_BIN=python
export SGLANG_PORT="$((20000 + SLURM_JOB_ID % 20000))"

echo "Model source: $MODEL_SOURCE"
if [[ -n "${TRAINING_RUN:-}" ]]; then
  echo "Checkpoint:   global_step_$STEP"
fi
echo "Resolved model: ${MODEL_PATH:-$MODEL_ID}"
echo "Answer count: $VS_K"
echo "W&B run:     $RUN_NAME"
echo "Eval config: $EVAL_CONFIG"

scripts/cluster/run_causal_micro_lab_eval_openai.sh "$@"
