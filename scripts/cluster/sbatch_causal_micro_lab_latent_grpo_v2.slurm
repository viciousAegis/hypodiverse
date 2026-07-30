#!/bin/bash
#SBATCH -J cml-latent-v2
#SBATCH -A MLMI-as3727-SL2-GPU
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --mail-type=NONE
#SBATCH -p ampere

set -euo pipefail
set -x

VARIANT="${1:-combined}"
if [[ "$#" -gt 0 ]]; then
  shift
fi
case "$VARIANT" in
  combined)
    RUN_CONFIG_PATH="configs/verl/runs/causal_micro_lab_cluster_latent_ips_grpo_v2.yaml"
    ;;
  latent-only)
    RUN_CONFIG_PATH="configs/verl/runs/causal_micro_lab_cluster_latent_only_grpo_v2.yaml"
    ;;
  *)
    echo "Usage: sbatch $0 [combined|latent-only] [Hydra overrides ...]" >&2
    exit 2
    ;;
esac

INIT_RUN="causal_micro_lab_cluster_latent_ips_grpo_v1_k8_r1"
INIT_STEP=""
INIT_MODEL=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --init-run)
      INIT_RUN="${2:?--init-run requires a veRL experiment name}"
      shift 2
      ;;
    --init-step)
      INIT_STEP="${2:?--init-step requires a numeric global step}"
      shift 2
      ;;
    --init-model)
      INIT_MODEL="${2:?--init-model requires a merged Hugging Face directory}"
      shift 2
      ;;
    --base-model)
      INIT_RUN=""
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

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

python scripts/cluster/check_ips_grpo_verl.py \
  --latent \
  --run-config "$RUN_CONFIG_PATH"
python scripts/cluster/check_latent_grpo.py

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY was not loaded from $REPO_DIR/.env" >&2
  exit 1
fi
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-scattered-discovery}"
export PROJECT_NAME="${PROJECT_NAME:-$WANDB_PROJECT}"
export NNODES="${SLURM_NNODES:-1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-2}}"
export ROLLOUT_TP="${ROLLOUT_TP:-${TP:-1}}"
export RUN_CONFIG="$RUN_CONFIG_PATH"

if [[ -n "$INIT_MODEL" ]]; then
  MODEL_PATH="$INIT_MODEL"
elif [[ -n "$INIT_RUN" ]]; then
  INIT_ROOT="$CHECKPOINT_ROOT/scattered-discovery/$INIT_RUN"
  if [[ ! -d "$INIT_ROOT" ]]; then
    echo "Missing initialization run: $INIT_ROOT" >&2
    exit 1
  fi
  if [[ -z "$INIT_STEP" ]]; then
    while IFS= read -r candidate; do
      actor_dir="$candidate/actor"
      if compgen -G "$actor_dir/model_world_size_*_rank_*.pt" >/dev/null; then
        INIT_STEP="${candidate##*/global_step_}"
        break
      fi
    done < <(
      find "$INIT_ROOT" -maxdepth 1 -type d -name 'global_step_*' -print \
        | sort -Vr
    )
  fi
  if [[ -z "$INIT_STEP" ]]; then
    echo "No complete actor checkpoint found under: $INIT_ROOT" >&2
    exit 1
  fi
  ACTOR_DIR="$INIT_ROOT/global_step_$INIT_STEP/actor"
  if [[ ! -f "$ACTOR_DIR/huggingface/config.json" ]]; then
    echo "Checkpoint lacks Hugging Face metadata: $ACTOR_DIR" >&2
    exit 1
  fi
  MODEL_PATH="$MODEL_ROOT/training_initializers/${INIT_RUN}_global_step_${INIT_STEP}_hf"
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
    echo "Reusing merged initialization model: $MODEL_PATH"
  fi
fi

if [[ -n "${MODEL_PATH:-}" && ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Initialization model is not a Hugging Face directory: $MODEL_PATH" >&2
  exit 1
fi
export MODEL_PATH
export DOWNLOAD_MODEL=0
unset RESUME_FROM_PATH
echo "Fresh v2 run initialized from: ${MODEL_PATH:-Qwen/Qwen3-4B}"

scripts/cluster/run_verl_pilot_grpo.sh "$@" trainer.use_v1=True
