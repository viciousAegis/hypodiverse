```bash
#!/usr/bin/env bash
set -xeuo pipefail

# Run from the project root on the cluster.

if [[ -f scripts/env.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/env.sh
fi

if [[ -f scripts/cluster/resolve_model_path.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/cluster/resolve_model_path.sh
fi

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/sd-ray-${USER:-$(id -u)}-${SLURM_JOB_ID:-local}}"
export WANDB_DIR="${WANDB_DIR:-$PWD/.wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PWD/.wandb/cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$PWD/.wandb/config}"

mkdir -p \
  "$HF_HOME" \
  "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$RAY_TMPDIR" \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" \
  "$WANDB_CONFIG_DIR"

DEVICE=${DEVICE:-gpu}
INFER_BACKEND=${INFER_BACKEND:-sglang}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
TRAIN_FILE=${TRAIN_FILE:-data/verl/hypospace_causal_train.parquet}

format_hydra_list() {
  local joined="["
  local first=1

  for item in "$@"; do
    if [[ "$first" == "1" ]]; then
      joined+="$item"
      first=0
    else
      joined+=",$item"
    fi
  done

  joined+="]"
  printf '%s\n' "$joined"
}

if [[ -n "${VAL_FILES:-}" ]]; then
  RESOLVED_VAL_FILES="$VAL_FILES"
else
  if [[ -z "${VAL_FILE:-}" ]]; then
    CANDIDATE_VAL_FILE="${TRAIN_FILE/_train.parquet/_val.parquet}"

    if [[ "$CANDIDATE_VAL_FILE" != "$TRAIN_FILE" && -f "$CANDIDATE_VAL_FILE" ]]; then
      VAL_FILE="$CANDIDATE_VAL_FILE"
    else
      SPLIT_VAL_PREFIX="${TRAIN_FILE/_train.parquet/_val_}"

      shopt -s nullglob
      SPLIT_VAL_FILES=("${SPLIT_VAL_PREFIX}"*.parquet)
      shopt -u nullglob

      if [[ "$SPLIT_VAL_PREFIX" != "$TRAIN_FILE" && "${#SPLIT_VAL_FILES[@]}" -gt 0 ]]; then
        VAL_FILE="$(format_hydra_list "${SPLIT_VAL_FILES[@]}")"
      else
        VAL_FILE="$TRAIN_FILE"
      fi
    fi
  fi

  RESOLVED_VAL_FILES="$VAL_FILE"
fi

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}

# Fixed per-GPU micro-batches.
#
# Actor training includes backward/optimizer work and therefore uses more
# memory. Rollout/ref log-prob computation is forward-only and can usually
# use a larger micro-batch.
ACTOR_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-2}
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}
ROLLOUT_N=${ROLLOUT_N:-4}

ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-False}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}

RESUME_MODE=${RESUME_MODE:-auto}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-}

MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-3}
MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-3}

PROJECT_NAME=${PROJECT_NAME:-scattered-discovery}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-discovery_grpo_${INFER_BACKEND}_$(date +%Y%m%d_%H%M)}
WANDB_LOGGER=${WANDB_LOGGER:-'["console","wandb"]'}
DISCOVERY_ALGO=${DISCOVERY_ALGO:-grpo}

mapfile -t ALGO < <(
  python3 -m scattered_discovery.algos.cli \
    --algo "${DISCOVERY_ALGO}"
)

DATA=(
  "${ALGO[@]}"
  data.train_files="${TRAIN_FILE}"
  data.val_files="${RESOLVED_VAL_FILES}"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  data.filter_overlong_prompts=True
  data.truncation=error
)

MODEL=(
  actor_rollout_ref.model.path="${MODEL_PATH}"
  +actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}"
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}"
  actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"

  # Dynamic batching currently enters veRL's no-padding path, which imports
  # flash_attn padding utilities. Keep it disabled for this SDPA-only setup.
  actor_rollout_ref.actor.use_dynamic_bsz=False
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ACTOR_MICRO_BATCH_SIZE_PER_GPU}"

  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}"

  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
  actor_rollout_ref.rollout.name="${INFER_BACKEND}"
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}"
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"

  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"

  actor_rollout_ref.rollout.multi_turn.enable=True
  actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable
  actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/verl/agent_loop.yaml
  actor_rollout_ref.rollout.agent.default_agent_loop=discovery_agent_loop
)

REF=(
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
  actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
  trainer.balance_batch=True
  trainer.logger="${WANDB_LOGGER}"
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${EXPERIMENT_NAME}"
  trainer.default_local_dir="${CHECKPOINT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}"

  trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
  trainer.nnodes="${NNODES}"

  trainer.save_freq="${SAVE_FREQ}"
  trainer.test_freq="${TEST_FREQ}"

  trainer.resume_mode="${RESUME_MODE}"
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}"
  trainer.max_critic_ckpt_to_keep="${MAX_CRITIC_CKPT_TO_KEEP}"

  trainer.total_epochs="${TOTAL_EPOCHS}"
)

if [[ -n "$RESUME_FROM_PATH" ]]; then
  TRAINER+=(trainer.resume_from_path="${RESUME_FROM_PATH}")
fi

python3 -m verl.trainer.main_ppo \
  "${DATA[@]}" \
  "${MODEL[@]}" \
  "${ACTOR[@]}" \
  "${ROLLOUT[@]}" \
  "${REF[@]}" \
  "${TRAINER[@]}" \
  "$@"
```
