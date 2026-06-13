#!/usr/bin/env bash
#SBATCH --job-name=sd-smoke-grpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -xeuo pipefail

# Submit from the project root:
#   sbatch scripts/cluster/sbatch_verl_smoke_grpo.slurm

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

export NNODES="${SLURM_NNODES:-1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-scattered_smoke_qwen3_4b_${SLURM_JOB_ID:-manual}}"

scripts/cluster/run_verl_smoke_grpo.sh "$@"
