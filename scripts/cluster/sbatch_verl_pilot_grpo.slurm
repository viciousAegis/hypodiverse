#!/usr/bin/env bash
#SBATCH --job-name=sd-pilot-grpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -xeuo pipefail

# Submit from the project root:
#   sbatch scripts/cluster/sbatch_verl_pilot_grpo.slurm

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

export NNODES="${SLURM_NNODES:-1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-scattered_pilot_qwen3_4b_${SLURM_JOB_ID:-manual}}"

scripts/cluster/run_verl_pilot_grpo.sh "$@"
