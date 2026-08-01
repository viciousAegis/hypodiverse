#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

submit() {
  local label="$1"
  shift
  local result
  result="$(sbatch --parsable "$@")"
  printf '%-10s %s\n' "$label" "$result"
}

echo "Submitting frozen final_v3 K=16 evaluations (one GPU each):"
submit base \
  "$SCRIPT_DIR/sbatch_causal_micro_lab_v3_base_eval.slurm"
submit validity \
  "$SCRIPT_DIR/sbatch_causal_micro_lab_v3_checkpoint_eval.slurm" \
  causal_micro_lab_cluster_validity_grpo_r1 standard 90
submit ips \
  "$SCRIPT_DIR/sbatch_causal_micro_lab_v3_checkpoint_eval.slurm" \
  causal_micro_lab_cluster_ips_grpo_v1_eps02_r1 standard 60
submit latent \
  "$SCRIPT_DIR/sbatch_causal_micro_lab_v3_checkpoint_eval.slurm" \
  causal_micro_lab_cluster_latent_ips_grpo_v2_fulltraj_k8_r1 latent 55

cat <<'EOF'

Each job logs live metrics to W&B project scattered-discovery. K=4,8,12 are
stable prefixes of the same K=16 bank. Run the comparison command documented in
docs/causal_micro_lab_v3_cluster_eval.md after all four jobs complete.
EOF
