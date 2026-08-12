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

echo "Submitting frozen HypoDiverse evaluations (one GPU each):"
submit base "$SCRIPT_DIR/sbatch_causal_micro_lab_released_eval.slurm" base
submit grpo "$SCRIPT_DIR/sbatch_causal_micro_lab_released_eval.slurm" grpo
submit lifpo "$SCRIPT_DIR/sbatch_causal_micro_lab_released_eval.slurm" lifpo

cat <<'EOF'

Each job logs live metrics to W&B. B=4,8,12 are stable prefixes of the same
B=16 generation bank. See docs/causal_micro_lab_reproducibility.md for report
generation.
EOF
