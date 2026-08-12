#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cluster/publish_causal_micro_lab_hf.sh \
    --namespace HF_USER_OR_ORG \
    {--dataset-only|--models-only|--all} \
    [--dataset-revision SHA] [--push] [--private]

Modes:
  --dataset-only  Package, validate, and optionally upload the frozen dataset.
  --models-only   Package, validate, and optionally upload both merged models.
  --all           Publish the dataset first, then pin and publish both models.

Without --push, the command only stages and validates artifacts. A model dry
run requires --dataset-revision so its card can pin an immutable dataset commit.
EOF
}

NAMESPACE=""
MODE=""
DATASET_REVISION=""
PUSH=0
PRIVATE=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --namespace)
      NAMESPACE="${2:?--namespace requires a value}"
      shift 2
      ;;
    --dataset-revision)
      DATASET_REVISION="${2:?--dataset-revision requires a value}"
      shift 2
      ;;
    --dataset-only|--models-only|--all)
      if [[ -n "$MODE" ]]; then
        echo "Choose exactly one release mode." >&2
        exit 2
      fi
      MODE="${1#--}"
      shift
      ;;
    --push)
      PUSH=1
      shift
      ;;
    --private)
      PRIVATE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$NAMESPACE" || -z "$MODE" ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$BOOTSTRAP_REPO_DIR"

# shellcheck disable=SC1091
source scripts/env.sh
cd "$REPO_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Python environment not found: $VENV_DIR/bin/python" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if ! python -c 'import huggingface_hub' >/dev/null 2>&1; then
  echo "Python environment is missing required publishing package: huggingface_hub" >&2
  exit 1
fi

COMMON=()
if [[ "$PUSH" -eq 1 ]]; then
  COMMON+=(--push)
else
  COMMON+=(--dry-run)
fi
if [[ "$PRIVATE" -eq 1 ]]; then
  COMMON+=(--private)
fi

DATASET_REPO="$NAMESPACE/hypodiverse"
GRPO_REPO="$NAMESPACE/hypodiverse-grpo"
LIFPO_REPO="$NAMESPACE/hypodiverse-lifpo"

echo "Repository: $REPO_DIR"
echo "Model root: $MODEL_ROOT"
echo "Dataset repository: $DATASET_REPO"
echo "This is a CPU/network release task; no GPU is required after merging."

case "$MODE" in
  dataset-only)
    exec python -m scattered_discovery.release.causal_micro_lab dataset \
      --repo-root "$REPO_DIR" \
      --repo-id "$DATASET_REPO" \
      "${COMMON[@]}"
    ;;
  models-only)
    if [[ -z "$DATASET_REVISION" ]]; then
      echo "--models-only requires --dataset-revision SHA" >&2
      exit 2
    fi
    python -m scattered_discovery.release.causal_micro_lab model \
      --repo-root "$REPO_DIR" \
      --model-root "$MODEL_ROOT" \
      --method grpo \
      --dataset-repo-id "$DATASET_REPO" \
      --dataset-revision "$DATASET_REVISION" \
      --repo-id "$GRPO_REPO" \
      "${COMMON[@]}"
    exec python -m scattered_discovery.release.causal_micro_lab model \
      --repo-root "$REPO_DIR" \
      --model-root "$MODEL_ROOT" \
      --method lifpo \
      --dataset-repo-id "$DATASET_REPO" \
      --dataset-revision "$DATASET_REVISION" \
      --repo-id "$LIFPO_REPO" \
      "${COMMON[@]}"
    ;;
  all)
    ARGS=(
      all
      --repo-root "$REPO_DIR"
      --model-root "$MODEL_ROOT"
      --dataset-repo-id "$DATASET_REPO"
      --grpo-repo-id "$GRPO_REPO"
      --lifpo-repo-id "$LIFPO_REPO"
      "${COMMON[@]}"
    )
    if [[ -n "$DATASET_REVISION" ]]; then
      ARGS+=(--dataset-revision "$DATASET_REVISION")
    fi
    exec python -m scattered_discovery.release.causal_micro_lab "${ARGS[@]}"
    ;;
esac
