# Setup

This project uses `uv`. Keep all project cache and generated state inside the repo.

## Local Python Environment

For local smoke tests:

```bash
UV_CACHE_DIR=.uv-cache uv sync
```

For veRL dataset generation with Parquet and W&B support:

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra verl
```

The repo ignores `.uv-cache/`, `.uv-tools/`, `.venv/`, `.ruff_cache/`, `data/`, `checkpoints/`, `.cache/`, and `.wandb/`.

`scripts/env.sh` keeps project and cluster cache state under the repo by
default:

```text
data/                 generated train/validation/eval datasets
checkpoints/          veRL trainer checkpoints
.cache/models/        downloaded Hugging Face model snapshots
.cache/huggingface/   HF hub/assets/xet/transformers/datasets caches
.cache/ray/           Ray temp state
.cache/pip/           pip/uv pip wheel cache
.cache/torch/         Torch cache
.cache/torch_extensions/
.cache/triton/        Triton kernel cache
.cache/nv/            CUDA JIT cache
.cache/vllm/          vLLM config/cache root
.cache/sglang/        SGLang cache root if honored by the installed version
.wandb/               W&B run/cache/config state
artifacts/wandb/      W&B artifacts
```

## Cluster Environment

Copy the example env file only if you need secrets/tokens:

```bash
cp .env.example .env
```

`.env` is intentionally secrets-only. Normal configuration lives elsewhere:

```text
configs/verl/runs/*.yaml       model, train/val files, batch sizes, rollout count
configs/verl/datasets/*.yaml   generated dataset specs
scripts/env.sh                 repo-local cache/path defaults
scripts/cluster/*.slurm        Slurm account, partition, GPU count, modules
```

The Slurm scripts source `scripts/env.sh`, which imports only recognized
secret/token keys from `.env` with shell xtrace disabled. Currently that list is
`WANDB_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN`, `HF_HUB_TOKEN`, and
`HUGGING_FACE_HUB_TOKEN`. By default, submit from the repo root and do not set
`REPO_DIR`; the Slurm scripts use `SLURM_SUBMIT_DIR` and then `scripts/env.sh`
derives the repo paths.

If `$VENV_DIR` does not exist, the first Slurm run runs:

```bash
uv sync --python "$PYTHON_VERSION" --extra verl
```

That bootstraps this project and light Parquet/W&B dependencies. It does not
silently install the full CUDA veRL/SGLang/vLLM training stack; the bootstrap
checks for `verl`, `torch`, `ray`, and the selected rollout backend and fails
early with a missing-package list if they are absent.

Cluster training defaults to `PYTHON_VERSION=3.11`. Do not use Python 3.13 for
the CUDA training stack: the pinned PyTorch/torchvision CUDA 12.1 wheels do not
provide `cp313` builds. If a previous failed install created a Python 3.13 venv,
rebuild it explicitly:

```bash
RECREATE_VENV=1 bash scripts/cluster/install_verl_stack.sh
```

Install that training stack once before submitting GRPO jobs:

```bash
bash scripts/cluster/install_verl_stack.sh
```

The helper uses `uv pip install` inside `$VENV_DIR`. Its defaults target the
current Ampere/CUDA 12.1 Slurm scripts:

```text
torch==2.5.1 from the cu121 PyTorch wheel index
ray[data,train,tune,serve]
veRL from a source checkout under $CACHE_ROOT/src/verl
veRL's [sglang] extra, which installs the SGLang backend
```

Override `PYTORCH_INDEX_URL`, `TORCH_SPEC`, `VERL_SRC`, or `INSTALL_TORCH`
inline when invoking `scripts/cluster/install_verl_stack.sh` if the cluster
image already provides a compatible CUDA stack.

If a large wheel download times out, rerun the same command. The installer uses
repo-local uv caches and retries, so completed downloads are reused:

```bash
UV_HTTP_TIMEOUT=300 UV_INSTALL_RETRIES=5 bash scripts/cluster/install_verl_stack.sh
```

If uv prints warnings that `sglang` has no extras named `openai` or `srt`, treat
them as dependency-metadata warnings unless the install fails. The installer and
Slurm bootstrap explicitly check for the real SGLang runtime module
`sglang.srt`.

If `import sglang` fails after those warnings, either the install aborted before
SGLang was installed or you are testing outside `$VENV_DIR`. From the repo root:

```bash
source "$VENV_DIR/bin/activate"
uv pip install "$SGLANG_SPEC"
python -c "import sglang, sglang.srt; print('sglang ok')"
```

The default is `SGLANG_SPEC=sglang==0.5.8`, matching the version veRL currently
resolves in this setup.

## Ollama Baselines

Keep Ollama model files inside the repo:

```bash
HOME="$PWD/.ollama-home" \
OLLAMA_MODELS="$PWD/.ollama-models" \
ollama serve
```

Pull the local smoke/baseline models:

```bash
HOME="$PWD/.ollama-home" OLLAMA_MODELS="$PWD/.ollama-models" ollama pull qwen3:1.7b
HOME="$PWD/.ollama-home" OLLAMA_MODELS="$PWD/.ollama-models" ollama pull qwen3:4b
```

The Ollama path is only for local smoke tests and baselines. Cluster training should use veRL with SGLang or another supported rollout backend.

## W&B

The cluster script enables W&B by default via:

```text
trainer.logger='["console","wandb"]'
```

On the cluster, either run `wandb login` once in the environment or set:

```bash
export WANDB_API_KEY=...
```

The run script keeps W&B files in repo-local directories:

```text
.wandb/
.wandb/cache/
.wandb/config/
```

To disable W&B for a debug run:

```bash
sbatch --export=ALL,WANDB_LOGGER='["console"]' \
  scripts/cluster/sbatch_verl_smoke_grpo.slurm
```
