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
