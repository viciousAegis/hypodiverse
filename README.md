# Scattered Discovery

Synthetic interactive causal discovery environment for testing diverse hypothesis generation.

The environment keeps the latent hypothesis space enumerable for exact evaluation, but the agent
interacts through a small DSL:

```text
INTERVENE x00
TEST edge(x00,x01)
TEST path(x00,x01,x02)
COMMIT path(x00,x01,x02,x03)
COMMIT [path(x00,x01,x02,x03); path(x10,x11,x12,x13)]
```

Project dependencies are managed with `uv`. To keep project cache local:

```bash
UV_CACHE_DIR=.uv-cache uv sync
```

Ollama model files and home state should also be kept local when running baselines:

```bash
HOME="$PWD/.ollama-home" OLLAMA_MODELS="$PWD/.ollama-models" ollama serve
```

The Ollama backend supports Qwen thinking models through structured `thinking` and `content`
fields. Configs keep `think: true`; if Qwen returns an empty final content field, the runner can
make a small no-thinking finalizer call with the hidden thinking trace and ask for one DSL action.

Run eval configs:

```bash
UV_CACHE_DIR=.uv-cache uv run scattered-discovery-eval --config configs/local_eval/scattered_smoke_qwen3_1_7b.yaml
UV_CACHE_DIR=.uv-cache uv run scattered-discovery-eval --config configs/local_eval/scattered_qwen3_4b_single_quick.yaml
UV_CACHE_DIR=.uv-cache uv run scattered-discovery-eval --config configs/local_eval/scattered_qwen3_4b_set_quick.yaml
```

Run the scattered-causal browser viewer:

```bash
UV_CACHE_DIR=.uv-cache uv run scattered-discovery-viewer --port 8765
```

Generate example graphs across dispersion values:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run --no-sync python -m scattered_discovery.graph_gallery
```

## Documentation

See [docs/README.md](docs/README.md) for setup, dataset generation, scenario examples, local runs, cluster veRL training, and the environment contract. See [configs/README.md](configs/README.md) for the config directory map.

## Interactive veRL Backend

The cluster-facing path uses a shared `DiscoveryEnv` contract and a custom veRL AgentLoop:

```text
dataset row -> EnvSpec -> DiscoveryEnv -> model ACTION -> env.step -> final reward
```

Implemented env families:

```text
scattered_causal      synthetic evidence-backed causal discovery
hypospace_causal      active intervention version of HypoSpace causal graphs
hypospace_boolean     active query version of HypoSpace Boolean expressions
hypospace_3d          active view/probe version of HypoSpace 3D reconstruction
```

HypoSpace is intentionally used as an interactive extension, not the original static repeated-proposal benchmark.

Generate veRL datasets:

```bash
UV_CACHE_DIR=.uv-cache uv run --extra verl scattered-discovery-make-dataset \
  --config configs/verl/datasets/all_envs.yaml
```

Edit `configs/verl/datasets/all_envs.yaml` to control outputs, seeds, counts, protocols, task parameters, and scattered-causal `WorldConfig`/`AgentConfig` fields. `scripts/cluster/prepare_verl_datasets.sh` generates the datasets listed in that YAML, or set `DATASET_CONFIG=/path/to/file.yaml` to use another file.

The custom AgentLoop is registered in `configs/verl/agent_loop.yaml` and uses veRL's Qwen3-compatible fixed-base message tokenization for response-side observation tokens. This matters because Qwen3 chat templates can drop prior reasoning content in multi-turn histories.

Cluster launch:

```bash
scripts/cluster/prepare_verl_datasets.sh
sbatch scripts/cluster/sbatch_verl_smoke_grpo.slurm
sbatch scripts/cluster/sbatch_verl_pilot_grpo.slurm
```

W&B is enabled through veRL:

```bash
trainer.logger='["console","wandb"]'
```

The cluster script keeps project caches local by default:

```text
.cache/huggingface
.cache/ray
data/verl
.wandb
checkpoints
```
