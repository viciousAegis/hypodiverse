# Scattered Discovery Docs

Start here when setting up, generating datasets, running local smoke tests, or launching veRL training.

For the config directory layout itself, see [../configs/README.md](../configs/README.md).

## Guides

- [Setup](setup.md): local dependencies, repo-local caches, Ollama, and W&B.
- [Dataset Generation](datasets.md): YAML schema, CLI flags, generated row format, and Parquet/JSONL outputs.
- [Scenario Examples](scenarios.md): ready-to-copy examples for easy/hard scattered causal, set protocol, noisy evidence, and mixed training files.
- [Scattered Discovery Environment](scattered_discovery_env.md): what the synthetic env is, how dispersion controls diversity, and how path terminals extend to motifs.
- [Local Runs](local_runs.md): Ollama baselines, generic interactive smoke tests, and the scattered-causal browser viewer.
- [Evaluation Runs](evaluation.md): local and cluster eval over shared EnvSpec rows.
- [veRL Cluster Runs](verl_cluster.md): dataset prep, GRPO launch, SGLang rollout settings, W&B, and common overrides.
- [Environment Contract](environment_contract.md): how `EnvSpec`, `DiscoveryEnv`, dataset rows, rewards, and the veRL agent loop fit together.
- [Algorithm Recipes](algorithms.md): GRPO, set-reward GRPO, and custom objective scaffolding.
- [Benchmark Notes](benchmarks.md): HypoSpace and related benchmark options, risks, and remedies.
- [ECHO / World Modeling](world_modeling_echo.md): how environment-observation prediction would connect to our setting.

## Fast Path

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra verl
sbatch scripts/cluster/sbatch_verl_smoke_grpo.slurm
sbatch scripts/cluster/sbatch_verl_pilot_grpo.slurm
```

Use `configs/verl/datasets/all_envs.yaml` as the main cluster-facing control surface.
Use `configs/verl/datasets/scattered_smoke.yaml` and `configs/verl/datasets/scattered_pilot.yaml`
for the canned scattered-causal cluster runs.
