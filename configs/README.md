# Config Index

Use this directory by workflow, not by file extension.

```text
configs/local_eval/
```

Local Ollama scattered-causal evaluation configs for `scattered-discovery-eval`.
These are for smoke tests and local baselines, not veRL training.

```text
configs/envspecs/
```

Single `EnvSpec` JSON files for `scattered-discovery-local`. These are small
interactive checks for one environment instance.

```text
configs/verl/agent_loop.yaml
```

veRL AgentLoop registration. The cluster launcher passes this to veRL so rows
with `agent_name: discovery_agent_loop` use our interactive environment loop.

```text
configs/verl/datasets/
```

YAML specs consumed by `scattered-discovery-make-dataset`. These generate
Parquet/JSONL rows with `env_spec_json` for veRL training or evaluation.

Current dataset presets:

```text
all_envs.yaml          mixed HypoSpace + scattered-causal train/val datasets
scattered_smoke.yaml   small scattered-causal cluster smoke run
scattered_signal_pilot.yaml allocation-conscious scattered-causal signal run
scattered_pilot.yaml   larger scattered-causal mixed-dispersion pilot run
```

Reward controls live in `reward_profile` or `task.reward`. Environment dynamics
live under `task.world` for `scattered_causal`.
Use `dispersion_values` for balanced mixed scattered-causal train/val files and
`world_values` for graph-shape distributions.

```text
configs/verl/runs/
```

YAML defaults for veRL training launches: dataset config path, train/val files,
model ID/download behavior, batch sizes, rollout count, save/test frequency,
token lengths, and experiment name prefix. Slurm resources stay in
`scripts/cluster/*.slurm`.
