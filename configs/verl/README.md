# veRL Backend Notes

Full docs:

```text
docs/datasets.md
docs/scenarios.md
docs/verl_cluster.md
docs/environment_contract.md
```

`agent_loop.yaml` registers `scattered_discovery.verl.agent_loop.DiscoveryAgentLoop`.

The launch script targets:

```text
actor_rollout_ref.rollout.name=sglang
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.agent.default_agent_loop=discovery_agent_loop
algorithm.adv_estimator=grpo
```

Each dataset row must include:

```text
env_spec_json
agent_name
prompt
raw_prompt
```

`env_spec_json` is the source of truth for the active environment. The prompt fields exist so veRL's dataset loader has a normal prompt column; the custom AgentLoop builds the real prompt from the env.

Use `scattered-discovery-make-dataset --config configs/verl/datasets/all_envs.yaml` or `scripts/cluster/prepare_verl_datasets.sh` to generate compatible rows.
Install the local cluster extras with `uv sync --extra verl`; this includes Parquet writers and the W&B client.

`configs/verl/datasets/*.yaml` files are dataset-generation specs. Each dataset
entry supports:

```text
datasets/all_envs.yaml          mixed env-family train/val generation
datasets/scattered_smoke.yaml   small scattered-causal cluster smoke generation
datasets/scattered_signal_pilot.yaml allocation-conscious scattered-causal signal generation
datasets/scattered_pilot.yaml   larger mixed-dispersion scattered-causal pilot generation
```

```text
output
data_source
env_type
count
seed
dispersion_values
world_values
protocol
max_steps
max_commit
agent_name
task
reward_profile
```

`configs/verl/runs/*.yaml` files are veRL training launch defaults:

```text
runs/scattered_smoke.yaml   2-GPU smoke training defaults
runs/scattered_signal_pilot.yaml 4-GPU 32-step signal-pilot training defaults
runs/scattered_pilot.yaml   4-GPU full pilot training defaults
```

These control train/val file paths, batch sizes, rollout count, save/test
frequency, token lengths, model ID/download behavior, and experiment name
prefix. Environment variables passed through Slurm still override YAML values.

Reward defaults are centralized in `src/scattered_discovery/rewards.py`.
The default profile is `terminal_only` for clean GRPO/pass@K comparisons.
The scattered pilot uses `terminal_clean_invalid_bonus`, which adds a small
reward for clean parseable invalid final commits. Dataset entries can also opt
into `shaped` with `reward_profile: shaped`, or override individual values
under `task.reward`.

For `scattered_causal`, `task.world` maps directly to `WorldConfig`,
`dispersion_values` cycles shared-prefix/diversity settings, `world_values`
samples graph-shape fields such as branch count/depth/distractors,
`base_budget_from_branch_depth_overhead` can derive the budget from depth,
`reward_profile`/`task.reward` controls reward values, and `task.agent` maps directly to
`AgentConfig`, so all environment shape, evidence noise, action cost, reward,
and prompt display controls live in YAML.

Currently supported `env_type` values:

```text
hypospace_causal
hypospace_boolean
hypospace_3d
scattered_causal
```
