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
causal_micro_lab
```

## Boolean Causal Micro-Lab cluster runs

Frozen causal micro-lab artifacts are generated from code on the cluster if the
configured files are missing. The training pilot rows live under:

```text
data/causal_micro_lab/pilot/
```

The canonical eval rows for model-to-model comparisons live separately under:

```text
data/causal_micro_lab/canonical_eval/
```

The default causal micro-lab target counts are `M={4,8,16}`. The canonical eval
preset uses the held-out `test` mode split from the same split seed as the pilot
dataset, so it does not overlap the pilot train/val rows by hidden mode. The
builder also avoids duplicate frozen state rows across output splits.

The final paper eval file is:

```text
data/causal_micro_lab/canonical_eval/verl_test.jsonl
```

It contains 384 rows when generated with the default preset: 128 rows each for
`M=4`, `M=8`, and `M=16`.

Use the SFT files for supervised warmup and the veRL rows for GRPO/eval:

```text
sft_train.jsonl, sft_val.jsonl, sft_test.jsonl
verl_train.jsonl, verl_val.jsonl, verl_test.jsonl
```

Launch validity-only GRPO on the cluster with:

```bash
sbatch scripts/cluster/sbatch_causal_micro_lab_validity_smoke.slurm
sbatch scripts/cluster/sbatch_causal_micro_lab_validity_pilot.slurm
```

These use `causal_micro_lab_agent_loop`, which verifies rule-line outputs with
the exact local verifier and logs parse/syntax/evidence-validity metrics through
`reward_extra_info`. Dataset generation is controlled by the run config fields
`causal_micro_lab_generate_dataset_if_missing`,
`causal_micro_lab_dataset_output_dir`, and
`causal_micro_lab_eval_output_dir`.

Run grouped final eval with fixed answer budgets using:

```bash
EVAL_CONFIG=configs/verl/eval/causal_micro_lab_test_k4.yaml sbatch scripts/cluster/sbatch_causal_micro_lab_eval.slurm
EVAL_CONFIG=configs/verl/eval/causal_micro_lab_test_k8.yaml sbatch scripts/cluster/sbatch_causal_micro_lab_eval.slurm
EVAL_CONFIG=configs/verl/eval/causal_micro_lab_test_k16.yaml sbatch scripts/cluster/sbatch_causal_micro_lab_eval.slurm
```

For the faster sharded version, use one SGLang server per GPU:

```bash
EVAL_CONFIG=configs/verl/eval/causal_micro_lab_test_k4.yaml sbatch scripts/cluster/sbatch_causal_micro_lab_eval_sharded.slurm
EVAL_CONFIG=configs/verl/eval/causal_micro_lab_test_k8.yaml sbatch scripts/cluster/sbatch_causal_micro_lab_eval_sharded.slurm
EVAL_CONFIG=configs/verl/eval/causal_micro_lab_test_k16.yaml sbatch scripts/cluster/sbatch_causal_micro_lab_eval_sharded.slurm
```

The default sharded eval requests 4 GPUs and sets `eval_num_shards: 4`. Each
shard runs its own single-GPU SGLang server on a separate port and evaluates
every fourth row. Per-shard worker count is `256` for all `k` values. Eval
response length is `4096` tokens so reasoning models have room to think before
emitting the three final rule lines. On A100 80GB, the causal eval configs set
`sglang_mem_fraction_static: 0.82` so SGLang can use substantially more KV/cache
memory than the conservative scattered eval default. For single-GPU eval, the
same `256` workers feed one SGLang server; override with `EVAL_WORKERS=128` or
`SGLANG_MEM_FRACTION_STATIC=0.75` if the server reports memory pressure.

Each run writes a per-sample `summary.json` and a grouped set-level
`set_summary.json`. The grouped summary reports `pass_at_k`, exact coverage,
budget-normalized coverage, effective mode count, family coverage, duplicate
valid modes, unavoidable duplicate valid modes, and extra duplicate valid modes,
sliced by `M`, separation bucket, and family bucket.

For the cleanest k comparison, run the `k=16` config and use its prefix
summaries:

```text
set_summary_k4.json
set_summary_k8.json
set_summary_k16.json
```

Those three files are computed from the same generated sample slots
`sample0000` through `sample0015`, so `k=4` is the first four samples of the
same `k=16` run rather than a separate stochastic eval.
