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
every fourth row. Per-shard worker count is `96` for all `k` values. Eval
response length is `4096` tokens so reasoning models have room to think before
emitting the three final rule lines. On A100 80GB, the causal eval configs set
`sglang_mem_fraction_static: 0.82` so SGLang can use substantially more KV/cache
memory than the conservative scattered eval default. The 96-worker default is
chosen to stay just under the observed 4096-token KV-cache saturation point; if
token usage stays below about 0.90, probe `EVAL_WORKERS=112`, and if SGLang logs
`KV cache pool is full`, drop to `EVAL_WORKERS=80`.

The causal eval Slurm wrappers default to `WANDB_PROJECT=scattered-discovery`.
Set `WANDB_PROJECT=` to disable W&B for a run, or override it with another
project name. Per-sample validity metrics and final set summaries are logged.

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

## Frozen final comparison eval

The replacement comparison set is committed under:

```text
eval_sets/causal_micro_lab/final_v1/
```

It contains 96 states: 24 each for `M={4,8,12,16}`. Every `M` cell contains
eight low-, eight medium-, and eight high-separation states, and every row uses
a distinct held-out hidden mode. Its manifest records SHA-256 hashes and the
zero-overlap audit against the previous canonical validation/test set.

Run the Qwen3-4B base-model evaluation on one cluster GPU with:

```bash
sbatch scripts/cluster/sbatch_causal_micro_lab_final_eval.slurm
```

The run samples `K=16` independent completions per state and derives
`K={4,8,12}` from stable prefixes of the same bank. Each primary request gets
4096 response tokens with Qwen thinking enabled. A request ending with
`finish_reason=length` is automatically finalized by a short deterministic
second request with thinking disabled and the original reasoning supplied as
context.

Results and live W&B metrics are written under:

```text
artifacts/causal_micro_lab_final_eval/
```

The `latest/report/` directory contains CSV tables for bootstrap confidence
intervals, per-state metrics, `K x M x separation` slices, and mode
reachability. Corresponding comparison plots are logged to W&B. Important
outputs include exact and
budget-normalized coverage, valid-output rate, duplicity among valid outputs,
dominant-mode mass, effective mode count, mechanism-family coverage, generated
consequence separation, cap-hit rate, and fallback success.

## CD-GRPO on the Slurm cluster

CD-GRPO uses a dedicated agent loop and a project-owned veRL v1 TaskRunner.
It does not patch the installed veRL checkout. Start with the two-update,
four-GPU integration smoke:

```bash
sbatch scripts/cluster/sbatch_causal_micro_lab_cd_grpo_smoke.slurm
```

The Slurm wrapper bootstraps the project environment and runs the veRL
compatibility check before model allocation. It generates the smoke dataset on
first use, samples `G=16`, and uses full probes, log-det credit, `beta=0.3`, and
the checkpointed archive. A successful preflight prints:

```text
CD-GRPO veRL compatibility check passed.
```

Method diagnostics are logged under `cd_grpo/*` in W&B. In particular,
`groups_with_2plus_unique_valid_rate` and `diversity_signal_active_rate` show
whether sampled groups contain distinct valid alternatives and whether the
diversity term contributes a nonzero signal. Pairwise consequence distance,
validity/diversity advantage magnitudes, archive novelty, and all-truncated
group rates are logged alongside coverage and effective mode count. Each veRL
checkpoint also contains `cd_grpo_archive.json`; normal `trainer.resume_mode=auto`
restores the actor and archive together.

After the smoke succeeds, submit the 384-update trainable run:

```bash
sbatch scripts/cluster/sbatch_causal_micro_lab_cd_grpo.slurm
```

The full CD-GRPO run reads the same frozen
`data/causal_micro_lab/trainable/verl_train.jsonl` and `verl_val.jsonl` files
as the validity-GRPO run. If those files were not transferred to the cluster,
the shared seed and trainable preset regenerate them at those exact paths.
The run-specific `cd_grpo_agent_loop.yaml` remaps their existing
`causal_micro_lab_agent_loop` routing label in memory; it does not rewrite the
shared rows. CD-GRPO computes its sparse validity reward in its own loop, so
the validity run's embedded syntax-shaping setting does not enter the method
reward.

The full run starts directly from `Qwen/Qwen3-4B`; it does not require the
discarded no-thinking warmup checkpoint.

The full config uses a stable experiment name, so resubmitting that command
after the 12-hour wall-time limit resumes from its latest checkpoint. Change
the YAML experiment name before launching a separate independent run.

Method arms remain YAML-selectable. Set `cd_grpo_beta: 0.0` for the matched
GRPO arm, `cd_grpo_variant: count` for count credit,
`cd_grpo_archive: "false"` to disable archive scaling, or change
`cd_grpo_probe_fraction` for deterministic probe subsampling.

For the scheduler-free Blackwell machine, the optional equivalent launcher is:

```bash
bash scripts/blackwell/run_causal_micro_lab_cd_grpo.sh
```
