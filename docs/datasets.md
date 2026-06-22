# Dataset Generation

The veRL backend reads dataset rows that contain an `env_spec_json` field. The custom agent loop builds the real prompt from that spec, runs the interactive environment, and returns a final scalar reward.

## Recommended YAML Path

Generate all default datasets:

```bash
UV_CACHE_DIR=.uv-cache uv run --extra verl scattered-discovery-make-dataset \
  --config configs/verl/datasets/all_envs.yaml
```

The cluster helper does the same thing:

```bash
DATASET_CONFIG=configs/verl/datasets/all_envs.yaml scripts/cluster/prepare_verl_datasets.sh
```

Use another YAML file by changing `DATASET_CONFIG`.

The default YAML writes train and validation splits. Scattered-causal train and
validation files both use `dispersion_values` to keep one balanced mixed file
instead of separate files per dispersion.

Preset scattered-causal configs:

```text
configs/verl/datasets/scattered_smoke.yaml    320 mixed train / 80 mixed val, easier pipeline run
configs/verl/datasets/scattered_pilot.yaml    8192 mixed train / 1280 mixed val, target pilot run
```

## YAML Schema

Top-level shape:

```yaml
defaults:
  agent_name: discovery_agent_loop
  protocol: single
  max_steps: 13
  max_commit: 1

datasets:
  - name: scattered_causal_train
    data_source: scattered_causal_mixed
    env_type: scattered_causal
    output: data/verl/scattered_causal_train.parquet
    count: 256
    seed: 301
    dispersion_values: [0.0, 0.25, 0.5, 0.75, 1.0]
    world_values:
      num_branches: [3, 4, 5]
      branch_depth: [2, 3]
      distractors_per_node: [1, 2]
      base_budget: [9, 11, 13]
    task:
      world:
        noise_sigma: 0.35
        accept_threshold: 0.82
      agent:
        max_evidence_items: 12
```

Default split sizes:

```text
*_train    256 rows
*_val       64 rows
```

Supported dataset fields:

```text
name          optional label for humans
data_source   optional veRL/W&B source label
output        required path, .parquet writes Parquet, other suffixes write JSONL
env_type      scattered_causal, hypospace_causal, hypospace_boolean, hypospace_3d
count         number of rows
seed          base seed
protocol      single or set
max_steps     max interaction turns
max_commit    max final hypotheses in COMMIT
agent_name    veRL agent loop name, normally discovery_agent_loop
reward_profile terminal_only or shaped
reward        optional top-level reward override merged into task.reward
dispersion_values scattered_causal-only round-robin mixture over task.dispersion
world_values  scattered_causal-only sampled lists for WorldConfig fields
task          environment-specific task config
```

## Reward Config

Default reward values live in one central module:

```text
src/scattered_discovery/rewards.py
```

Print all profiles and per-env defaults:

```bash
UV_CACHE_DIR=.uv-cache uv run scattered-discovery-rewards
```

Print one env default:

```bash
UV_CACHE_DIR=.uv-cache uv run scattered-discovery-rewards --env-type scattered_causal
```

The default profile is `terminal_only`:

```text
reward = valid_hypothesis_reward * valid_unique_count
false/non-final/unsupported penalties = 0
format/admissibility shaping = 0
```

Use this for clean GRPO-collapse and pass@K comparisons. Recovery is measured
during eval as `valid_unique_count / target_count`; it is not a separate reward
component.

The opt-in `shaped`
profile adds small syntax/admissibility rewards and small negative guardrails:

```bash
UV_CACHE_DIR=.uv-cache uv run scattered-discovery-rewards --profile shaped
```

Select a profile per dataset with `reward_profile`, or override individual
values under `task.reward` for one-off ablations:

```yaml
defaults:
  reward_profile: terminal_only

datasets:
  - name: hypospace_causal_custom
    env_type: hypospace_causal
    output: data/verl/hypospace_causal_custom.parquet
    count: 1024
    seed: 101
    task:
      nodes: [A, B, C]
      max_edges: 2
      query_budget: 2
      reward:
        profile: shaped
        format_reward: 0.01
```

`hypospace_causal`, `hypospace_boolean`, `hypospace_3d`, and
`scattered_causal` each resolve through the same profile interface. For
`scattered_causal`, final reward values are copied into `WorldConfig`; the
shaping values are applied by the generic `DiscoveryEnv` adapter.

There is no numeric duplicate penalty. Following Puri et al.'s multi-answer
setup, duplicates in set `COMMIT` fail the uniqueness condition and zero the
final reward. In single-answer/pass@K mode, independent rollouts are not
penalized for matching one another.

For each row `i`, the builder sets `seed = base_seed + i`. For `scattered_causal`, it also sets:

```text
world_seed = base_seed + i
episode_seed = (base_seed + i) * 1009 + 19
```

If `dispersion_values` is present, row `i` gets:

```text
task.dispersion = dispersion_values[i % len(dispersion_values)]
```

If `world_values` is present, each row gets deterministic sampled values merged
into `task.world`:

```yaml
world_values:
  num_branches: [3, 4, 5]
  branch_depth: [2, 3, 4]
  distractors_per_node: [1, 2, 3]
  base_budget: [9, 11, 13]
```

Use `dispersion_values` for shared-prefix/diversity structure and `world_values`
for graph-shape/difficulty distribution. They are independent: one controls how
branches overlap, the other controls graph size, depth, distractors, budget, and
other `WorldConfig` fields.

Do not put the same field in both `world_values` and `task.world`. Sampled
fields belong only in `world_values`; fixed fields belong only in `task.world`.
The dataset builder rejects overlaps.

Use this for both mixed training files and mixed validation files. If the row
count is divisible by the number of dispersion values, the generated file is
exactly balanced across those values.

## Scattered Causal Controls

`task.world` maps directly to `WorldConfig`:

```yaml
task:
  dispersion: 1.0
  budget: 10       # optional override; otherwise world.base_budget is used
  world:
    num_branches: 4
    branch_depth: 3
    distractors_per_node: 2
    true_mean: 1.0
    false_mean: 0.0
    noise_sigma: 0.35
    accept_threshold: 0.82
    reject_threshold: 0.18
    base_budget: 10
    test_cost: 1
    intervene_cost: 2
    invalid_action_cost: 1
  agent:
    include_hidden_debug_in_prompt: false
    include_evidence_status_in_prompt: false
    max_evidence_items: 12
```

For mixed training, put `dispersion_values` at the dataset-entry level instead
of `task.dispersion`:

```yaml
datasets:
  - name: scattered_causal_train
    data_source: scattered_causal_mixed
    dispersion_values: [0.0, 0.25, 0.5, 0.75, 1.0]
    world_values:
      num_branches: [3, 4, 5]
      branch_depth: [2, 3, 4]
    task:
      world:
        noise_sigma: 0.35
        accept_threshold: 0.82
```

Important scattered-causal knobs:

```text
num_branches            number of final causal paths
branch_depth            edges per final path
distractors_per_node    false outgoing candidates shown during interventions
dispersion              0 means shared-prefix/easier, 1 means independent branches/harder
noise_sigma             evidence noise
accept_threshold        posterior threshold for accepted evidence
reject_threshold        posterior threshold for rejected evidence
base_budget             starting action budget
test_cost               cost of TEST
intervene_cost          cost of INTERVENE
invalid_action_cost     cost of invalid/admissibility failures
max_evidence_items      evidence-summary length in prompts
include_evidence_status_in_prompt
                        debug/curriculum flag; false keeps verifier status hidden
```

Reward controls normally live in `reward_profile` or `task.reward`, not in
`task.world`. The `WorldConfig` reward fields still exist for backwards
compatibility with older local scattered-causal configs.

## Other Environment Controls

HypoSpace causal:

```yaml
task:
  nodes: [A, B, C]
  max_edges: 2
  query_budget: 2
```

HypoSpace boolean:

```yaml
task:
  variables: [x, y]
  operators: [AND, OR, NOT, XOR, NOR]
  max_depth: 2
  query_budget: 2
```

HypoSpace 3D:

```yaml
task:
  grid_size: 2
  max_height: 3
  max_blocks: 3
  query_budget: 2
```

## Quick CLI Path

For one-off scattered-causal generation:

```bash
UV_CACHE_DIR=.uv-cache uv run scattered-discovery-make-dataset \
  --env-type scattered_causal \
  --output data/verl/scattered_causal_train.jsonl \
  --count 256 \
  --seed 301 \
  --protocol single \
  --max-steps 6 \
  --max-commit 1 \
  --dispersion 0.5 \
  --num-branches 3 \
  --branch-depth 2 \
  --base-budget 8
```

Use YAML for reproducible experiments; use CLI flags for quick smoke checks.
