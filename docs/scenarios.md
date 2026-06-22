# Scenario Examples

These examples are meant to be copied into a dataset YAML file and run with:

```bash
UV_CACHE_DIR=.uv-cache uv run --extra verl scattered-discovery-make-dataset \
  --config path/to/datasets.yaml
```

Omitting `reward_profile` uses the clean `terminal_only` baseline.

## Local JSONL Smoke Dataset

Use JSONL to avoid Parquet dependencies during fast local checks.

```yaml
datasets:
  - name: scattered_local_smoke
    env_type: scattered_causal
    output: data/verl/scattered_local_smoke.jsonl
    count: 8
    seed: 11
    protocol: single
    max_steps: 6
    max_commit: 1
    task:
      dispersion: 0.0
      world:
        num_branches: 2
        branch_depth: 2
        distractors_per_node: 1
        base_budget: 6
        test_cost: 1
        intervene_cost: 2
```

## Easy Single-Answer Scattered Causal

Use this to check whether the model can learn the environment at all.

```yaml
datasets:
  - name: scattered_easy_single
    env_type: scattered_causal
    output: data/verl/scattered_easy_single.parquet
    count: 512
    seed: 101
    protocol: single
    max_steps: 8
    max_commit: 1
    task:
      dispersion: 0.0
      world:
        num_branches: 3
        branch_depth: 2
        distractors_per_node: 1
        noise_sigma: 0.25
        base_budget: 8
        test_cost: 1
        intervene_cost: 2
```

Why this is easier:

```text
low dispersion       branches share prefix structure
short branch_depth   fewer variables needed for a final path
low distractors      fewer false interventions to filter
low noise            evidence accepts faster
```

## Hard High-Dispersion Scattered Causal

Use this when testing whether training handles scattered final-answer goals.

```yaml
datasets:
  - name: scattered_hard_single
    env_type: scattered_causal
    output: data/verl/scattered_hard_single.parquet
    count: 1024
    seed: 201
    protocol: single
    max_steps: 13
    max_commit: 1
    task:
      dispersion: 1.0
      world:
        num_branches: 5
        branch_depth: 3
        distractors_per_node: 3
        noise_sigma: 0.35
        base_budget: 13
        test_cost: 1
        intervene_cost: 2
```

Why this is harder:

```text
high dispersion      branches do not share much evidence
more branches        more final hypotheses exist
more distractors     interventions expose more false candidates
longer depth         more evidence is needed for support
```

## Set-Commit Protocol

Use this to test single-rollout multi-answer behavior. This is closer to a set reward setup.

```yaml
datasets:
  - name: scattered_set_k4
    env_type: scattered_causal
    output: data/verl/scattered_set_k4.parquet
    count: 512
    seed: 301
    protocol: set
    max_steps: 16
    max_commit: 4
    task:
      dispersion: 1.0
      budget: 16
      world:
        num_branches: 4
        branch_depth: 3
        distractors_per_node: 2
        base_budget: 16
        test_cost: 1
        intervene_cost: 2
```

Use `protocol: single` with `ROLLOUT_N=K` for pass@K-style multiple independent completions. Use `protocol: set` with `max_commit: K` when one rollout must commit up to K hypotheses.

## Noisy Evidence Stress Test

Use this to test whether format/admissibility rewards and repeated evidence help under uncertainty.

```yaml
datasets:
  - name: scattered_noisy
    env_type: scattered_causal
    output: data/verl/scattered_noisy.parquet
    count: 1024
    seed: 401
    protocol: single
    max_steps: 14
    max_commit: 1
    reward_profile: shaped
    task:
      dispersion: 0.5
      world:
        num_branches: 4
        branch_depth: 3
        distractors_per_node: 2
        true_mean: 1.0
        false_mean: 0.0
        noise_sigma: 0.6
        accept_threshold: 0.9
        reject_threshold: 0.1
        base_budget: 14
        test_cost: 1
        intervene_cost: 2
```

## Budget Pressure

Use this to force efficient querying.

```yaml
datasets:
  - name: scattered_budget_pressure
    env_type: scattered_causal
    output: data/verl/scattered_budget_pressure.parquet
    count: 512
    seed: 501
    protocol: single
    max_steps: 7
    max_commit: 1
    task:
      dispersion: 0.75
      world:
        num_branches: 4
        branch_depth: 3
        distractors_per_node: 2
        base_budget: 7
        test_cost: 1
        intervene_cost: 2
      reward:
        budget_penalty: 0.02
```

## Mixed Environment Curriculum

A single YAML can write multiple train files. Train on one file at a time by changing `TRAIN_FILE`, or concatenate upstream later if you want a mixed dataset.

```yaml
defaults:
  agent_name: discovery_agent_loop
  protocol: single
  max_steps: 6
  max_commit: 1
  reward_profile: terminal_only

datasets:
  - name: causal_easy
    env_type: scattered_causal
    output: data/verl/causal_easy.parquet
    count: 512
    seed: 101
    task:
      dispersion: 0.0
      world:
        num_branches: 3
        branch_depth: 2

  - name: causal_hard
    env_type: scattered_causal
    output: data/verl/causal_hard.parquet
    count: 512
    seed: 201
    task:
      dispersion: 1.0
      world:
        num_branches: 5
        branch_depth: 3

  - name: hypospace_boolean
    env_type: hypospace_boolean
    output: data/verl/hypospace_boolean.parquet
    count: 512
    seed: 301
    task:
      variables: [x, y]
      operators: [AND, OR, NOT, XOR, NOR]
      max_depth: 2
      query_budget: 2
```
