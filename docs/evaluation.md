# Evaluation Runs

Evaluation uses the same `EnvSpec` rows as veRL training. This keeps base-model, checkpoint, local, and cluster eval comparable: the model changes, but the environment contract does not.

## Local Ollama Eval

Generate an eval file:

```bash
UV_CACHE_DIR=.uv-cache uv run scattered-discovery-make-dataset \
  --env-type hypospace_causal \
  --output data/eval/hypospace_causal_eval.jsonl \
  --count 32 \
  --seed 900 \
  --max-steps 6 \
  --query-budget 2
```

Run a local model:

```bash
UV_CACHE_DIR=.uv-cache uv run scattered-discovery-eval-envspecs \
  --input data/eval/hypospace_causal_eval.jsonl \
  --provider ollama \
  --model qwen3:1.7b \
  --num-predict 2048 \
  --rollouts-per-spec 1 \
  --transcripts
```

Outputs:

```text
results/envspec_eval/<run>/
  config.json
  episodes.jsonl
  summary.json
```

## Cluster Eval

Serve a base model or checkpoint with SGLang or vLLM using an OpenAI-compatible endpoint, then run:

```bash
MODEL_PATH=Qwen/Qwen3-4B \
EVAL_FILE=data/verl/hypospace_causal_val.parquet \
BASE_URL=http://127.0.0.1:30000/v1 \
RUN_NAME=qwen3_4b_hypospace_eval \
ROLLOUTS_PER_SPEC=4 \
WANDB_PROJECT=scattered-discovery \
scripts/cluster/run_discovery_eval_openai.sh
```

`MODEL_PATH` can be a Hugging Face base model or a trained checkpoint path. The eval runner only needs the served model name to match what the inference server expects.

For scattered-causal validation, use the balanced mixed validation file:

```text
data/verl/scattered_causal_val.parquet
```

It contains all dispersion values in a round-robin balanced mix. For standalone
checkpoint evaluation, run this script against that same file so base,
checkpoint, and training-time validation use the same held-out distribution.

## Metrics

The summary reports:

```text
reward_mean
reward_stdev
valid_unique_count_mean
validity_mean
uniqueness_mean
false_count_mean
non_final_count_mean
unsupported_count_mean
parse_failures_mean
invalid_actions_mean
recovery_mean
model_seconds_total
```

`valid_unique_count_mean` is the mean number of unique valid committed answers
per rollout. `validity_mean` is valid committed answers divided by total
committed answers. `uniqueness_mean` is non-duplicate committed answers divided
by total committed answers. `recovery_mean` is the fraction of the environment's
finite target set recovered.

`non_final_count_mean` tracks true intermediate claims submitted as final
answers. Those claims are not factually false; they are incomplete under the
environment's final-answer schema.

For pass@K-style eval, keep `protocol: single` and set `ROLLOUTS_PER_SPEC=K`. For single-rollout K-answer eval, use `protocol: set` and `max_commit: K` in the eval dataset.

## Scattered Pilot Calibration

Before launching the pilot GRPO run, run base-model calibration evals. These
jobs start an SGLang server inside the Slurm allocation, run the eval, then stop
the server.

Run the easy smoke validation first:

```bash
sbatch --export=ALL,EVAL_CONFIG=configs/verl/eval/scattered_smoke_base_single.yaml \
  scripts/cluster/sbatch_scattered_eval.slurm
```

Then run the mixed pilot subset:

```bash
sbatch --export=ALL,EVAL_CONFIG=configs/verl/eval/scattered_pilot_base_single_subset.yaml \
  scripts/cluster/sbatch_scattered_eval.slurm
```

If the single-rollout pilot subset has weak but nonzero signal, run pass@4:

```bash
sbatch --export=ALL,EVAL_CONFIG=configs/verl/eval/scattered_pilot_base_pass4_probe.yaml \
  scripts/cluster/sbatch_scattered_eval.slurm
```

Use the full pass@4 subset only after the probe shows useful signal:

```bash
sbatch --export=ALL,EVAL_CONFIG=configs/verl/eval/scattered_pilot_base_pass4_subset.yaml \
  scripts/cluster/sbatch_scattered_eval.slurm
```

Each run writes:

```text
results/envspec_eval/<run_name>/
  config.json
  episodes.jsonl
  summary.json
  summary_by_task.json
```

Use `summary.json` for aggregate difficulty and `summary_by_task.json` to check
which task regions are learnable. The stratified summary is broken out by:

```text
dispersion
num_branches
branch_depth
distractors_per_node
base_budget
```

Interpretation:

- High `parse_failures_mean`: prompt/action-format problem.
- High `invalid_actions_mean`: admissibility/search problem.
- High `non_final_count_mean`: model finds true partials but misses terminal paths.
- High `unsupported_count_mean`: model commits without enough evidence.
- Low single-rollout signal but nonzero pass@4: GRPO has something to amplify.
- Near-zero pass@4: simplify or shape before running the pilot.

For scattered-causal eval, keep the rollout `max_steps` at least as large as the
largest visible `base_budget` in the evaluated split. Otherwise the model sees a
larger remaining experiment budget than the outer rollout loop will actually
allow. The scattered smoke preset uses `max_steps: 10` for `base_budget: 10`;
the pilot presets use `max_steps: 13` for `base_budget: [9, 11, 13]`.
