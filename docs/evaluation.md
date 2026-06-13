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
