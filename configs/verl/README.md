# veRL configurations

This directory contains the veRL data, training, and evaluation configurations
used by HypoDiverse. The public comparison has three conditions:

- unmodified `Qwen/Qwen3-4B`;
- validity-reward GRPO; and
- LIFPO.

## Frozen data

The canonical training configurations read the same files:

```text
data/causal_micro_lab/trainable/verl_train.jsonl
data/causal_micro_lab/trainable/verl_val.jsonl
```

Training and offline-evaluation rows are downloaded rather than duplicated in
Git. The downloader places the frozen evaluation set at:

```text
eval_sets/causal_micro_lab/canonical_eval/verl_test.jsonl
```

Exact reproduction should use the published dataset revision documented in the
root `README.md`. Dataset generation is retained for new experiments, but it is
not part of the frozen evaluation workflow.

Download and SHA256-verify the frozen rows with:

```bash
hypodiverse-download-data
```

## Training

The compute-matched training configurations are:

```text
runs/causal_micro_lab_cluster_grpo.yaml
runs/causal_micro_lab_cluster_lifpo.yaml
```

Both use Qwen3-4B, two GPUs, 16 prompts per update, eight rollouts per prompt,
a maximum response length of 6000 tokens, and 100 updates. They differ only in
the learning objective and the metadata required by that objective.

Launch them with:

```bash
sbatch scripts/cluster/sbatch_causal_micro_lab_grpo.slurm
sbatch scripts/cluster/sbatch_causal_micro_lab_lifpo.slurm
```

Run configuration fields are converted to environment variables by
`scripts/cluster/load_run_config.py`. Explicit environment variables take
precedence, which allows cluster-specific paths to be supplied without editing
the committed YAML files.

## Evaluation

The released-model configurations are:

```text
eval/hypodiverse_base.yaml
eval/hypodiverse_grpo.yaml
eval/hypodiverse_lifpo.yaml
```

Each condition generates one ordered bank of 16 completions per state. Metrics
at budgets 4, 8, and 12 are computed from prefixes of the same bank. Generation
uses temperature 1.0, top-p 1.0, a 6000-token thinking pass, and a deterministic
256-token non-thinking fallback only when no final answer is emitted.

Submit all three conditions with:

```bash
bash scripts/cluster/submit_hypodiverse_evals.sh
```

The evaluator downloads the released GRPO and LIFPO repositories from Hugging
Face and logs live metrics to W&B. See
`docs/causal_micro_lab_reproducibility.md` for the immutable dataset revision,
checkpoint steps, and report-generation procedure.

## Agent loops

`agent_loop.yaml` registers the standard interactive and causal-micro-lab
loops. `lifpo_agent_loop.yaml` registers the single-shot LIFPO loop. Dataset
rows retain `causal_micro_lab_agent_loop` as their routing label so the same
frozen rows can be consumed by either compute-matched training condition.

The causal-micro-lab rows contain `env_spec_json`, `state_json`, `prompt`,
`raw_prompt`, and `agent_name`. Private verifier fields are available to the
local reward implementation but are never rendered in the model prompt.
