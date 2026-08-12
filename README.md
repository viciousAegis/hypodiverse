# HypoDiverse

HypoDiverse is an enumerable synthetic benchmark for measuring whether language
models generate **predictively diverse sets of valid scientific hypotheses**.
It accompanies the thesis *Beyond Correctness: Measuring and Optimising
Predictive Diversity in Verifiable Hypothesis Generation*.

The benchmark uses small Boolean causal systems with three observed inputs, two
intermediate variables, and one output. A prompt reveals a subset of
observational or interventional evidence. The model must return a causal
program consistent with that evidence. Because all programs and experiments are
enumerable, the verifier can determine exactly:

- whether an output parses and is evidence-consistent;
- which distinct predictive hypothesis it represents;
- how often a generation bank repeats the same hypothesis; and
- how strongly the best fixed-size subset disagrees on unresolved experiments,
  reported as Diversity@K.

The repository contains the benchmark engine, frozen datasets, GRPO baseline,
Latent Inverse Frequency Policy Optimisation (LIFPO), SGLang/veRL launchers,
evaluation code, and thesis analysis scripts.

## Released Artifacts

The release uses three Hugging Face repositories:

- [Dataset: `viciousa3gis/hypodiverse`](https://huggingface.co/datasets/viciousa3gis/hypodiverse),
  revision `d16867cc49836f72ace9e3667164fa6e4ae76eda`
- Model: `viciousa3gis/hypodiverse-grpo` (published after cluster merge)
- Model: `viciousa3gis/hypodiverse-lifpo` (published after cluster merge)

Model cards pin the immutable dataset revision used by their training and
evaluation. See [the reproducibility guide](docs/causal_micro_lab_reproducibility.md)
for the exact artifact hashes and release procedure.

## Installation

The training stack targets Linux, Python 3.11 or 3.12, CUDA, veRL, and SGLang.
Project dependencies are managed with `uv`.

```bash
git clone https://github.com/viciousAegis/open-discovery.git
cd open-discovery
UV_CACHE_DIR=.uv-cache uv sync --extra verl
source .venv/bin/activate
```

For CPU-only benchmark development and tests, install the package without the
`verl` extra. The pure hypothesis engine and verifier do not require a GPU.

```bash
uv sync
PYTHONPATH=src python -m unittest discover -s tests
```

Cluster launchers load secrets from `.env` through `scripts/env.sh`. Keep this
file untracked. A typical file contains:

```text
WANDB_API_KEY=...
HF_TOKEN=...
```

## Data

The Hugging Face dataset exposes standard JSONL splits:

```python
from datasets import load_dataset

dataset = load_dataset(
    "viciousa3gis/hypodiverse",
    revision="d16867cc49836f72ace9e3667164fa6e4ae76eda",
)
print(dataset["train"][0])
```

The rows retain the exact veRL structure consumed by training. The frozen test
split contains 192 states, balanced across compatible-hypothesis counts
`M = 4, 8, 12, 16`. Its source file is tracked at:

```text
eval_sets/causal_micro_lab/final_v3/verl_test.jsonl
```

Do not regenerate data when reproducing the reported model results. Use the
released files and revision. Dataset generation remains available for new
experiments:

```bash
causal-micro-lab-build-split-dataset \
  --preset trainable \
  --output-dir data/causal_micro_lab/trainable \
  --seed 1
```

## Training

The canonical configurations use Qwen3-4B, two A100 80 GB GPUs, 16 prompts per
update, eight rollouts per prompt, a 6000-token response limit, and 100 updates.
Both methods use the same frozen training rows and generation budget.

```bash
# Validity-reward GRPO baseline
sbatch scripts/cluster/sbatch_causal_micro_lab_validity_grpo.slurm

# LIFPO
sbatch scripts/cluster/sbatch_causal_micro_lab_lifpo.slurm
```

The public LIFPO configuration is
[`configs/verl/runs/causal_micro_lab_cluster_lifpo.yaml`](configs/verl/runs/causal_micro_lab_cluster_lifpo.yaml).
Historical configuration aliases remain importable only so completed runs and
checkpoints can still be resolved.

## Evaluation

The frozen comparison generates one ordered bank of 16 completions per state.
Metrics for budgets 4, 8, 12, and 16 use prefixes of that same bank. This keeps
the budget comparison paired and avoids rerunning independent samples.

```bash
bash scripts/cluster/submit_causal_micro_lab_v3_evals.sh
```

The evaluator uses:

```text
temperature             1.0
top_p                    1.0
maximum response         6000 tokens
thinking                 enabled
empty-answer fallback    deterministic, 256 tokens
SGLang memory fraction   0.82
```

The primary measurements are:

- **Pass@B:** fraction of states with at least one valid output within raw
  generation budget `B`;
- **Validity@B:** fraction of the first `B` outputs that are valid;
- **distinct valid hypotheses:** number of unique valid predictive programs;
- **duplicate share:** repeated valid hypotheses as a fraction of valid outputs;
- **Diversity@K:** mean pairwise predictive disagreement of the most diverse
  `K` hypotheses recovered within a fixed valid-generation budget.

Evaluation outputs are written beneath
`artifacts/causal_micro_lab_final_eval_v3/` and logged to W&B. The reporting
pipeline derives all budget prefixes from the same episode bank.

## Publishing Exact Artifacts

Publishing is a CPU/network operation. The dataset was staged, hashed, and
published from the fixed local splits. Run the model stage on the cluster where
the evaluated merged checkpoints reside:

```bash
source scripts/env.sh
source "$VENV_DIR/bin/activate"

bash scripts/cluster/publish_causal_micro_lab_hf.sh \
  --namespace viciousa3gis \
  --models-only \
  --dataset-revision d16867cc49836f72ace9e3667164fa6e4ae76eda \
  --push
```

Run the same command without `--push` first. It refuses to substitute a
different checkpoint. Re-running an interrupted Hugging Face upload resumes
through `huggingface_hub`/Xet.

## Repository Map

```text
src/scattered_discovery/envs/causal_micro_lab/  enumeration, prompts, verifier
src/scattered_discovery/verl/                   GRPO and LIFPO integration
configs/verl/runs/                              canonical training configs
configs/verl/eval/                              frozen evaluation configs
eval_sets/causal_micro_lab/final_v3/            frozen test set and manifest
scripts/cluster/                                Slurm training/evaluation tools
scripts/                                       analysis and plotting
tests/                                         engine, verifier, metric, launcher tests
docs/causal_micro_lab_reproducibility.md        exact end-to-end procedure
```

The repository also retains earlier exploratory environments and analyses. They
are not required to reproduce the HypoDiverse results above.

## Reproducibility Contract

An exact reproduction must keep the following fixed:

1. the published dataset revision;
2. Qwen3-4B base weights and tokenizer;
3. the method-specific YAML configuration;
4. the ordered 16-completion evaluation bank and prefix budgets;
5. the evaluator and analysis code revision; and
6. the merged checkpoint identified in the model card.

The release manifests record SHA256 hashes, row counts, code revision, source
configuration hashes, and split-overlap checks. See the detailed
[release checklist](docs/causal_micro_lab_reproducibility.md#verification-checklist)
before publishing derived results.
