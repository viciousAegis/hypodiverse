# HypoDiverse

HypoDiverse is an enumerable synthetic benchmark for measuring whether language
models generate **predictively diverse sets of valid scientific hypotheses**.
It accompanies the thesis *Measuring Hypothesis Diversity in
Verifiable Scientific Generation*.

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

The repository contains the benchmark engine, a pinned dataset downloader, the
GRPO baseline, Latent Inverse Frequency Policy Optimisation (LIFPO), SGLang/veRL
launchers, evaluation code, and thesis analysis scripts. Dataset rows are
published separately on Hugging Face and are not tracked in Git.

## Released Artifacts

The release uses three Hugging Face repositories:

- [Dataset: `viciousa3gis/hypodiverse`](https://huggingface.co/datasets/viciousa3gis/hypodiverse),
  revision `d16867cc49836f72ace9e3667164fa6e4ae76eda`
- [GRPO model: `viciousa3gis/hypodiverse-grpo`](https://huggingface.co/viciousa3gis/hypodiverse-grpo)
- [LIFPO model: `viciousa3gis/hypodiverse-lifpo`](https://huggingface.co/viciousa3gis/hypodiverse-lifpo)

Model cards pin the immutable dataset revision used by their training and
evaluation. See [the reproducibility guide](docs/causal_micro_lab_reproducibility.md)
for the exact artifact hashes and release procedure.

## Installation

The training stack targets Linux, Python 3.11 or 3.12, CUDA, veRL, and SGLang.
Project dependencies are managed with `uv`.

```bash
git clone https://github.com/viciousAegis/hypodiverse.git
cd hypodiverse
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
`M = 4, 8, 12, 16`. Training and offline-evaluation rows are not duplicated in
this Git repository. The downloader materializes and verifies all three frozen
splits at their canonical local paths:

```bash
hypodiverse-download-data
```

Do not regenerate data when reproducing the reported model results. Use the
released files and revision. Dataset generation remains available for new
experiments:

```bash
hypodiverse-build-dataset \
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
sbatch scripts/cluster/sbatch_causal_micro_lab_grpo.slurm

# LIFPO
sbatch scripts/cluster/sbatch_causal_micro_lab_lifpo.slurm
```

The public LIFPO configuration is
[`configs/verl/runs/causal_micro_lab_cluster_lifpo.yaml`](configs/verl/runs/causal_micro_lab_cluster_lifpo.yaml).

## Evaluation

The frozen comparison generates one ordered bank of 16 completions per state.
Metrics for budgets 4, 8, 12, and 16 use prefixes of that same bank. This keeps
the budget comparison paired and avoids rerunning independent samples.

```bash
bash scripts/cluster/submit_hypodiverse_evals.sh
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

Evaluation outputs are written beneath `artifacts/hypodiverse_eval/` and logged
to W&B. The reporting
pipeline derives all budget prefixes from the same episode bank.

The canonical comparison tables and figures can then be generated from the
three report directories:

```bash
python scripts/analyze_hypodiverse_results.py \
  --report base=/path/to/base/report \
  --report grpo=/path/to/grpo/report \
  --report lifpo=/path/to/lifpo/report
```

For the model-backed closed loop, submit the same three released models with:

```bash
sbatch scripts/cluster/sbatch_causal_micro_lab_closed_loop_eval.slurm base
sbatch scripts/cluster/sbatch_causal_micro_lab_closed_loop_eval.slurm grpo
sbatch scripts/cluster/sbatch_causal_micro_lab_closed_loop_eval.slurm lifpo
```

The small closed-loop world set is version-controlled under
`eval_sets/causal_micro_lab/closed_loop_v1/` because its original selection
excluded historical local datasets that are not part of the public release.
The launcher verifies its SHA256 digest before evaluation.

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
configs/verl/runs/                              GRPO and LIFPO training configs
configs/verl/eval/hypodiverse_*.yaml             frozen evaluation configs
src/scattered_discovery/release/download.py      pinned dataset downloader
scripts/cluster/                                Slurm training/evaluation tools
scripts/                                       analysis and plotting
tests/                                         engine, verifier, metric, launcher tests
docs/causal_micro_lab_reproducibility.md        exact end-to-end procedure
```

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

## License

HypoDiverse is released under the [Apache License 2.0](LICENSE).
