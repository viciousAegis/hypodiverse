# HypoDiverse: Release and Reproducibility

This document defines the release order and provenance required to reproduce the
thesis results. The release has three parts:

1. publish the exact training, validation, and frozen evaluation data;
2. publish the exact merged model checkpoints used for evaluation;
3. publish and tag the cleaned code, using **LIFPO** as the public method name.

The order matters. Model cards must pin the immutable commit SHA of the dataset
release, and the final code tag must describe the already-published artifacts.

## Frozen artifacts

### Data

Training, validation, and offline-evaluation rows are not duplicated in Git.
The exact inputs used by GRPO and LIFPO are downloaded from the immutable
Hugging Face dataset revision and materialized at:

```text
data/causal_micro_lab/trainable/verl_train.jsonl
data/causal_micro_lab/trainable/verl_val.jsonl
```

The construction is deterministic. The release splits are generated with the
canonical run configuration and then frozen. Publishing copies those files; it
does not regenerate them. Their expected row counts and SHA256 hashes are:

| Split | Rows | SHA256 |
|---|---:|---|
| train | 6144 | `90b21e2532757ad7661f3f61d3e14901d20e167ad0525318a7e03378151cc2f4` |
| validation | 128 | `5865f9d985becd6d33cdbbe62090c8baac450282716760c838ba40064d0d5e6f` |
| test | 192 | `4e6ee134a105276b91ebdedab55afe3f5af53689f7c3f666b037cfa0aae36967` |

The frozen evaluation input is materialized at:

```text
eval_sets/causal_micro_lab/canonical_eval/verl_test.jsonl
```

It contains 192 states: 48 each with compatible-hypothesis count
`M in {4, 8, 12, 16}`. Its SHA256 digest is:

```text
4e6ee134a105276b91ebdedab55afe3f5af53689f7c3f666b037cfa0aae36967
```

The separate 128-world closed-loop state set is version-controlled at
`eval_sets/causal_micro_lab/closed_loop_v1/`. Its manifest records the exact
state-file digest. This is deliberate: its original selection excluded local
historical datasets that are not in the release, so regenerating it in a clean
checkout would not recover the evaluated worlds.

### Evaluated models

The thesis compares three model conditions:

| Public name | Evaluated weights |
|---|---|
| Base | Unmodified Qwen3-4B base checkpoint |
| GRPO | Frozen actor checkpoint at `global_step_90` |
| LIFPO | Frozen actor checkpoint at `global_step_55` |

The publisher resolves the exact GRPO and LIFPO source artifacts from canonical
release directories. The release manifest records their checkpoint steps and
per-file hashes.

The evaluated step-90 and step-55 models are distinct from the canonical
100-update reproduction configurations. A 100-update reproduction run is the
recommended clean rerun of each method; it is not the source of the frozen
model weights above. Model cards must report both facts without implying that
the evaluated artifacts were taken at update 100.

## Exact evaluation protocol

The frozen comparison uses one ordered bank of `K = 16` completions per state.
Metrics at `K = 4, 8, 12, 16` are computed from prefixes of that same bank.

| Setting | Value |
|---|---:|
| Maximum response length | 6000 tokens |
| Thinking | enabled |
| Fallback | deterministic non-thinking pass, 256 tokens |
| Temperature | 1.0 |
| Top-p | 1.0 |
| SGLang GPU memory fraction | 0.82 |

Using independently generated banks for different values of `K` is not an
exact reproduction because it breaks the paired-prefix comparison.

## Hugging Face layout

```text
viciousa3gis/hypodiverse
viciousa3gis/hypodiverse-grpo
viciousa3gis/hypodiverse-lifpo
```

The frozen dataset revision is
`d16867cc49836f72ace9e3667164fa6e4ae76eda`.

The dataset repository has the following layout:

```text
README.md
data/
  train.jsonl
  validation.jsonl
  test.jsonl
source/
  trainable/                 # exact cluster source files and manifests
  final_v3/                  # frozen source files in the published revision
configs/
  training/
  evaluation/
release_manifest.json       # paths, row counts, sizes, and SHA256 digests
```

`data/train.jsonl`, `data/validation.jsonl`, and `data/test.jsonl` are
byte-for-byte copies of the corresponding veRL JSONL files. This makes the
repository directly readable with the Hugging Face `datasets` JSON loader while
preserving the original rows used by training and evaluation.

Each model repository contains the merged Hugging Face weights, tokenizer,
configuration, model card, and a release manifest. Its model card records:

- the public method name;
- the exact checkpoint step;
- the base model;
- the training and evaluation configuration files;
- the dataset repository and immutable dataset commit SHA;
- hashes of all merged model files;
- the frozen evaluation protocol above.

The machine-readable manifest additionally records the checkpoint step and
source configuration hashes needed to audit the release.

## Ordered release procedure

Run model commands on the cluster checkout containing the merged evaluated
checkpoints. Load the project environment and authenticate to the Hub without
printing the token.

```bash
cd /path/to/hypodiverse
source scripts/env.sh
source .venv/bin/activate
export HF_NAMESPACE="viciousa3gis"
```

### 1. Confirm the exact source files

Download and verify the immutable splits. This command checks each SHA256
digest before placing the file at its canonical local path:

```bash
hypodiverse-download-data
```

```bash
test -f data/causal_micro_lab/trainable/verl_train.jsonl
test -f data/causal_micro_lab/trainable/verl_val.jsonl
test -f eval_sets/causal_micro_lab/canonical_eval/verl_test.jsonl

sha256sum \
  data/causal_micro_lab/trainable/verl_train.jsonl \
  data/causal_micro_lab/trainable/verl_val.jsonl \
  eval_sets/causal_micro_lab/canonical_eval/verl_test.jsonl
```

The final line must contain the frozen evaluation digest stated above. Do not
invoke dataset generation as a fallback.

### 2. Package and publish the dataset

First build and verify a local staging tree:

```bash
python -m scattered_discovery.release.causal_micro_lab dataset \
  --train-file data/causal_micro_lab/trainable/verl_train.jsonl \
  --validation-file data/causal_micro_lab/trainable/verl_val.jsonl \
  --test-file eval_sets/causal_micro_lab/canonical_eval/verl_test.jsonl \
  --output-dir artifacts/hf_release/hypodiverse \
  --repo-id "$HF_NAMESPACE/hypodiverse"
```

After inspecting `release_manifest.json`, publish the same staging tree:

```bash
python -m scattered_discovery.release.causal_micro_lab dataset \
  --train-file data/causal_micro_lab/trainable/verl_train.jsonl \
  --validation-file data/causal_micro_lab/trainable/verl_val.jsonl \
  --test-file eval_sets/causal_micro_lab/canonical_eval/verl_test.jsonl \
  --output-dir artifacts/hf_release/hypodiverse \
  --repo-id "$HF_NAMESPACE/hypodiverse" \
  --push
```

Record the returned Hub commit SHA. The cluster wrapper performs the same
operation and writes the revision for the model-release stage:

```bash
bash scripts/cluster/publish_causal_micro_lab_hf.sh \
  --namespace "$HF_NAMESPACE" \
  --dataset-only \
  --push
```

### 3. Locate or merge the evaluated checkpoints

The publisher expects canonical merged GRPO and LIFPO directories below
`$MODEL_ROOT/eval_checkpoints`. The exact evaluated source checkpoints are GRPO
step 90 and LIFPO step 55. Point `GRPO_ACTOR_DIR` and `LIFPO_ACTOR_DIR` at those
FSDP actor directories, then merge them with veRL's standard merger:

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$GRPO_ACTOR_DIR" \
  --target_dir "$MODEL_ROOT/eval_checkpoints/hypodiverse-grpo-step-90"

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$LIFPO_ACTOR_DIR" \
  --target_dir "$MODEL_ROOT/eval_checkpoints/hypodiverse-lifpo-step-55"
```

The release registry pins GRPO to step 90 and LIFPO to step 55. Explicit
`--model-dir` arguments are accepted so the frozen merged checkpoints do not
need to be renamed before packaging.

### 4. Publish the evaluated models

Pin the dataset revision returned by step 2:

```bash
export CML_DATASET_REVISION="d16867cc49836f72ace9e3667164fa6e4ae76eda"
```

Package and inspect each model without uploading:

```bash
python -m scattered_discovery.release.causal_micro_lab model \
  --method grpo \
  --dataset-repo-id "$HF_NAMESPACE/hypodiverse" \
  --dataset-revision "$CML_DATASET_REVISION" \
  --repo-id "$HF_NAMESPACE/hypodiverse-grpo"

python -m scattered_discovery.release.causal_micro_lab model \
  --method lifpo \
  --dataset-repo-id "$HF_NAMESPACE/hypodiverse" \
  --dataset-revision "$CML_DATASET_REVISION" \
  --repo-id "$HF_NAMESPACE/hypodiverse-lifpo"
```

Repeat both commands with `--push` after the manifests have been checked. The
cluster wrapper can publish both models in order:

```bash
bash scripts/cluster/publish_causal_micro_lab_hf.sh \
  --namespace "$HF_NAMESPACE" \
  --models-only \
  --dataset-revision "$CML_DATASET_REVISION" \
  --push
```

To run the complete ordered workflow after a dry run:

```bash
bash scripts/cluster/publish_causal_micro_lab_hf.sh \
  --namespace "$HF_NAMESPACE" \
  --all \
  --push
```

### 5. Evaluate the released models

The released-model launcher downloads the base, GRPO, and LIFPO repositories
and evaluates all three with the frozen protocol:

```bash
bash scripts/cluster/submit_hypodiverse_evals.sh
```

### 6. Tag the code

Only after the data and model repositories are immutable:

1. record the three Hub repository revisions in the root README;
2. run the test and smoke-evaluation suite;
3. create the final public Git tag and release from the tested commit.

This keeps public terminology consistent without breaking frozen checkpoints or
obscuring which source artifact generated each released model.

## Verification checklist

### Dataset

- [ ] Training and validation files came from the completed cluster run, not a
      regeneration.
- [ ] Source and staged SHA256 hashes match for all three split files.
- [ ] `test.jsonl` has 192 rows and the required frozen SHA256 digest.
- [ ] The test distribution is exactly 48 rows for each `M = 4, 8, 12, 16`.
- [ ] Every JSONL line parses as one object.
- [ ] Stable state IDs have no overlap across train, validation, and test.
- [ ] Private verifier fields are absent from rendered prompts.
- [ ] The Hub dataset loads with:

```bash
python - <<'PY'
from datasets import load_dataset

dataset = load_dataset(
    "viciousa3gis/hypodiverse",
    revision="d16867cc49836f72ace9e3667164fa6e4ae76eda",
)
print(dataset)
print({name: len(split) for name, split in dataset.items()})
PY
```

### Models

- [ ] GRPO resolves to the frozen step-90 actor checkpoint.
- [ ] LIFPO resolves to the frozen step-55 actor checkpoint.
- [ ] All expected FSDP shards existed before merging.
- [ ] The merged directories contain model weights, `config.json`, and tokenizer
      files.
- [ ] Local and Hub release manifests contain matching file hashes.
- [ ] Both Hub models load with `AutoTokenizer` and `AutoModelForCausalLM`.
- [ ] A deterministic one-row generation smoke test succeeds for each model.
- [ ] A frozen-eval smoke uses the 6000-token thinking pass and 256-token
      non-thinking fallback.
- [ ] Model cards pin the same immutable dataset commit SHA.

### Code release

- [ ] Public prose and current launchers say LIFPO.
- [ ] Canonical GRPO and LIFPO reproduction configs each specify 100 updates.
- [ ] The exact evaluated step-90 and step-55 artifacts are not described as
      update-100 checkpoints.
- [ ] The final README pins dataset, model, and code revisions.
- [ ] The public Git tag is created only after all release smoke tests pass.
