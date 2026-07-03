# Blackwell Workstation Runs

This machine has no Slurm scheduler. Use the direct scripts in this directory.
By default, caches, models, datasets, checkpoints, temporary files, and W&B files
are placed under `/scratch/$CSRID/open-discovery/` when `/scratch/$CSRID` exists
or `/scratch` is writable. If scratch is not writable, the scripts fall back to:

```text
$HOME/open-discovery/
```

For causal micro-lab eval, the Blackwell wrapper overrides the shared YAML's
repo-relative paths so generated rows and results go to:

```text
$BLACKWELL_RUN_ROOT/data/causal_micro_lab/
$BLACKWELL_RUN_ROOT/artifacts/causal_micro_lab_eval/
```

Set `CSRID` if your shell username is not your Cambridge CRSid:

```bash
export CSRID=as3727
```

Initial setup:

```bash
bash scripts/blackwell/setup_env.sh
```

Run the causal micro-lab final eval on one GPU:

```bash
bash scripts/blackwell/run_causal_micro_lab_eval.sh
```

Run causal micro-lab eval directly through Transformers, without SGLang. This is
slower, but useful for SFT smoke checks and avoids SGLang/TorchAO serving issues:

```bash
bash scripts/blackwell/run_causal_micro_lab_eval_hf.sh
```

Run one-GPU causal micro-lab LoRA SFT:

```bash
bash scripts/blackwell/run_causal_micro_lab_sft.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=3 bash scripts/blackwell/run_causal_micro_lab_eval.sh
EVAL_WORKERS=96 bash scripts/blackwell/run_causal_micro_lab_eval.sh
MODEL_PATH=/scratch/$CSRID/open-discovery/checkpoints/my_model DOWNLOAD_MODEL=0 bash scripts/blackwell/run_causal_micro_lab_eval.sh
WANDB_PROJECT= bash scripts/blackwell/run_causal_micro_lab_eval.sh
MODEL_PATH=/homes/$CSRID/open-discovery/checkpoints/causal_micro_lab_sft/causal_micro_lab_sft_smoke/merged DOWNLOAD_MODEL=0 MAX_EXAMPLES=32 bash scripts/blackwell/run_causal_micro_lab_eval_hf.sh
```

Useful SFT overrides:

```bash
SFT_EPOCHS=2 bash scripts/blackwell/run_causal_micro_lab_sft.sh
SFT_MAX_STEPS=20 RUN_NAME=causal_micro_lab_sft_smoke bash scripts/blackwell/run_causal_micro_lab_sft.sh
SFT_BATCH_SIZE=4 SFT_GRAD_ACCUM=8 bash scripts/blackwell/run_causal_micro_lab_sft.sh
SFT_RESUME_FROM_CHECKPOINT=/homes/$CSRID/open-discovery/checkpoints/causal_micro_lab_sft/causal_micro_lab_sft_qwen3_4b_lora/checkpoint-500 bash scripts/blackwell/run_causal_micro_lab_sft.sh
```

LoRA runs save both the adapter at `final/` and a merged serving model at
`merged/`. Use the merged directory for SGLang eval:

```bash
MODEL_PATH=/homes/$CSRID/open-discovery/checkpoints/causal_micro_lab_sft/causal_micro_lab_sft_qwen3_4b_lora/merged DOWNLOAD_MODEL=0 THINK=true bash scripts/blackwell/run_causal_micro_lab_eval.sh
```

The default eval config is `configs/verl/eval/causal_micro_lab_test_k16.yaml`.
It emits prefix summaries for `k=4`, `k=8`, and `k=16` from the same generated
samples.
