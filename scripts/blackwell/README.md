# Blackwell Workstation Runs

This machine has no Slurm scheduler. Use the direct scripts in this directory.
All caches, models, datasets, checkpoints, temporary files, and W&B files are
placed under:

```text
/scratch/$CSRID/open-discovery/
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

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=3 bash scripts/blackwell/run_causal_micro_lab_eval.sh
EVAL_WORKERS=96 bash scripts/blackwell/run_causal_micro_lab_eval.sh
MODEL_PATH=/scratch/$CSRID/open-discovery/checkpoints/my_model DOWNLOAD_MODEL=0 bash scripts/blackwell/run_causal_micro_lab_eval.sh
WANDB_PROJECT= bash scripts/blackwell/run_causal_micro_lab_eval.sh
```

The default eval config is `configs/verl/eval/causal_micro_lab_test_k16.yaml`.
It emits prefix summaries for `k=4`, `k=8`, and `k=16` from the same generated
samples.
