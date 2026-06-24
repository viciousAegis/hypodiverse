# veRL Cluster Runs

The cluster path uses:

```text
dataset row -> EnvSpec -> DiscoveryEnv -> veRL AgentLoop -> SGLang rollout -> final reward
```

The default script assumes one node with several GPUs, SGLang rollouts, GRPO, and W&B logging.

Before submitting, copy `.env.example` to `.env` only if you need secrets:

```bash
cp .env.example .env
```

Do not put normal run configuration in `.env`. The split is:

```text
.env                         secrets only: WANDB_API_KEY, HF_TOKEN, OPENAI_API_KEY
configs/verl/runs/*.yaml     model, train/val files, batch sizes, rollout count
configs/verl/datasets/*.yaml dataset generation specs
scripts/env.sh               repo-local cache/path defaults
scripts/cluster/*.slurm      Slurm account, partition, GPU count, modules
```

The Slurm scripts load the cluster modules, source `scripts/env.sh`, create
repo-local cache directories, verify the Python training stack, then call the
run wrapper. The wrapper loads the YAML run config and only then resolves or
downloads the model into `$MODEL_ROOT`. If `$VENV_DIR` is missing, the bootstrap
runs `uv sync --extra verl`; this initializes the project venv but does not
replace a proper veRL/SGLang CUDA install.

Ray runtime sockets are kept out of the repo by default:

```text
RAY_TMPDIR=/tmp/sd-ray-${USER}-${SLURM_JOB_ID}
```

This avoids Ray's `AF_UNIX path length cannot exceed 107 bytes` failure on deep
RDS checkout paths. Durable outputs still go to `data/`, `.cache/models/`,
`.wandb/`, `artifacts/`, and `checkpoints/` under the repo.

Install the CUDA training stack once:

```bash
bash scripts/cluster/install_verl_stack.sh
```

That script installs PyTorch, Ray, and veRL with the SGLang extra into
`$VENV_DIR` using `uv pip install`.

Use `PYTHON_VERSION=3.11` for the training venv. The pinned CUDA 12.1 Torch
stack does not have compatible `torchvision` wheels for Python 3.13. If uv
already created a bad 3.13 venv, rebuild it with:

```bash
RECREATE_VENV=1 bash scripts/cluster/install_verl_stack.sh
```

For slow cluster networking during dependency downloads:

```bash
UV_HTTP_TIMEOUT=300 UV_INSTALL_RETRIES=5 bash scripts/cluster/install_verl_stack.sh
```

Algorithm selection is routed through `DISCOVERY_ALGO`. The default is vanilla
veRL-compatible GRPO:

```bash
DISCOVERY_ALGO=grpo
```

Set-reward experiments still use the same GRPO trainer; the dataset controls
`protocol: set` and `max_commit`. Experimental objectives such as ECHO have
recipes under `src/scattered_discovery/algos/`, but require trainer loss patches
before launch.

## 1. Prepare Datasets

Edit `configs/verl/datasets/all_envs.yaml`, then run:

```bash
UV_CACHE_DIR=.uv-cache uv run --extra verl scattered-discovery-make-dataset \
  --config configs/verl/datasets/all_envs.yaml
```

or:

```bash
DATASET_CONFIG=configs/verl/datasets/all_envs.yaml scripts/cluster/prepare_verl_datasets.sh
```

Default outputs:

```text
data/verl/hypospace_causal_train.parquet
data/verl/hypospace_causal_val.parquet
data/verl/hypospace_boolean_train.parquet
data/verl/hypospace_boolean_val.parquet
data/verl/hypospace_3d_train.parquet
data/verl/hypospace_3d_val.parquet
data/verl/scattered_causal_train.parquet
data/verl/scattered_causal_val.parquet
```

The training script automatically uses a sibling validation file when present.
For example, `TRAIN_FILE=data/verl/hypospace_causal_train.parquet` will use
`data/verl/hypospace_causal_val.parquet` unless `VAL_FILE` is set explicitly.
Scattered-causal validation is a single balanced mixed file by default. The
generic script still supports `VAL_FILES` for custom split validation runs.

Preset scattered-causal dataset configs:

```text
configs/verl/datasets/scattered_smoke.yaml
configs/verl/datasets/scattered_pilot.yaml
```

The smoke preset is an easier pipeline/debug distribution. The pilot preset is
the first target-distribution learning run.

## 2. W&B

Either log in once:

```bash
wandb login
```

or set:

```bash
export WANDB_API_KEY=...
```

Disable W&B:

```bash
sbatch --export=ALL,WANDB_LOGGER='["console"]' \
  scripts/cluster/sbatch_verl_smoke_grpo.slurm
```

## 3. Submit GRPO With Slurm

Cluster runs should be submitted through `sbatch`.

Smoke run:

```bash
sbatch scripts/cluster/sbatch_verl_smoke_grpo.slurm
```

Smoke training defaults live in:

```text
configs/verl/runs/scattered_smoke.yaml
```

Pilot run:

```bash
sbatch scripts/cluster/sbatch_verl_signal_pilot_grpo.slurm
```

Signal-pilot training defaults live in:

```text
configs/verl/runs/scattered_signal_pilot.yaml
```

Full pilot run:

```bash
sbatch scripts/cluster/sbatch_verl_pilot_grpo.slurm
```

Full pilot training defaults live in:

```text
configs/verl/runs/scattered_pilot.yaml
```

Logs are written to:

```text
logs/slurm/%x-%j.out
logs/slurm/%x-%j.err
```

Override runtime variables with `--export=ALL,...`; this is for one-off
overrides, not the normal place to maintain experiment defaults:

```bash
sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-4B,WANDB_LOGGER='["console"]' \
  scripts/cluster/sbatch_verl_smoke_grpo.slurm
```

Override Slurm resources with normal `sbatch` flags:

```bash
sbatch --partition=gpu --gres=gpu:8 --export=ALL,NGPUS_PER_NODE=8,ROLLOUT_N=8 \
  scripts/cluster/sbatch_verl_pilot_grpo.slurm
```

The `.slurm` scripts call the corresponding shell wrappers:

```text
sbatch_verl_smoke_grpo.slurm -> run_verl_smoke_grpo.sh -> configs/verl/runs/scattered_smoke.yaml -> run_verl_discovery_grpo.sh
sbatch_verl_signal_pilot_grpo.slurm -> run_verl_pilot_grpo.sh -> configs/verl/runs/scattered_signal_pilot.yaml -> run_verl_discovery_grpo.sh
sbatch_verl_pilot_grpo.slurm -> run_verl_pilot_grpo.sh -> configs/verl/runs/scattered_pilot.yaml -> run_verl_discovery_grpo.sh
```

## 4. Direct Shell Launch

Use direct shell launch only inside an allocated interactive job or for debugging.

Smoke run:

```bash
scripts/cluster/run_verl_smoke_grpo.sh
```

Defaults:

```text
train rows: 320 mixed, 64 per dispersion
val rows: 80 mixed, 16 per dispersion
train batch: 32
mini-batch: 16
rollouts per prompt: 4
global steps: 10
actor updates: 20
sampled training trajectories: 1280
```

Pilot run:

```bash
scripts/cluster/run_verl_pilot_grpo.sh
```

Defaults:

```text
train rows: 8192
val rows: 1280 mixed, 256 per dispersion
train batch: 64
mini-batch: 32
rollouts per prompt: 4
global steps: 128
actor updates: 256
sampled training trajectories: 32768
```

Skip dataset regeneration if files already exist:

```bash
PREPARE_DATASETS=0 scripts/cluster/run_verl_pilot_grpo.sh
```

Train on scattered causal:

```bash
TRAIN_FILE=data/verl/scattered_causal_train.parquet \
VAL_FILE=data/verl/scattered_causal_val.parquet \
MODEL_ID=Qwen/Qwen3-4B \
NGPUS_PER_NODE=4 \
ROLLOUT_N=4 \
scripts/cluster/run_verl_discovery_grpo.sh
```

Set-reward/Puri-style protocol:

```bash
DISCOVERY_ALGO=set_reward_grpo \
TRAIN_FILE=data/verl/hypospace_causal_set_train.parquet \
MODEL_ID=Qwen/Qwen3-4B \
scripts/cluster/run_verl_discovery_grpo.sh
```

Train on HypoSpace causal:

```bash
TRAIN_FILE=data/verl/hypospace_causal_train.parquet \
VAL_FILE=data/verl/hypospace_causal_val.parquet \
MODEL_ID=Qwen/Qwen3-4B \
NGPUS_PER_NODE=4 \
ROLLOUT_N=4 \
scripts/cluster/run_verl_discovery_grpo.sh
```

Run 8 GPUs:

```bash
NGPUS_PER_NODE=8 \
TRAIN_BATCH_SIZE=128 \
PPO_MINI_BATCH_SIZE=64 \
ROLLOUT_N=8 \
TRAIN_FILE=data/verl/scattered_causal_train.parquet \
VAL_FILE=data/verl/scattered_causal_val.parquet \
scripts/cluster/run_verl_discovery_grpo.sh
```

## 5. Common Overrides

Model and rollout:

```bash
MODEL_ID=Qwen/Qwen3-4B
DOWNLOAD_MODEL=1
MODEL_PATH=/path/to/existing/checkpoint  # optional; overrides MODEL_ID download path
INFER_BACKEND=sglang
ROLLOUT_N=4
ROLLOUT_TP=1
ROLLOUT_GPU_MEM_UTIL=0.55
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=4096
```

Training:

```bash
TRAIN_BATCH_SIZE=64
PPO_MINI_BATCH_SIZE=32
ACTOR_LR=1e-6
KL_LOSS_COEF=0.001
ENTROPY_COEFF=0
TOTAL_EPOCHS=1
SAVE_FREQ=20
TEST_FREQ=5
RESUME_MODE=auto
MAX_ACTOR_CKPT_TO_KEEP=3
MAX_CRITIC_CKPT_TO_KEEP=3
```

Logging and output:

```bash
PROJECT_NAME=scattered-discovery
EXPERIMENT_NAME=scattered_qwen3_4b_seed301
WANDB_LOGGER='["console","wandb"]'
```

Checkpoints go to:

```text
${CHECKPOINT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}
```

## 6. Curriculum

Do not use a curriculum for the first comparison unless the target-distribution
pilot is reward-dead. A curriculum changes the training distribution, so it can
hide whether GRPO itself can climb the target task.

Recommended order:

```text
1. Smoke preset: verify infrastructure, parsing, nonzero rewards, W&B, resume.
2. Pilot preset: train directly on the target scattered-causal distribution.
3. Only add a curriculum if pilot rewards are mostly zero or invalid actions dominate.
```

If a curriculum is needed, keep it identical across methods and report both the
direct-target baseline and the curriculum result. A minimal curriculum should be:

```text
stage 0: easier scattered causal, 50-100 actor updates
stage 1: target scattered causal, 250+ actor updates
stage 2: target/harder mix only if validation is still improving
```

## 7. Crash Recovery

The launch script now makes veRL's resume behavior explicit:

```bash
RESUME_MODE=auto
SAVE_FREQ=20
MAX_ACTOR_CKPT_TO_KEEP=3
MAX_CRITIC_CKPT_TO_KEEP=3
```

With `RESUME_MODE=auto`, rerunning the same command with the same
`PROJECT_NAME` and `EXPERIMENT_NAME` resumes from the latest checkpoint under:

```text
checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_*
```

Resume from a specific checkpoint:

```bash
RESUME_MODE=resume_path \
RESUME_FROM_PATH=checkpoints/scattered-discovery/my_run/global_step_120 \
PROJECT_NAME=scattered-discovery \
EXPERIMENT_NAME=my_run \
scripts/cluster/run_verl_discovery_grpo.sh
```

Start over intentionally:

```bash
RESUME_MODE=disable \
EXPERIMENT_NAME=fresh_run \
scripts/cluster/run_verl_discovery_grpo.sh
```

Practical crash-safety notes:

```text
Lower SAVE_FREQ for expensive jobs or preemptible queues.
Use a stable EXPERIMENT_NAME; the default timestamp creates a new run each launch.
Keep checkpoints on persistent storage, not node-local scratch.
Do not set MAX_*_CKPT_TO_KEEP=1 until resume is tested on the cluster.
```

## 8. Qwen3 Multi-Turn Tokenization

The custom veRL agent loop uses a fixed-base chat history to tokenize environment observation messages on the response side. This follows the veRL Qwen3 multi-turn guidance and avoids unstable deltas when the model chat template drops prior reasoning content.

Relevant files:

```text
src/scattered_discovery/verl/agent_loop.py
src/scattered_discovery/verl/qwen3_tokenization.py
configs/verl/agent_loop.yaml
```

## 9. Debugging Checklist

If training fails before rollouts:

```text
Check TRAIN_FILE exists and is readable.
Check uv sync --extra verl was run in the cluster environment.
Check MODEL_ID can be downloaded or MODEL_PATH is accessible from the cluster.
Check W&B login/API key if logger includes wandb.
```

If rollouts fail:

```text
Check env_spec_json is present in the dataset rows.
Check actor_rollout_ref.rollout.agent.default_agent_loop=discovery_agent_loop.
Check configs/verl/agent_loop.yaml is present from the run directory.
Reduce TRAIN_BATCH_SIZE, ROLLOUT_N, MAX_RESPONSE_LENGTH, or ROLLOUT_GPU_MEM_UTIL.
```

If rewards are all zero:

```text
Inspect reward_parse_failures and reward_invalid_action in W&B.
Run a smaller local spec with scattered-discovery-local.
Lower task difficulty in datasets.yaml.
Use easier scenarios from docs/scenarios.md first.
```
