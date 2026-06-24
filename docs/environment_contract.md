# Environment Contract

The training backend is intentionally environment-agnostic. Every task family implements the same contract.

## Data Flow

```text
YAML/CLI config
  -> EnvSpec
  -> dataset row with env_spec_json
  -> veRL DiscoveryAgentLoop
  -> DiscoveryEnv.reset()
  -> model ACTION
  -> DiscoveryEnv.step()
  -> final DiscoveryScore
```

## Module Layout

Environment implementations live under `src/scattered_discovery/envs/`:

```text
base.py                  shared DiscoveryEnv, EnvSpec, DiscoveryStep, DiscoveryScore
factory.py               EnvSpec -> concrete environment
scattered_causal.py      scattered causal engine plus generic adapter
scattered_dsl.py         scattered-causal ACTION/COMMIT parser
scattered_evidence.py    scattered-causal evidence model and store
scattered_world.py       scattered-causal world generator
hypospace_causal.py      active causal-graph HypoSpace env
hypospace_boolean.py     active Boolean-expression HypoSpace env
hypospace_3d.py          active 3D reconstruction HypoSpace env
```

Only `config.py` lives directly at the package root. New code should import environment code from explicit `scattered_discovery.envs.*` modules.

## Prompt Locations

All prompt text lives under `src/scattered_discovery/prompts/`:

```text
prompts/scattered.py    scattered-causal system, initial, observation, final commit, repair, and Qwen finalizer prompts
prompts/hypospace.py    HypoSpace causal, Boolean, and 3D initial prompts
prompts/generic.py      generic local and veRL system prompts plus next-action observation prompt
```

Environment modules call these prompt helpers from `reset()` or from rollout/agent-loop message assembly. Environment observation strings still live next to the environment mechanics because they are state-dependent results, not prompt templates.

Every `DiscoveryEnv` owns the same prompt surface:

```text
system_prompt(runtime)       system message for local or veRL runtime
reset()                      initial user prompt
observation_prompt(step)     user prompt after an environment step
```

`scattered_causal` uses a richer observation prompt that includes updated public state and evidence summaries. HypoSpace envs use the shared generic observation prompt because their step observations already contain the task result. Internal version-space sizes stay in `metrics` and `diagnostics` unless a task explicitly sets `show_version_space_size: true` for debugging.

## EnvSpec

Every environment is configured by:

```json
{
  "env_type": "scattered_causal",
  "task": {},
  "protocol": "single",
  "max_steps": 8,
  "max_commit": 1,
  "seed": 0
}
```

Supported `env_type` values:

```text
scattered_causal
hypospace_causal
hypospace_boolean
hypospace_3d
```

## Dataset Row

Each row written by `scattered-discovery-make-dataset` contains:

```json
{
  "index": 0,
  "data_source": "scattered_causal",
  "agent_name": "discovery_agent_loop",
  "prompt": "Interactive discovery task. The custom AgentLoop will construct the full prompt.",
  "raw_prompt": "Interactive discovery task.",
  "env_spec_json": "...",
  "reward_model": {"style": "rule"}
}
```

`env_spec_json` is the source of truth. The prompt fields exist so veRL's dataset loader sees normal prompt columns.

## DiscoveryEnv

Every interactive environment exposes:

```text
reset() -> initial user prompt
system_prompt(runtime) -> system message
observation_prompt(step, runtime) -> post-step user prompt
step(model_text_or_action) -> DiscoveryStep
force_finalize() -> DiscoveryScore
diagnostics() -> dict
done -> bool
```

`DiscoveryStep` contains:

```text
observation
done
parse_ok
action_text
reward
score
metrics
debug
```

`DiscoveryScore` contains:

```text
reward
breakdown
valid_keys
valid_branch_ids
valid_committed_count
valid_unique_count
committed_count
false_count
non_final_count
unsupported_count
duplicate_count
validity
uniqueness
parse_failures
invalid_actions
metrics
reward_vector
```

## Reward Components

The default training profile is `terminal_only`. Here "terminal" means
end-of-episode RL reward, not a special model-visible hypothesis label. It
returns only the final answer reward:

```text
valid_hypothesis     positive reward per unique valid final hypothesis
```

Use this for the clean GRPO baseline and collapse/diversity experiments.

There is no reward component named coverage. The clean reward is:

```text
reward = valid_hypothesis_reward * valid_unique_count
```

The scattered pilot can use `terminal_clean_invalid_bonus`, which keeps the
terminal reward sparse but adds `clean_invalid_final = 0.2` when the model makes
a parseable final `COMMIT`, finds no valid target, and had no parse failures or
invalid actions earlier in the rollout. This is meant to distinguish clean
wrong-final behavior from malformed exploration without adding dense per-step
format shaping.

Recovery is an evaluation metric computed from the verifier's target set:

```text
recovery = valid_unique_count / target_count
recovery_at_k = unique valid target IDs found across K rollouts / target_count
```

`validity` and `uniqueness` are separate diagnostics:

```text
validity = valid_committed_count / committed_count
uniqueness = (committed_count - duplicate_count) / committed_count
```

Committed answers are classified with separate predicates:

```text
false_count       factually false under the hidden world/verifier
non_final_count   true intermediate claims that are incomplete as final answers
unsupported_count final-form claims without enough episode evidence/support
```

The opt-in `shaped` profile adds the remaining components:

```text
format               small reward for parseable action format
admissible           small reward for valid/admissible intermediate actions
commit_format        small reward for syntactically valid commit
invalid_action       penalty for syntax/action failures
false_commit         penalty for false committed hypotheses
non_final_commit     penalty for true but incomplete final submissions
unsupported_commit   penalty for final-form hypotheses without enough support
duplicate_commit     reserved; duplicates in set COMMIT zero final reward
clean_invalid_final  small bonus for clean parseable invalid final answers
budget               optional budget penalty
```

Default values and per-env reward configs are centralized in:

```text
src/scattered_discovery/rewards.py
```

Inspect them with:

```bash
UV_CACHE_DIR=.uv-cache uv run scattered-discovery-rewards
```

Dataset rows can override an env's defaults with `task.reward`.
They can also select a named profile with `task.reward_profile` or
`task.reward.profile`.

There is no separate duplicate penalty. In `set` protocol, duplicate committed
answers make the set fail the uniqueness condition and zero the final reward.
In `single` protocol/pass@K evaluation, duplicate answers across independent
rollouts are not penalized.

This means vanilla GRPO can run against a clean final scalar reward. Shaped
reward is available as an explicit ablation or bootstrapping profile when the
base model struggles with syntax.

## Final Answers And Hidden Verifiers

The model is not expected to know hidden target membership. It only sees the
public final-answer schema and the interaction protocol.

For `scattered_causal`, the prompt says a final answer is a complete directed
path of the required length with evidence gathered in the episode. Model-facing
observations report raw measurements and sample counts. The environment still
maintains posterior evidence, accepted/rejected claim sets, and verifier labels
internally for scoring and diagnostics, but those fields are hidden from prompts
by default. Set `agent.include_evidence_status_in_prompt: true` only for debug
or curriculum runs where this leak is intentional. The internal world calls
complete paths terminal graph leaves, but the model-facing task wording is
"valid final causal hypothesis."

For HypoSpace envs, a final answer is a complete object in the task schema:
`graph(...)` for causal graphs, a Boolean expression, or a 3D structure. The
environment validates whether committed objects are in the remaining compatible
hypothesis set and logs recovery against that finite target set.

For a real-world environment, this has to be supplied by the environment
designer: define the `COMMIT` schema, define what evidence/query state makes an
answer valid, and implement a verifier. Without a verifier or judged target set,
we can still run interaction logs, but we cannot produce the clean GRPO reward
or exact recovery metrics.

## Protocols

`single`:

```text
One final COMMIT.
Use ROLLOUT_N=K for pass@K-style independent samples.
```

`set`:

```text
One rollout can use COMMIT [hypothesis; hypothesis; ...] with up to max_commit hypotheses.
Use this when testing set-reward behavior or single-rollout K-answer behavior.
```

## Qwen3 and Multi-Turn

The veRL loop appends generated model tokens to the response with loss mask 1, and environment observation tokens with loss mask 0. Observation tokens are generated with the fixed-base Qwen3 method in `qwen3_tokenization.py` to avoid chat-template issues with prior reasoning content.

## Adding a New Environment

To add another interactive environment:

```text
1. Implement reset, step, force_finalize, diagnostics, and done.
2. Return DiscoveryStep and DiscoveryScore objects.
3. Register it in src/scattered_discovery/envs/factory.py.
4. Add dataset YAML task examples.
5. Add at least one deterministic unit test.
```
