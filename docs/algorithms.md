# Algorithm Recipes

The repo separates three layers:

```text
DiscoveryEnv        interaction, observations, reward construction
AgentLoop           multi-turn rollout and token masks
Algorithm recipe    veRL trainer overrides / loss family
```

Algorithm recipes live in `src/scattered_discovery/algos/`.

## Current Recipes

`grpo`

```text
Vanilla veRL GRPO.
Emits:
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
```

`set_reward_grpo`

```text
Same veRL objective as GRPO.
The difference is environment-level: dataset rows use protocol=set and max_commit=K.
Use this for Puri-style single-rollout K-answer set rewards.
```

`echo_grpo`

```text
Experimental scaffold.
Intended objective: GRPO action-token loss + lambda * env-observation CE.
Requires a veRL trainer patch and observation-token masks before use.
```

List available recipes:

```bash
UV_CACHE_DIR=.uv-cache uv run scattered-discovery-algo-overrides --list
```

Print overrides:

```bash
UV_CACHE_DIR=.uv-cache uv run scattered-discovery-algo-overrides --algo grpo
```

## Cluster Launch

`scripts/cluster/run_verl_discovery_grpo.sh` reads:

```bash
DISCOVERY_ALGO=grpo
```

The default is compatible with current veRL. Experimental recipes that need a custom trainer intentionally fail unless explicitly allowed by the recipe CLI. This prevents accidentally running an ECHO-labeled experiment with a vanilla GRPO loss.

## Adding A Custom Algorithm

Create a child of `VerlAlgorithm`:

```text
src/scattered_discovery/algos/my_algo.py
```

Implement:

```text
name
description
requires_custom_trainer
verl_overrides()
notes()
```

Register it in `src/scattered_discovery/algos/registry.py`.

If the algorithm changes only reward semantics, prefer implementing it in the environment and keep veRL GRPO unchanged. If it changes token-level loss, add a recipe plus a trainer patch.
