# ECHO And World Modeling

Reference: <https://arxiv.org/pdf/2605.24517>

## What ECHO Changes

Standard multi-turn GRPO computes log-probs over the full rollout but applies policy-gradient loss only on assistant action tokens. Environment observations are context for later actions, but they are not direct training targets.

ECHO adds an auxiliary environment-prediction loss:

```text
L_total = L_GRPO(action_tokens) + lambda * CE(environment_observation_tokens)
```

The ECHO paper emphasizes:

- no extra rollouts,
- no teacher model,
- same actor forward pass,
- cross-entropy gathered from observation-token positions,
- typical productive `lambda` range around `0.01-0.05`, with `0.05` used in reported experiments.

## Mapping To Our Setting

Our rollouts already interleave:

```text
assistant ACTION
environment observation
assistant ACTION
environment observation
...
final score
```

For scattered causal, observations reveal measured consequences/evidence summaries. For HypoSpace, observations reveal query/intervention outcomes. These are exactly the world-modeling targets ECHO wants: predict what the environment will return after the model's own action.

Why this may help diversity:

- Better observation prediction should improve the model's latent estimate of which hypotheses remain plausible.
- Better latent state tracking should make query choices more information-seeking.
- Better information-seeking should reduce premature convergence to the first valid hypothesis.
- Diversity gains should show up as higher Recovery, higher valid unique count, lower duplicate rate, and sustained entropy/information gain over committed hypotheses.

## Code Changes Needed

Current state:

- `DiscoveryAgentLoop` emits model action tokens with `response_mask=1`.
- It appends environment observation tokens with `response_mask=0`.
- veRL therefore conditions on observations but does not train on them.

Needed for ECHO:

1. Extend `DiscoveryAgentLoop` to emit a second mask:

```text
action_loss_mask          assistant/generated action tokens
env_observation_mask      factual environment observation tokens
```

2. Keep boilerplate and repair warnings out of `env_observation_mask` where possible. ECHO excludes low-entropy harness warnings because they are easy to memorize and stop being useful.

3. Patch the veRL actor loss to read `env_observation_mask` and compute:

```text
env_ce = -sum(logp[target_token] * env_observation_mask) / observation_token_count
loss = grpo_loss + env_loss_coef * env_ce
```

4. Log:

```text
env_ce
env_observation_tokens
action_tokens
reward
recovery
duplicate_count
valid_unique_count
query_information_gain
```

5. Add an algorithm recipe:

```bash
DISCOVERY_ALGO=echo_grpo
```

The recipe already exists as a scaffold, but it intentionally requires a custom trainer patch before launch.

## Extra Work Beyond Standard ECHO

ECHO optimizes world prediction, not diversity directly. To make the claim that world modeling helps diversity, we need additional measurement and possibly one extra objective or analysis.

Minimum viable experiment:

- Train vanilla GRPO and ECHO-GRPO on identical interactive envs.
- Evaluate both with `protocol: single`, `rollouts_per_spec=K`.
- Report pass@K-like recovery across independent rollouts.
- Report Validity, Uniqueness, Recovery, duplicates, and invalids.
- Report environment-token CE on held-out trajectories from a stronger or different policy.

Better experiment:

- Add `query_information_gain`: reduction in diagnostic version-space size after each query, logged but not shown to the agent.
- Compare whether ECHO increases information gain before commit.
- Compare entropy of recovered target hypotheses across rollout groups.

Possible extension:

- Counterfactual observation prediction: sample candidate actions from the current state, ask the model or a lightweight head to predict their observations, and reward/query-select actions with high expected information gain. This is beyond standard ECHO and should be a later method, not part of the first comparison.

Failure points:

- The model may learn to predict easy, repetitive observation templates without improving exploration.
- Observation CE can compete with policy learning if weighted too high.
- If failed trajectories dominate, env prediction may model unproductive loops.
- Diversity may not improve unless the policy uses the learned world model to ask better queries.

Remedies:

- Mask only factual observation payloads, not generic wrapper text.
- Start with `lambda=0.01-0.05`.
- Track held-out env-token CE separately from reward.
- Filter malformed-action trajectories for env-only ablations.
- Use diversity-specific metrics as primary evidence, not solve rate alone.
