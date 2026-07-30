# Latent IPS-GRPO

## Purpose

Validity-only GRPO can make valid hypotheses more likely without making its
samples cover more semantic modes. Latent IPS-GRPO adds two independent
pressures:

1. a latent label gives each rollout in a group a stable coordinate on which
   generation can specialize;
2. empirical inverse-frequency weighting gives more credit to valid outcomes
   that are rare within the current group.

The latent-only ablation keeps the complete latent mechanism and disables only
the inverse-frequency weight.

## Rollout construction

Each evidence state has eight rollouts. Rollout `i` receives one deterministic
label:

```text
Strategy z | <original prompt>
```

where `z = i mod 8 + 1`. The label has no predefined semantic meaning and
contains no answer information. It is a coordination label that the policy may
learn to associate with different valid solution regions.

For each generated response under latent `z`, the trainer also evaluates the
same response under one rotating counterfactual latent:

```text
z_negative = z mod 8 + 1
```

This is a teacher-forced actor pass. It does not generate another response.

## Objective

Let `r_i` retain the existing reward ladder:

- `0.0`: unparseable or otherwise unrewarded;
- `0.2`: syntax-valid but evidence-inconsistent;
- `1.0`: evidence-consistent valid hypothesis;
- the existing response-length penalty is added separately.

The original `v1` implementation scored only valid responses and only final
answer tokens. That made the signal ineffective: the full reasoning prefix had
already determined the answer before the latent was compared, and invalid
early-training samples received no latent signal.

The minimally repaired `v2` objective scores every non-truncated generated
trajectory `y_i` under its assigned latent `z_i` and one counterfactual latent
`z_i^-`:

```text
m_i = sum_t [
  log(2) + log p(y_it | x, z_i, y_i,<t)
         - logaddexp(
             log p(y_it | x, z_i, y_i,<t),
             log p(y_it | x, z_i^-, y_i,<t)
           )
]
```

All generated response tokens selected by `response_mask` are scored. The
specificity term is clipped to `[-1, 1]` and multiplied by `alpha = 0.5`.
Consequently its contribution is always in `[-0.5, 0.5]`.

The v2 run initializes from the trained v1 checkpoint, which already has high
Pass@16 and non-sparse validity. Latent specificity is therefore applied only
to valid responses. This makes the second stage optimize diversity within the
valid set rather than allowing an invalid but latent-specific trajectory to
outscore a valid hypothesis. Truncated responses remain masked.

For the combined method, valid outcome `h_i` first receives empirical
inverse-frequency weight

```text
w_i = 1 / max(n(h_i) / G, epsilon), epsilon = 0.2
```

where `n(h_i)` is its canonical consequence-mode frequency in the eight-sample
group. Rather than replacing the `1.0` validity reward with a value as large as
`5.0`, v2 converts this weight into a bounded rarity bonus:

```text
b_i = 0.5 * (w_i - 1) / (1 / epsilon - 1)
```

Thus `b_i` lies in `[0, 0.5]`. Syntax shaping and length penalties are not
weighted. The final per-sequence score before ordinary within-group GRPO
normalization is

```text
s_i = r_i + 1[h_i is valid] (b_i + 0.5 clip(m_i))
```

For the latent-only ablation, `b_i = 0`. Everything else is identical.

## Compute matching

The combined and latent-only runs are exactly matched by construction: same
prompts, generations, counterfactual pass, optimizer settings, and 84 updates.

Matching against the 100-update validity-only baseline is approximate in model
forward/backward FLOPs. The baseline uses about five teacher-forced model-pass
equivalents per rollout; latent training adds one counterfactual forward pass.
Therefore:

```text
100 * 5 / 6 = 83.3 -> 84 updates
```

Generation count is lower than the 100-update baseline (`84 * 128` versus
`100 * 128`), while test-time evaluation remains exactly matched at the same
`K` and response cap. Both update-matched and generated-token-matched results
should be reported if resources permit.

## Runs

Combined method:

```bash
sbatch scripts/cluster/sbatch_causal_micro_lab_latent_grpo_v2.slurm combined
```

By default, this finds the newest complete checkpoint under
`causal_micro_lab_cluster_latent_ips_grpo_v1_k8_r1`, merges its actor weights,
and uses that model to initialize a fresh v2 run. It does not restore the v1
optimizer, scheduler, or training step. Override the source when needed:

```bash
sbatch scripts/cluster/sbatch_causal_micro_lab_latent_grpo_v2.slurm \
  combined --init-run OTHER_RUN --init-step 40
```

Use `--init-model /absolute/merged/hf/path` for an already merged model, or
`--base-model` to initialize directly from Qwen3-4B.

Latent-only ablation:

```bash
sbatch scripts/cluster/sbatch_causal_micro_lab_latent_grpo_v2.slurm latent-only
```

Both use the same `data/causal_micro_lab/trainable` files as validity GRPO,
Qwen3-4B, two GPUs, 16 states per update, eight rollouts per state, a 6000-token
response cap, and length shaping beginning at 3072 tokens.

Dataset rows keep their generic `causal_micro_lab_agent_loop` name. The
latent run's agent configuration maps that name to `LatentGRPOAgentLoop` in
each worker's local registry. Validity, IPS, and latent jobs can therefore read
the same immutable files concurrently without rewriting routing metadata.

After merging a checkpoint to Hugging Face format, run the matched latent
evaluation with:

```bash
sbatch scripts/cluster/sbatch_causal_micro_lab_latent_eval.slurm \
  /absolute/path/to/merged_hf_model \
  causal_micro_lab_final_k16_latent_ips
```

It makes one K=16 generation bank and derives K=4,8,12,16 prefix metrics. The
unconditioned final-eval config uses the same 6000-token cap, temperature, and
top-p settings.

## Diagnostics

W&B receives:

- `latent_ips/mi_raw_mean` and `latent_ips/mi_raw_std`;
- `latent_ips/mi_reward_bound`, `mi_reward_abs_mean`, and
  `mi_reward_max_abs`;
- `latent_ips/mi_to_valid_reward_bound_ratio` (configured as `0.5`);
- `latent_ips/ips_bonus_bound`, `ips_bonus_mean`, and
  `ips_bonus_max_observed`;
- `latent_ips/mi_clip_rate`;
- `latent_ips/trajectory_logp_margin_mean`;
- `latent_ips/answer_mi_raw_mean` as a diagnostic for the failed v1 signal;
- `latent_ips/answer_logp_margin_mean`;
- `latent_ips/mi_reward_mean`;
- `latent_ips/cross_latent_outcome_collision_rate`;
- `latent_ips/unique_valid_outcomes_per_group`;
- `latent_ips/groups_with_multiple_valid_outcomes_rate`;
- `latent_ips/groups_all_latents_present_rate`;
- `latent_ips/weight_mean`, `weight_max`, and `clipped_valid_rate`;
- per-latent row counts and validity rates;
- per-latent specificity, IPS weight, and unique valid outcomes;
- validity and unique valid outcomes per group sliced by `M`;
- visible-answer token counts;
- counterfactual scoring time, tokens, and throughput.

The central success criterion is not a large specificity score by itself. It is
higher canonical mode coverage at matched validity and evaluation budget.

## Known veRL integration invariant

veRL's non-TQ validation postprocessor derives `reward_extra_info` keys from
the first rollout and indexes every later rollout with that schema. Every
rollout class in one batch must therefore return exactly the same scalar keys.
A mixed base/IPS batch fails with errors such as:

```text
KeyError: 'reward_syntax_valid'
```

This previously affected both IPS-GRPO and latent IPS-GRPO. Latent runs map
the dataset's generic agent name to one implementation for the whole worker
and normalize reward extras in that rollout class before veRL batches them.

## Design history

An earlier CLIPS-GRPO draft proposed learning a separate classifier
`q(z | consequence)` and using consequence-level mutual information. That
version was not implemented: because each evidence state is encountered only
once, the classifier introduces an extra learner whose generalization and
same-batch leakage would be difficult to separate from the policy effect.

The implemented objective instead follows the latent policy itself. It asks
whether the final answer generated under `z` is more likely under `z` than
under a rotating alternative label. This removes the classifier, keeps the
mechanism end to end, and requires exactly one additional actor forward pass
per completion. IPS still operates on canonical consequence identities, so
latent specialization and outcome balancing remain separately ablatable.

The two direct inspirations should be treated as related work rather than as
the motivation for the construction:

- [UpSkill](https://arxiv.org/abs/2602.22296) motivates latent-conditioned
  specialization and likelihood-based latent information;
- [IPS-GRPO](https://arxiv.org/abs/2601.21669) motivates empirical
  inverse-frequency credit over observed outcomes.
