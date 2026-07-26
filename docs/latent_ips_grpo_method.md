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

For valid response `y_i` with assigned latent `z_i` and counterfactual latent
`z_i^-`, the answer-only specificity score is

```text
m_i = mean_t [
  log(2) + log p(y_it | x, z_i, y_i,<t)
         - logaddexp(
             log p(y_it | x, z_i, y_i,<t),
             log p(y_it | x, z_i^-, y_i,<t)
           )
]
```

Only tokens in the visible final answer are scored. The specificity term is
clipped to `[-1, 1]`, multiplied by `alpha = 0.1`, and gated by validity. An
invalid response cannot earn latent-specificity reward.

For the combined method, valid outcome `h_i` receives empirical
inverse-frequency weight

```text
w_i = 1 / max(n(h_i) / G, epsilon), epsilon = 0.2
```

where `n(h_i)` is its canonical consequence-mode frequency in the eight-sample
group. Syntax shaping and length penalties are not inverse weighted. The final
per-sequence score before ordinary within-group GRPO normalization is

```text
s_i = r_i - r_valid_i + w_i r_valid_i + alpha clip(m_i)
```

For the latent-only ablation, `w_i = 1`. Everything else is identical.

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
sbatch scripts/cluster/sbatch_causal_micro_lab_latent_grpo.slurm combined
```

Latent-only ablation:

```bash
sbatch scripts/cluster/sbatch_causal_micro_lab_latent_grpo.slurm latent-only
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
