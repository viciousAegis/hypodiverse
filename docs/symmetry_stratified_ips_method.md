# Symmetry-Stratified IPS-GRPO

Status: current method candidate.

Working name: **Symmetry-Stratified IPS-GRPO** (`SS-IPS`).

## 1. Problem

Validity-only GRPO has two distinct failure modes in plural scientific
hypothesis generation.

First, independent rollouts repeatedly spend generation budget in the same
high-probability basin. A semantic mode that is not sampled receives no reward
signal. This is a **reachability** failure.

Second, after multiple valid modes are sampled, expected-return optimization
amplifies whichever modes already have more policy probability. This is a
**retention** failure.

Reward reweighting, including IPS, addresses retention only. A complete method
must also change the rollout proposal so that lower-probability paths are
actually visited.

SS-IPS uses two corresponding operations:

1. **symmetry-stratified rollouts** expose different parts of the policy's
   existing hypothesis support;
2. **canonical outcome IPS** prevents the newly exposed valid modes from being
   immediately suppressed during learning.

No desired hypothesis, private valid-mode list, or hidden-world information is
provided to the model.

## 2. Exact task symmetries

Let \(x\) be a visible evidence state and let \(\mathcal T_x\) be a set of
invertible transformations that preserve its scientific meaning.

For the Boolean causal micro-lab, the initial transformation family contains:

- all six permutations of the exogenous variable names
  \((X_1,X_2,X_3)\), applied consistently to the operator description,
  evidence, and returned program;
- deterministic permutations of the visible evidence rows;
- deterministic permutations of explicitly unordered operator descriptions.

The primary run should use only exogenous-variable and evidence-row
permutations. Operator-description order is an ablation.

Every transformed prompt has exactly the same solution set up to an invertible
renaming. A generated program is mapped back through the inverse transform
before verification and semantic comparison.

These are nuisance transformations, not answer constraints. They neither add
an observation nor specify a candidate's prediction on an unobserved
experiment.

## 3. Why symmetries can expose support

An ideal reasoner would be invariant to every \(\tau\in\mathcal T_x\).
Current language models are not. Let

\[
p_m^\tau
=
\Pr_{\pi_\theta}\left(
  \text{canonical mode}(y)=m
  \mid \tau(x)
\right)
\]

be the probability of canonical semantic mode \(m\) under presentation
\(\tau(x)\).

If all \(p_m^\tau\) are identical, symmetry stratification provides no
benefit. If they differ, the presentations expose complementary portions of
the policy's support.

This converts nuisance sensitivity, normally treated only as a robustness
problem, into a controlled exploration proposal.

## 4. Stratified rollout proposal

For a group of \(G\) completions:

1. choose \(G\) transformations before generation;
2. allocate exactly one rollout to each selected transformation;
3. generate independently within each transformed prompt;
4. inverse-transform each parsed candidate into canonical variable names;
5. verify and group candidates in canonical semantic space.

For the primary \(G=8\) run:

- include each of the six exogenous-variable permutations once;
- use two additional variable permutations with different evidence-row
  permutations;
- derive all transformation choices deterministically from
  `(state_id, rollout_index, seed)`;
- include the identity presentation in every group.

Transforms are allocated without replacement where possible. They are not
selected after inspecting completions or verifier results.

## 5. Coverage result

For a fixed mode \(m\), let the selected transformations have probabilities
\(p_{m,1},\ldots,p_{m,G}\). One rollout is sampled from each transformation.
The probability of observing \(m\) at least once is:

\[
P_{\mathrm{strat}}(m)
=
1-\prod_{j=1}^{G}(1-p_{m,j}).
\]

Now consider randomly choosing a transformation independently for every
rollout from the same \(G\)-element collection. Its marginal mode probability
is:

\[
\bar p_m=\frac{1}{G}\sum_{j=1}^{G}p_{m,j},
\]

and its hit probability is:

\[
P_{\mathrm{mix}}(m)=1-(1-\bar p_m)^G.
\]

By the arithmetic-geometric mean inequality:

\[
\prod_{j=1}^{G}(1-p_{m,j})
\le
\left(
  \frac{1}{G}\sum_{j=1}^{G}(1-p_{m,j})
\right)^G
=(1-\bar p_m)^G.
\]

Therefore:

\[
P_{\mathrm{strat}}(m)\ge P_{\mathrm{mix}}(m).
\]

Summing over modes gives:

\[
\mathbb E[C_{\mathrm{strat}}@G]
\ge
\mathbb E[C_{\mathrm{random\ mix}}@G].
\]

The inequality is strict whenever the selected presentations assign different
probabilities to at least one reachable mode.

This result does **not** say stratification always beats the canonical prompt.
That requires the empirical transformation mixture to expose useful support.
It says that, once a transformation family is chosen, balanced allocation uses
that proposal family at least as efficiently for coverage as random transform
allocation.

## 6. Canonical outcome retention

After inverse transformation, every evidence-valid candidate is executed on
the public experiment set to obtain a canonical consequence signature
\(c_i\). Let:

\[
n(c_i)
=
\sum_{j=1}^{G}\mathbf 1\{c_j=c_i,\ v_j=1\}.
\]

Apply the existing empirical IPS validity credit:

\[
r_i^{\mathrm{IPS}}
=
\frac{v_i}
{\max(n(c_i)/G,\epsilon)}.
\]

The count is computed across the entire original-state group, not separately
inside each transformed presentation. Two differently written or differently
renamed programs with the same canonical consequences are duplicates.

Keep the existing verifier ladder and length handling:

```text
empty or unparseable:                    0.0
syntax-valid but evidence-invalid:       0.2
evidence-valid:                          1.0 before IPS scaling
length penalty begins:                   3072 tokens
maximum response length:                 6000 tokens
cap-hit policy loss:                     masked
```

The diversity mechanism may not read the private valid-mode set or target
mode count \(M\).

## 7. Learning interpretation

Suppose the canonical prompt strongly favors mode \(a\), while two renamed
presentations give non-negligible probability to modes \(b\) and \(c\).

1. Stratification makes \(b\) and \(c\) more likely to occur in the same
   finite rollout group as \(a\).
2. Canonicalization recognizes that they are scientifically distinct rather
   than merely different surface forms.
3. IPS gives the singleton modes more credit than repeated copies of \(a\).
4. Shared model parameters increase the probability of the successful
   transformed trajectories.
5. Reusing the same stratified proposal at inference preserves access to the
   complementary support.

SS-IPS therefore does not claim that reward creates an unseen hypothesis.
The representation orbit supplies the proposal; IPS supplies retention.

## 8. Compute-matched training

Match the existing validity-GRPO and IPS-GRPO runs:

```text
model: Qwen/Qwen3-4B
original states per update: 16
rollouts per original state: 8
total completions per update: 128
updates: 100
temperature: 1.0
top_p: 1.0
maximum response length: 6000
GPUs: 2
dataset: data/causal_micro_lab/trainable
```

Transformation and inverse canonicalization are CPU operations and require no
additional model calls. Prompt token counts may change slightly because
variable names have equal token lengths but evidence ordering changes cache
behavior; actual generated and prompt tokens must be logged for compute
matching.

## 9. Inference

For inference budget \(K\):

- select \(K\) transformations using the same deterministic balanced schedule;
- generate one completion per transformation;
- inverse-transform candidates;
- verify and calculate set metrics in canonical mode space.

The method supports any \(K\). For \(K>|\mathcal T_x|\), cycle through the
transformation family with independently sampled decoding noise and balanced
replication counts.

Report canonical-prompt inference as an additional transfer diagnostic. The
primary SS-IPS result uses symmetry-stratified inference because that is the
policy proposal optimized during training.

## 10. Required experiment design

Use a \(2\times2\) design:

| Arm | Rollout proposal | Retention objective |
|---|---|---|
| Validity GRPO | canonical i.i.d. | validity |
| IPS-GRPO | canonical i.i.d. | IPS |
| SS-GRPO | symmetry-stratified | validity |
| SS-IPS | symmetry-stratified | IPS |

This separates:

- the support gain from representation stratification;
- the retention gain from IPS;
- any interaction between them.

All arms use the same original states, total completions, maximum generation
tokens, optimizer updates, and evaluation states.

## 11. Metrics specific to the mechanism

In addition to existing validity and set metrics, log:

- exact and budget-normalized coverage by \(K\), \(M\), and separation bucket;
- canonical mode distribution for each transformation;
- pairwise Jensen-Shannon divergence between transformation-conditioned mode
  distributions;
- modes found by canonical only, transformed only, and both;
- fraction of groups with at least two valid canonical modes;
- transform-conditioned validity and truncation rate;
- IPS singleton and duplicate rates after canonicalization;
- coverage under stratified transforms versus randomly assigned transforms;
- canonical-prompt evaluation of the trained model.

## 12. Falsification gates

### Gate 1: zero-training support

Before RL, compare eight canonical rollouts with one rollout from each of eight
balanced transforms on the same states and seeds.

Reject the proposal mechanism if transformed sampling does not improve paired
canonical mode coverage at comparable validity.

### Gate 2: semantic rather than syntactic gain

All gains must remain after inverse transformation and exact consequence
canonicalization.

### Gate 3: no accidentally privileged transform

Reject the transformation family if almost all gains come from one fixed
presentation that simply has higher validity. That would be prompt selection,
not stratified support.

### Gate 4: retention interaction

SS-IPS must beat both SS-GRPO and canonical IPS on held-out coverage under the
same inference proposal. Otherwise the combined reachability-retention story
is unsupported.

### Gate 5: canonical transfer

Measure whether SS-IPS improves canonical-prompt coverage. Failure here does
not invalidate stratified inference, but it limits the claim to a joint
training-and-inference proposal rather than a generally diversified canonical
policy.

## 13. Failure modes

- **Model invariance:** all transformations induce the same mode distribution,
  so stratification has no effect.
- **Validity heterogeneity:** some representations substantially reduce
  correctness and waste rollout budget.
- **Surface-only changes:** transformed prompts alter text but not canonical
  scientific modes.
- **Mixture-only improvement:** diversity rises only when transformed prompts
  are used at inference.
- **IPS instability:** rare transformed modes receive noisy inverse-frequency
  weights.
- **Limited generality:** an open-ended task without known
  semantics-preserving transformations cannot use this method directly.

These are empirical and conceptual limits, not details to hide with
hyperparameter tuning.

## 14. Relationship to prior work

The method is motivated from the reachability-retention decomposition above.
Its components must still be situated accurately:

- [IPS-GRPO](https://arxiv.org/abs/2601.21669) supplies the outcome-frequency
  retention correction.
- [MEND](https://arxiv.org/abs/2502.17800) and
  [order-centric augmentation](https://arxiv.org/abs/2502.19907) use
  semantics-preserving transformations to improve reasoning consistency and
  robustness, not plural valid-mode coverage.
- [Language of Thought Shapes Output Diversity](https://arxiv.org/abs/2601.11227)
  demonstrates that equivalent reasoning representations can expose
  complementary output support.
- [SimpleStrat](https://arxiv.org/abs/2410.09038) asks a model to construct
  semantic answer strata and then conditions generation on selected strata.
  SS-IPS instead uses predefined exact nuisance symmetries and adds no desired
  answer property.
- UpSkill, correlated Monte Carlo, arithmetic sampling, and prefix branching
  are support-creation alternatives and required related-work baselines, not
  descriptions of SS-IPS.

The safe contribution claim is narrow:

> A compute-matched RLVR method that stratifies rollouts over exact scientific
> representation symmetries, canonicalizes executable outcomes, and applies
> outcome balancing to retain the complementary valid support exposed by those
> representations.

Do not claim that data augmentation, prompt permutations, stratified sampling,
or IPS is independently new.

## 15. Decision

SS-IPS passes the four design checks provisionally:

1. **Objective:** increase finite-budget coverage of evidence-valid semantic
   hypotheses.
2. **Mechanism:** representation stratification changes the current rollout
   proposal; IPS retains distinct modes that the proposal reaches.
3. **Novelty boundary:** the joint scientific RLVR construction and its
   reachability-retention analysis are the contribution, not either component
   independently.
4. **Failure analysis:** the method has explicit zero-training and short-run
   kill criteria and makes no claim when useful nuisance symmetries are absent.

Unlike a same-group reward modification, SS-IPS can change which modes appear
in the very first rollout group. That is the central reason to test it.
