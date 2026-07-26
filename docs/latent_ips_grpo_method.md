# Consequence-Latent IPS-GRPO

Status: superseded by
[`set_utility_grpo_method.md`](set_utility_grpo_method.md). Retained as a record
of the literature-combination framing that was rejected.

Working name: **Consequence-Latent IPS-GRPO** (`CLIPS-GRPO`).

## 1. One-sentence method

Condition each rollout in a GRPO group on a distinct latent strategy code,
train the codes to predictably control observable consequence modes, and apply
inverse probability scaling to validity using those same consequence modes as
outcomes.

## 2. Motivation

Plural scientific hypothesis generation has two coupled failure modes:

1. **Strategy collapse:** repeated samples follow the same generation strategy,
   so alternative hypotheses are rarely proposed.
2. **Outcome amplification:** once one valid hypothesis is sampled more often,
   expected-return optimization reinforces it more strongly and further
   increases its sampling probability.

The two components address different failures:

- latent skill conditioning and mutual-information skill discovery, inspired
  by UpSkill, gives the policy a controllable collection of conditional
  generation strategies;
- inverse probability scaling, inspired by IPS-GRPO, prevents frequently
  sampled semantic outcomes from monopolizing the validity signal.

The coupling is essential. Consequence-level mutual information makes the
latent codes select reproducibly different scientific behaviors. IPS ensures
that a newly reached valid behavior receives stronger credit than another
duplicate of an already common behavior.

## 3. Relationship to prior work

### 3.1 UpSkill

[UpSkill](https://arxiv.org/abs/2602.22296) introduces a discrete latent
strategy code \(z\), conditions the policy on that code, and maximizes
token-level mutual information between \(z\) and the generated trajectory.
Its purpose is to learn distinct, reproducible generation strategies.

We retain:

- uniformly sampled discrete strategy codes;
- one distinct code per inference attempt;
- the mutual-information skill-discovery objective;
- the original validity/correctness reward and KL control.

UpSkill reports an important failure mode: MI can be maximized through
stylistic, multilingual, chaotic, or gibberish differences rather than
different correct outcomes. CLIPS-GRPO therefore replaces token-trajectory MI
with consequence-level MI:

\[
I(Z;C\mid X),
\]

where \(C\) is computed by executing the candidate hypothesis. This is closely
related to UpSkill's semantic-MI ablation, but the micro-lab supplies an exact
task-grounded consequence representation rather than a generic text
embedding.

### 3.2 IPS-GRPO

[IPS-GRPO](https://arxiv.org/abs/2601.21669) defines a terminal outcome
\(o=\phi(\tau)\), estimates its probability by frequency within the rollout
group, and scales its reward by the inverse estimated probability:

\[
\widetilde r(o_i)
=
\frac{r(o_i)}
{\max(\widehat p(o_i),\epsilon)}.
\]

We retain this estimator and stop gradients through the empirical outcome
probability.

In CLIPS-GRPO, the outcome is not the output string. It is the candidate's
observable consequence signature:

\[
o_i = c_x(y_i).
\]

Programs that are syntactically different but make identical predictions are
therefore treated as one outcome.

### 3.3 New combination

UpSkill encourages token-trajectory mutual information \(I(Z;Y\mid X)\).
CLIPS-GRPO instead encourages scientific-behavior mutual information
\(I(Z;C\mid X)\).

IPS balances:

\[
p(C\mid X,\text{valid}),
\]

where \(C\) is the scientific consequence mode.

CLIPS-GRPO connects the two:

> Learn distinguishable latent-conditioned generation strategies, but allocate
> correctness credit according to the semantic outcomes those strategies
> actually recover.

The contribution is not either component in isolation. It is the hypothesis
that task-grounded latent specialization and semantic outcome balancing solve
complementary stages of plural-answer RLVR.

## 4. Training setup

### 4.1 Group construction

For the primary run:

```text
group size G = 8
latent count L = 8
one rollout per latent per evidence state
16 evidence states per update
128 total completions per update
```

This exactly matches the group size and 128-completion update budget of the
current validity-GRPO run:

```text
16 evidence states x 8 rollouts = 128 completions
```

For each evidence state \(x\), assign:

```text
rollout 1 -> z1
rollout 2 -> z2
...
rollout 8 -> z8
```

Use a lightweight textual prefix following UpSkill:

```text
Strategy 1 |
```

The strategy identifier contains no task information and requests no particular
hypothesis or consequence.

### 4.2 Candidate outcome

For each completion \(y_i\):

1. parse the three-rule hypothesis;
2. check consistency with visible evidence;
3. if valid, execute the candidate on the public probe experiments;
4. hash the resulting consequence signature to obtain \(c_i\).

No private valid-mode set or hidden-world field is read by the reward path.

### 4.3 IPS validity reward

For valid outcomes in the group:

\[
\widehat p(c)
=
\frac{1}{G}
\sum_{j=1}^{G}\mathbf 1[c_j=c].
\]

Then:

\[
r^{\mathrm{IPS}}_i
=
\frac{\mathbf 1[\mathrm{valid}_i]}
{\max(\widehat p(c_i),\epsilon)}.
\]

Equivalently, before GRPO normalization, a valid behavior occurring \(n(c_i)\)
times receives a value proportional to \(G/n(c_i)\). Scaling every reward by
the constant \(G\) does not affect standardized group advantages, so the
implementation may use \(1/n(c_i)\).

Counts are computed across all eight latents. Counting separately within each
latent would fail to penalize different latents that produce the same semantic
outcome.

### 4.4 Consequence-level mutual-information reward

Use the standard variational lower bound employed in mutual-information skill
discovery. Train a lightweight classifier \(q_\phi\) to predict the latent code
from the candidate consequence signature:

\[
q_\phi(z\mid c).
\]

With a uniform latent prior, the per-completion MI reward is:

\[
r^{\mathrm{MI}}_i
=
\mathbf 1[\mathrm{valid}_i]
\left(
\log q_\phi(z_i\mid x,c_i)+\log L
\right).
\]

The classifier receives no private mode data. Its only substantive input is the
consequence signature computed from the generated candidate. It is trained by
cross-entropy on generated `(consequence, latent)` pairs. The current evidence
state is deliberately omitted: because each state is seen only once, a
classifier conditioned on `state_id` could memorize rollout assignments rather
than learning a reusable partition of scientific behaviors.

This avoids UpSkill's expensive exact token-mixture calculation. With
\(G=L=8\), exact token MI would require scoring each generated sequence under
all eight latent codes. The consequence classifier adds negligible cost relative
to LLM generation and preserves the matched 128-completion budget.

Train the classifier on every syntax-valid executable candidate so it receives
enough early data, but gate the policy's MI reward by evidence validity. Compute
the policy reward using the classifier parameters from before the current
classifier update, then update the classifier. This prevents immediate
same-batch memorization and prevents the policy from gaining MI reward by
specializing in different invalid hypotheses.

As in UpSkill, clip or cap the MI reward to avoid an unstable strategy
receiving arbitrarily large credit.

The total terminal reward is:

\[
r_i
=
\alpha_{\mathrm{valid}}r^{\mathrm{IPS}}_i
+
\alpha_{\mathrm{MI}}
\operatorname{clip}(r^{\mathrm{MI}}_i,-r_{\mathrm{MI,max}},r_{\mathrm{MI,max}})
+
r^{\mathrm{syntax}}_i
+
r^{\mathrm{length}}_i.
\]

Initial settings:

```text
alpha_valid = 1.0
alpha_MI = one conservative value plus zero ablation
syntax-valid/evidence-invalid = 0.2
length penalty starts at 3072
maximum response length = 6000
cap-hit completion = -0.2 and masked from policy loss
```

The classifier should be a small MLP, not another language model. Its parameter
count, training time, and inputs must be logged.

## 5. Intended learning dynamics

Assume several latent codes initially produce a common valid outcome while one
latent happens to produce a rarer valid outcome.

1. Consequence MI reinforces differences between the latent-conditioned
   scientific behaviors.
2. IPS splits correctness credit among the duplicate common outcomes.
3. The rare valid outcome receives the largest validity credit.
4. The latent that generated it becomes more likely to reproduce that region.
5. At inference, selecting distinct latent codes yields complementary
   hypotheses rather than repeated i.i.d. attempts.

IPS alone has no explicit handle with which to reproduce a discovered mode.
Token-level latent MI alone has no requirement that distinguishable
trajectories correspond to distinct valid scientific outcomes. CLIPS-GRPO
supplies a task-grounded latent handle and outcome balancing.

## 6. Inference

For a generation budget \(K\), choose \(K\) distinct latent codes and generate
one completion from each.

With \(L=8\):

```text
K=4:  use a deterministic rotating subset of 4 latents
K=8:  use all 8 latents once
K=16: use all 8 latents twice
```

Rotate subsets across evidence states using a seed derived from `state_id`.
This prevents `K=4` and `K=8` from being determined by a permanently strong or
weak subset of latent codes.

For other \(K\), cycle through all latents evenly and draw additional stochastic
samples per latent.

## 7. Main thesis story

### Problem

Validity-only GRPO treats every valid hypothesis as equally rewarding at the
sample level but not at the distribution level. Common hypotheses are sampled
and reinforced more often, while repeated attempts lack a persistent mechanism
for selecting different strategies.

### Existing partial solutions

- UpSkill provides controllable strategy diversity but optimizes token-level
  distinguishability, which need not correspond to scientific outcome
  diversity.
- IPS-GRPO balances observed outcomes but does not provide a structured
  mechanism for querying different regions of the policy.

### Proposed synthesis

CLIPS-GRPO represents plural scientific reasoning as a uniform mixture of
consequence-specialized latent policies and balances validity credit over those
same observable scientific consequences.

### Testable claim

At matched training and inference generation budgets, CLIPS-GRPO should recover
more distinct valid consequence modes than validity GRPO, UpSkill-style token
MI, consequence-MI alone, or IPS-GRPO alone, while retaining comparable
per-completion validity.

## 8. Required comparisons

Use the same Qwen3-4B checkpoint, data, update count, 128 completions per
update, response budget, and evaluation states.

| Arm | Consequence latent MI | Consequence IPS |
|---|---:|---:|
| Validity GRPO | No | No |
| Consequence-skill GRPO | Yes | No |
| Consequence IPS-GRPO | No | Yes |
| CLIPS-GRPO | Yes | Yes |

This \(2\times2\) experiment identifies:

- the main effect of latent strategy learning;
- the main effect of semantic outcome balancing;
- whether their interaction is complementary or redundant.

## 9. Primary diagnostics

Log the following by training step and by `M`:

- evidence-valid rate;
- exact consequence-mode coverage at `K=4,8,16`;
- effective consequence-mode count;
- duplicate consequence-mode rate;
- consequence-level MI reward;
- empirical \(I(Z;C\mid X,\mathrm{valid})\);
- latent predictability from consequence signatures;
- number of distinct valid modes per latent;
- pairwise overlap between latent mode distributions;
- validity rate per latent;
- fraction of latents producing no valid output;
- response length and truncation rate per latent.

The critical diagnostics are:

- high classifier accuracy with poor held-out \(I(Z;C)\) indicates classifier
  overfitting;
- high \(I(Z;C)\) with poor validity indicates specialization into invalid or
  irrelevant regions;
- high validity and held-out \(I(Z;C)\) indicates the intended mechanism.

## 10. Stop conditions

Reject or revise the method if:

- consequence MI increases but exact consequence coverage does not;
- latents specialize mainly by language, formatting, or response length;
- the combined method is not better than IPS alone at matched validity;
- improvement exists only at `K=16` and disappears at `K=4` and `K=8`;
- a small set of latents accounts for nearly all valid outputs;
- the MI coefficient requires a narrow, post-hoc hyperparameter choice.

## 11. Scope of the claim

The micro-lab provides an exact consequence map, allowing semantic outcomes to
be identified without private answer lookup. In broader scientific generation,
CLIPS-GRPO requires an observable outcome descriptor: executable predictions,
simulator behavior, experimentally testable consequences, or another
task-grounded representation.

The general contribution is therefore not unrestricted text diversity. It is
latent-conditioned, outcome-balanced RL for domains where generated scientific
objects have observable consequences.
