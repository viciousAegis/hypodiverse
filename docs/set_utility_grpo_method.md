# Set-Utility GRPO

Status: rejected after algebraic audit. For cardinality coverage, its Shapley
credit reduces to inverse-count scaling, overlapping IPS/GAPO, and fair
Shapley allocation is not by itself the exact policy-gradient credit for the
stated set objective. See
[`budgeted_mode_coverage_method_audit.md`](budgeted_mode_coverage_method_audit.md).

Working name: **Set-Utility GRPO** (`SU-GRPO`).

## 1. Starting point

The model is not ultimately being used to produce one hypothesis. It is given a
budget of \(K\) attempts and is expected to return a useful set of hypotheses.
The training objective should therefore score the generated set directly.

For evidence state \(x\), let:

- \(y_i\) be generated hypothesis \(i\);
- \(v_i\in\{0,1\}\) indicate whether \(y_i\) is evidence-valid;
- \(c_i=c_x(y_i)\) be its observable consequence signature;
- \(S=(y_1,\ldots,y_G)\) be the rollout group.

Define:

\[
N_{\mathrm{valid}}(S)=\sum_{i=1}^{G}v_i
\]

and:

\[
C(S)=\left|\{c_i:v_i=1\}\right|.
\]

The first quantity measures valid yield. The second measures unique valid
scientific behaviors.

## 2. Set utility

Use one explicit validity-coverage tradeoff:

\[
U_\lambda(S)
=
(1-\lambda)N_{\mathrm{valid}}(S)
+
\lambda C(S),
\qquad \lambda\in[0,1].
\]

This objective has interpretable endpoints:

- \(\lambda=0\): validity-only training;
- \(\lambda=1\): pure unique-mode coverage;
- \(0<\lambda<1\): increase coverage without making valid yield irrelevant.

The primary setting is \(\lambda=0.5\). It should be fixed before the main run,
with \(\lambda=0\) providing the matched validity control.

## 3. Policy parameterization with output slots

Independent, unlabelled rollouts are exchangeable. Even if a rare hypothesis is
rewarded once, the user has no way to request that behavior again.

The set objective itself does not require latent variables. SU-GRPO
parameterizes the group as \(G\) indexed output slots because the desired
output is not only diverse during training; its components should remain
separately requestable at inference:

\[
P_\theta(S\mid x)
=
\prod_{i=1}^{G}
\pi_\theta(y_i\mid x,z_i),
\]

where \(z_i\) is a learned or textual slot identifier.

The slot carries no scientific information. It gives each factor of the set
policy a persistent identity, allowing different slots to specialize after
receiving different coverage credit. This is an implementation choice derived
from the requirement for reproducible set members, not an additional diversity
objective.

For the current experiment:

```text
G = 8 slots
z1 ... z8
one rollout per slot per state
16 states per update
128 completions per update
```

This exactly matches the current validity-GRPO group size and generation
budget.

## 4. Per-slot credit from the set objective

The set utility is non-additive because duplicate valid hypotheses contribute
less than distinct valid hypotheses. GRPO nevertheless requires a scalar credit
for each generated trajectory.

Allocate \(U_\lambda\) using the Shapley value of each slot.

### 4.1 Valid-yield component

\(N_{\mathrm{valid}}\) is additive, so valid slot \(i\) receives:

\[
\phi_i^{\mathrm{valid}}=v_i.
\]

### 4.2 Coverage component

Suppose consequence mode \(c_i\) occurs \(n(c_i)\) times among valid outputs.
Exactly one copy is the first representative of that mode in any random slot
ordering. By symmetry, each of the \(n(c_i)\) copies has probability
\(1/n(c_i)\) of being that representative. Its Shapley coverage credit is:

\[
\phi_i^{\mathrm{coverage}}
=
\frac{v_i}{n(c_i)}.
\]

The credits sum exactly to observed coverage:

\[
\sum_i\phi_i^{\mathrm{coverage}}=C(S).
\]

### 4.3 Combined reward

By Shapley linearity, the per-slot reward is:

\[
r_i^{\mathrm{set}}
=
v_i
\left[
(1-\lambda)
+
\frac{\lambda}{n(c_i)}
\right].
\]

Therefore:

- a unique valid mode receives reward \(1\);
- duplicate valid modes split the coverage portion;
- an invalid output receives no set reward.

With \(\lambda=0.5\) and \(G=8\), a valid duplicate always receives at least:

\[
0.5+\frac{0.5}{8}=0.5625.
\]

It therefore remains better than the existing \(0.2\) syntax-shaping reward.

The complete reward ladder is:

\[
r_i=
\begin{cases}
(1-\lambda)+\lambda/n(c_i), & \text{evidence-valid},\\
0.2, & \text{syntax-valid but evidence-invalid},\\
0, & \text{otherwise},
\end{cases}
+
r_i^{\mathrm{length}}.
\]

No distance kernel, determinant, archive, density model, auxiliary classifier,
or private valid-mode lookup is required.

## 5. Learning mechanism

Consider eight slots where six produce valid mode \(A\), one produces valid
mode \(B\), and one is invalid.

At \(\lambda=0.5\):

```text
each A slot: 0.5 + 0.5/6 = 0.5833
B slot:      0.5 + 0.5/1 = 1.0000
invalid:     0.0000 or syntax shaping
```

The update:

1. preserves positive credit for all valid hypotheses;
2. gives the strongest credit to the slot that found the distinct mode;
3. associates that mode with the slot identifier that produced it;
4. reduces reinforcement of redundant slots;
5. lets later calls request different slots explicitly.

The coverage reward itself creates specialization. A separate
mutual-information objective is not required.

## 6. Outcome definition

The consequence signature is computed from the generated candidate:

1. parse the three-rule hypothesis;
2. verify it against visible evidence;
3. execute it on the public probe experiments;
4. hash the resulting prediction vector.

Two syntactically different programs with identical predictions receive the
same consequence key and therefore count as duplicates.

The reward path may not read:

- `private.valid_mode_ids`;
- `private.hidden_mode_id`;
- the private mode table;
- target mode count \(M\);
- any feasibility query for a desired mode.

## 7. Training and inference

### Training

```text
model: Qwen3-4B
group size: 8
slot count: 8
states per update: 16
completions per update: 128
response limit: 6000
length shaping begins: 3072
primary lambda: 0.5
control lambda: 0.0
```

Each prompt is prefixed with a neutral slot identifier:

```text
Attempt 1 |
...
Attempt 8 |
```

The identifier must not describe a scientific strategy or requested outcome.

### Inference

- `K=4`: use a deterministic rotating subset of four slots;
- `K=8`: query every slot once;
- `K=16`: query every slot twice.

Slot subsets rotate by a seed derived from `state_id` so smaller-\(K\)
evaluation is not tied to permanently strong or weak slots.

## 8. Central thesis claim

Validity-only GRPO optimizes individual attempts even though scientific
hypothesis generation is consumed and evaluated as a set. SU-GRPO directly
optimizes a set utility that values both valid yield and semantic coverage, and
uses indexed output slots plus principled Shapley credit assignment to make
that non-additive objective trainable with GRPO.

The primary claim is:

> At equal training and inference generation budgets, optimizing
> validity-coverage set utility produces greater valid consequence coverage
> than optimizing independent validity, without sacrificing most of the valid
> yield.

## 9. Essential experiments

### 9.1 Main comparison

| Method | Slots | \(\lambda\) |
|---|---:|---:|
| Validity GRPO | no | 0 |
| Slot validity control | yes | 0 |
| SU-GRPO | yes | 0.5 |

The slot-validity control determines whether adding identifiers alone changes
behavior.

### 9.2 Objective sweep

Use:

```text
lambda in {0.0, 0.25, 0.5, 0.75, 1.0}
```

This gives an interpretable validity-coverage frontier. If compute is limited,
run `0`, `0.5`, and `1.0`.

### 9.3 Required metrics

- valid completion rate;
- unique valid consequence modes;
- exact and budget-normalized coverage at `K=4,8,16`;
- effective mode count;
- duplicate valid-mode rate;
- mode distribution per slot;
- pairwise overlap between slot distributions;
- slot-wise validity;
- response length and truncation by slot.

### 9.4 Mechanism checks

The method is behaving as intended only if:

- different slots develop different consequence distributions;
- slot specialization is semantic rather than merely textual;
- coverage increases beyond the slot-validity control;
- validity remains competitive with validity GRPO;
- improvements persist on held-out `M` and separation slices.

## 10. Falsification

Reject the method if:

- slots are ignored and have indistinguishable mode distributions;
- slots specialize only in wording, language, or response length;
- \(\lambda>0\) does not outperform the slot-validity control;
- gains disappear when evaluated at equal `K`;
- the validity-coverage frontier is dominated by ordinary temperature changes;
- results depend on selecting favorable slot identifiers after training.

## 11. Attribution boundary

The method above is motivated and derived from the set-valued objective.
Related work should separately discuss:

- latent strategy conditioning and mutual-information skill discovery;
- inverse probability scaling for outcome-level mode collapse;
- multi-answer/set-recovery RLVR;
- diversity-aware and quality-diversity objectives.

The algebraic coverage credit \(1/n(c)\) is related to inverse-frequency
scaling. That connection must be acknowledged. The methodological distinction
is the derivation from Shapley credit for an explicit validity-coverage set
utility and its use with indexed factors of a set-generating policy.
