# Method Audit: Budgeted Mode-Coverage Policy Optimization

Status: rejected as a standalone novel method after algebraic and
literature-overlap audit. Retained as a mathematically aligned baseline and
analysis tool.

Working name: **Budgeted Mode-Coverage Policy Optimization** (`BMCPO`).

## 1. Verdict

The previous candidates do not clear the novelty or objective-alignment bar:

- inverse-count credit is IPS/GAPO-style frequency correction;
- Shapley credit for cardinality coverage reduces to inverse-count credit;
- leave-one-out set credit is already covered by set-RL and SGRPO;
- latent strategy identifiers plus mutual information are already UpSkill;
- generic distance or determinant rewards return to CD-GRPO, DPPs, and generic
  quality-diversity optimization.

The candidate starts from the actual evaluation quantity:

> Given a fixed budget of \(K\) generations, maximize the expected number of
> distinct evidence-valid semantic hypothesis modes returned.

This yields a finite-budget objective, an analytic policy gradient, and a
bounded unbiased estimator. It is not inverse-probability scaling.

However, it does not clear the novelty bar as a standalone algorithm:

- Poly-EPO already provides set RL under arbitrary objectives with unbiased
  subset estimators. Choosing distinct valid-mode cardinality as that objective
  recovers the core construction here.
- SGRPO already redistributes set diversity through leave-one-out
  contributions. At \(K=N\), the estimator below is exactly the singleton
  leave-one-out contribution for cardinality.
- Outcome-Based Exploration already penalizes repeated outcomes within a
  rollout batch to improve test-time diversity.

The closed form below is still useful: it gives a clean, bounded,
budget-specific baseline and a precise analysis of how finite-\(K\) coverage
differs from IPS. It must not be presented as the thesis's new method.

## 2. Training constraints

The method must operate under the same conditions as the validity-GRPO
baseline:

```text
model: Qwen3-4B
dataset: causal_micro_lab/trainable
states per update: 16
rollouts per state N: 8
completions per update: 128
response limit: 6000
length shaping begins: 3072
training updates: 100
```

At reward time, it may use only:

- the sampled completion;
- strict parsing and evidence validation;
- the candidate's predictions on public probe experiments;
- other sampled completions from the same rollout group.

It may not use:

- the private valid-mode set;
- the hidden world;
- target mode count \(M\);
- an oracle query for an unseen valid mode;
- a persistent per-state archive, because states are generally seen once.

## 3. Outcome distribution

For evidence state \(x\), let:

- \(y\sim\pi_\theta(\cdot\mid x)\) be one completion;
- \(v_x(y)\in\{0,1\}\) indicate evidence validity;
- \(\phi_x(y)\) be the candidate's observable consequence signature;
- \(m=\phi_x(y)\) be its semantic outcome when \(v_x(y)=1\).

Define the policy probability of valid mode \(m\):

\[
p_m(\theta\mid x)
=
\Pr_{\pi_\theta}\left(v_x(y)=1,\ \phi_x(y)=m\mid x\right).
\]

The total single-attempt validity probability is:

\[
q(\theta\mid x)=\sum_m p_m(\theta\mid x).
\]

Invalid outputs occupy the remaining probability mass.

## 4. Finite-budget coverage

For \(K\) independent generations, mode \(m\) is observed at least once with
probability:

\[
1-(1-p_m)^K.
\]

Therefore, the expected number of distinct valid modes returned is:

\[
F_K(\theta\mid x)
=
\sum_m \left[1-(1-p_m(\theta\mid x))^K\right].
\]

This is exactly the expectation of the numerator used by unique-mode
coverage@\(K\). It does not require knowing how many valid modes exist.

Use the normalized validity-coverage objective:

\[
J_{\lambda,K}(\theta)
=
\mathbb{E}_{x}
\left[
(1-\lambda)q(\theta\mid x)
+
\lambda\frac{F_K(\theta\mid x)}{K}
\right].
\]

Interpretation:

- \(\lambda=0\): validity-only RL;
- \(K=1\): validity-only RL for every \(\lambda\);
- \(K>1,\lambda>0\): trade some concentration pressure for finite-budget
  mode coverage;
- division by \(K\) keeps both terms on a comparable \([0,1]\) scale.

The primary candidate is:

```text
N = 8 sampled rollouts
K = 4 target return budget
lambda = 0.5
```

The smallest evaluated return budget is the most demanding coverage regime.
Using \(K=4<N=8\) also gives smoother, lower-variance credit than \(K=N=8\).

## 5. Exact gradient

Differentiate \(F_K\):

\[
\nabla_\theta F_K
=
\sum_m
K(1-p_m)^{K-1}\nabla_\theta p_m.
\]

Using
\(\nabla p_m
=
\mathbb{E}[
\mathbf 1\{v=1,\phi=m\}\nabla\log\pi_\theta(y\mid x)]\),
the objective gradient is:

\[
\nabla_\theta J_{\lambda,K}
=
\mathbb{E}_{x,y\sim\pi_\theta}
\left[
v_x(y)
\left(
(1-\lambda)
+
\lambda(1-p_{\phi_x(y)})^{K-1}
\right)
\nabla_\theta\log\pi_\theta(y\mid x)
\right].
\]

The coverage component gives a valid outcome the weight:

\[
w_K(m)=(1-p_m)^{K-1}.
\]

Properties:

- \(w_K(m)\in[0,1]\);
- rare valid modes approach weight \(1\);
- common valid modes approach weight \(0\);
- the pressure is explicitly determined by inference budget \(K\);
- no inverse-probability singularity or clipping constant is required.

## 6. Unbiased group estimator

The true \(p_m\) is unknown. Suppose the training group contains \(N\ge K\)
i.i.d. rollouts and sampled valid mode \(m_i\) appears \(n_i\) times.

For rollout \(i\), estimate the probability that \(K-1\) other samples contain
no copy of \(m_i\):

\[
\widehat w_i^{(K)}
=
\frac{
\binom{N-n_i}{K-1}
}{
\binom{N-1}{K-1}
}.
\]

Conditional on \(m_i\), this is a U-statistic satisfying:

\[
\mathbb{E}\left[\widehat w_i^{(K)}\mid m_i\right]
=(1-p_{m_i})^{K-1}.
\]

Thus the raw valid credit:

\[
a_i
=
v_i
\left[
(1-\lambda)
+
\lambda\widehat w_i^{(K)}
\right]
\]

gives an unbiased policy-gradient estimator for \(J_{\lambda,K}\), before
PPO clipping and any group-dependent normalization.

For \(N=8\), the coverage weights are:

| within-group count \(n_i\) | \(K=2\) | \(K=4\) | \(K=8\) |
|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 1.000 |
| 2 | 0.857 | 0.571 | 0.000 |
| 3 | 0.714 | 0.286 | 0.000 |
| 4 | 0.571 | 0.114 | 0.000 |
| 5 | 0.429 | 0.029 | 0.000 |
| 6 | 0.286 | 0.000 | 0.000 |
| 7 | 0.143 | 0.000 | 0.000 |
| 8 | 0.000 | 0.000 | 0.000 |

At \(\lambda=0.5,K=4\), the complete verifier ladder before length shaping is:

```text
unique valid mode:                  1.000
valid mode appearing twice:         0.786
valid mode appearing three times:   0.643
valid mode appearing four times:    0.557
minimum valid-mode credit:          0.500
syntax-valid, evidence-invalid:      0.200
unparseable or empty:                0.000
```

Every valid output remains strictly preferable to a syntax-only output.

## 7. Advantage construction

Ordinary within-group standardization depends on rollout \(i\) and generally
breaks the exact unbiasedness claim. The theoretically clean version uses a
leave-one-out baseline that is recomputed using only the other \(N-1\)
rollouts:

\[
A_i=a_i-b(y_{-i}).
\]

Any \(b(y_{-i})\) is independent of \(y_i\) and is therefore a valid baseline.
A practical baseline is the mean BMCPO credit recomputed on the group with
rollout \(i\) removed.

This requires \(K\le N-1\), another reason to use \(K=4,N=8\) rather than
\(K=N\).

Do not divide by the observed group standard deviation in the primary
theoretical implementation. Credits are already bounded. A fixed,
predeclared scalar may be used to set optimizer scale.

Existing overlength handling remains unchanged:

- no penalty through 3072 response tokens;
- linear penalty to \(-0.2\) at 6000 tokens;
- cap-hit samples masked from policy loss.

## 8. What the objective guarantees

### 8.1 Uniformity at the global optimum

For \(K>1\), each term \(1-(1-p)^K\) is strictly concave in \(p\). Conditional
on all probability mass being distributed over \(M\) reachable valid modes,
\(F_K\) is symmetric and strictly concave. It is uniquely maximized by:

\[
p_1=\cdots=p_M=\frac{1}{M}.
\]

Moving probability mass from an invalid output to any valid mode increases
both \(q\) and \(F_K\). Therefore, the global optimum of
\(J_{\lambda,K}\), for \(\lambda\in(0,1)\), places all reachable mass on valid
modes and distributes it uniformly among them.

### 8.2 What it does not guarantee

BMCPO cannot update a mode that the policy never samples.

More strongly, if every rollout in a group produces the same valid mode, every
permutation-equivariant group-local method assigns equal credit to those
rollouts. Centering the credits yields zero advantage. This affects:

- validity GRPO;
- empirical IPS;
- GAPO-style count correction;
- BMCPO;
- any duplicate penalty based only on that homogeneous group.

No reward reshaping can manufacture evidence about an unsampled outcome.
Exploration must initially come from the base policy, sampling temperature,
prompt variation, sequential sampling, or an explicitly conditioned skill
policy.

BMCPO is therefore a **discovery-retention method**, not a support-creation
method: once a lower-probability valid mode appears, it receives more credit
than a dominant mode and is less likely to be erased.

## 9. Algebraic comparison with IPS

Empirical IPS uses:

\[
r_i^{\mathrm{IPS}}
=
\frac{v_i}{\max(n_i/N,\epsilon)}.
\]

BMCPO uses:

\[
r_i^{\mathrm{BMCPO}}
=
v_i
\left[
(1-\lambda)
+
\lambda
\frac{\binom{N-n_i}{K-1}}{\binom{N-1}{K-1}}
\right].
\]

They differ in objective and dynamics:

| Property | IPS | BMCPO |
|---|---|---|
| target | reward-proportional outcome distribution | expected unique valid modes at budget \(K\) |
| rare-mode weight | proportional to \(1/p\) | proportional to \((1-p)^{K-1}\) |
| maximum weight | unbounded without clipping | bounded |
| tuning parameter | probability clip \(\epsilon\) | evaluation budget \(K\) |
| finite-\(K\) alignment | indirect | explicit |
| exact group estimator | plug-in frequency | unbiased U-statistic |

Both have a uniform distribution as the equal-reward global optimum. The
claim is not a different optimum; it is a metric-aligned, bounded gradient and
finite-sample estimator.

## 10. Closest-work audit

### Validity GRPO

Optimizes \(q\) and is indifferent to how valid mass is distributed. It is the
required baseline.

### IPS-GRPO

[IPS-GRPO](https://arxiv.org/abs/2601.21669) cancels empirical
outcome-frequency amplification using inverse probability. BMCPO is not
inverse probability scaling and does not target the IPS reward-proportional
distribution dynamics.

### GAPO

[GAPO](https://arxiv.org/abs/2511.12596) introduces group-aware rewards and a
frequency-aware uniform-sampling reward. This rules out claiming group-aware
or frequency-aware GRPO as novel.

### UCPO

[UCPO](https://arxiv.org/abs/2605.00365) explicitly targets the uniform
distribution over correct solutions using conditional uniformity and
importance weighting. This rules out claiming uniform valid-mode allocation as
the sole novelty. BMCPO instead derives from finite-budget expected
cardinality and uses a bounded occupancy estimator.

### Poly-EPO and SGRPO

[Poly-EPO](https://arxiv.org/abs/2604.17654) gives a general set-RL recipe and
unbiased subset-based gradient estimators.
[SGRPO](https://arxiv.org/abs/2605.08659) trains with set-level diversity and
leave-one-out contributions. These rule out claiming set-level policy
optimization or leave-one-out diversity credit as novel.

BMCPO is an analytic specialization of set RL. For a subset
\(T\) of \(K\) rollouts, define

\[
U(T)=\left|\{\phi(y):y\in T,\ v(y)=1\}\right|.
\]

Using \(U\) as Poly-EPO's arbitrary set objective and averaging over
\(K\)-subsets produces the same finite-budget occupancy objective. The
closed-form coefficient in Section 6 is a simplification of that subset
estimator for categorical cardinality, not a distinct policy-optimization
principle.

When \(K=N\),

\[
\widehat w_i^{(N)}=\mathbf 1\{n_i=1\},
\]

which is exactly the leave-one-out cardinality contribution
\(U(S)-U(S\setminus\{i\})\) used by a set-credit construction such as SGRPO.

### Outcome-Based Exploration

[Outcome-Based Exploration](https://arxiv.org/abs/2509.06941) explicitly
distinguishes historical exploration from test-time batch diversity and uses
the within-batch bonus:

\[
b_i^{\mathrm{batch}}
=-\frac{1}{N}\sum_{j\ne i}\mathbf 1\{m_i=m_j\}.
\]

This is not the same finite-\(K\) coefficient as BMCPO, but it already
establishes outcome-level anti-repetition within an LLM RL group. It rules out
claiming that the central mechanism of rewarding non-repeated semantic
outcomes is new.

### PKPO

[Pass@K Policy Optimization](https://arxiv.org/abs/2505.15201) derives
unbiased estimators for the probability that at least one of \(K\) samples is
correct. That objective depends only on total correctness mass:

\[
1-(1-q)^K.
\]

BMCPO instead sums the occupancy probability separately for every valid
semantic mode:

\[
\sum_m [1-(1-p_m)^K].
\]

PKPO does not distinguish two correct solutions from repeated copies of one.

### Multi-Answer RL

[Multi-Answer RLVR](https://arxiv.org/abs/2603.24844) trains one completion to
contain a set of distinct answers and rewards set recovery. BMCPO retains the
single-hypothesis output interface and optimizes diversity across independent
rollouts.

### UpSkill

[UpSkill](https://arxiv.org/abs/2602.22296) conditions generation on latent
strategy identifiers and maximizes mutual information. This rules out
presenting neutral slots or latent controllability as a new contribution.
UpSkill is a relevant support-creation baseline.

### Sequential sampling

[SESA](https://arxiv.org/abs/2510.15502) conditions later generations on
earlier method sketches to force broader exploration. It is another relevant
support-creation baseline, but changes the rollout policy and inference
procedure.

### VPO

[Vector Policy Optimization](https://arxiv.org/abs/2605.22817) uses
vector-valued task rewards so candidates specialize to different downstream
trade-offs. BMCPO requires only scalar validity plus a semantic outcome map.

### CD-GRPO and DPP-style rewards

CD-GRPO rewards geometric separation between consequence signatures. BMCPO
uses only semantic equality for exact-mode coverage. It has no kernel,
bandwidth, determinant, archive, or pairwise-distance scale.

## 11. Novelty boundary

Safe claims if this objective is used as a baseline or analytical component:

- a task-specific instantiation of set RL for finite-budget unique valid-mode
  count in plural scientific hypothesis generation;
- a closed-form simplification of a subset/U-statistic credit estimator for
  categorical mode coverage;
- exact candidate-execution semantics rather than text embeddings or an
  LM judge;
- an enumerated environment where the objective, estimator, and true coverage
  can all be checked exactly.

Claims that are not safe:

- first set-level RL objective;
- first diversity-aware GRPO;
- first group-frequency correction;
- first uniform-correct policy method;
- first latent-conditioned diverse LLM;
- first leave-one-out diversity reward;
- first direct optimization of a generic pass@\(K\) objective.
- first within-batch outcome anti-repetition method.

## 12. Falsifiable predictions

Relative to validity GRPO:

- valid rate may fall slightly;
- unique valid modes at \(K=4\) should rise;
- duplicate valid-mode rate should fall;
- the effect should be strongest for \(M=8,16\);
- gains should be larger in medium/high-separation states if the base policy
  already samples multiple modes there.

Relative to IPS:

- reward and advantage variance should be lower;
- no samples should hit a probability-clipping boundary;
- validity should be more stable because every valid completion receives at
  least \(1-\lambda\);
- BMCPO should be strongest at its trained budget \(K=4\);
- IPS may achieve a flatter global mode distribution after much longer
  training.

Reject the method if:

- it does not improve exact or budget-normalized coverage@4 over validity
  GRPO;
- it is dominated by IPS at equal validity and compute;
- improvements disappear after controlling for temperature;
- only syntactic diversity rises while consequence-mode coverage does not;
- the estimator's empirical variance is not lower than clipped IPS;
- a closer prior work is found that uses the same finite-mode occupancy
  objective and estimator.

## 13. Minimum experiment

Run three compute-matched arms on the same rows and base checkpoint:

| Arm | Group objective |
|---|---|
| Validity GRPO | \(v_i\) |
| IPS-GRPO | \(v_i/\max(\hat p_i,\epsilon)\) |
| BMCPO | \(v_i[(1-\lambda)+\lambda\widehat w_i^{(K)}]\) |

Primary BMCPO configuration:

```text
N: 8
K: 4
lambda: 0.5
states/update: 16
responses/update: 128
updates: 100
temperature: 1.0
top_p: 1.0
max_response_length: 6000
length_penalty_start: 3072
```

Required ablations after the primary run:

```text
K in {2, 4, 8}
lambda in {0.25, 0.5, 0.75}
```

Do not run the full grid unless the primary arm first beats both validity GRPO
and IPS on held-out coverage@4.

## 14. Decision

BMCPO passes three of the four pre-implementation checks but fails the one
needed for a standalone method contribution:

1. **Objective:** explicit finite-budget expected valid-mode coverage.
2. **Mechanism:** rare sampled valid modes receive the exact marginal
   occupancy gradient; common modes receive less credit.
3. **Novelty:** **fail.** Although algebraically distinct from IPS, it is a
   direct categorical-cardinality specialization of Poly-EPO/set RL, with
   close overlap to SGRPO leave-one-out credit and Outcome-Based Exploration.
4. **Failure analysis:** cannot recover modes absent from the sampled group;
   requires initial support and should be compared with support-creation
   methods separately.

It is reasonable to implement BMCPO as a strong, metric-aligned baseline. It
is not reasonable to rename it and present it as the thesis's novel algorithm.
The thesis can still contribute the exact benchmark, the empirical diagnosis
of validity-RL collapse, and a controlled comparison of validity GRPO, IPS,
Outcome-Based Exploration, and set-coverage RL. A genuinely new method needs a
mechanism beyond same-group categorical frequency or generic set utility.
