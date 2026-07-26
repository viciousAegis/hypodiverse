# Non-Oracle Diversity Method Design

Status: superseded research exploration. The current method candidate is
documented in [`latent_ips_grpo_method.md`](latent_ips_grpo_method.md).
The rejected finite-budget set-reward candidate remains documented in
[`budgeted_mode_coverage_method_audit.md`](budgeted_mode_coverage_method_audit.md).

This document separates the problem into claims that can be supported by the
available training signal. It records rejected ideas as well as the one current
candidate so that a failed proposal is not repeatedly renamed and reintroduced.

## 1. Target problem

For an evidence state \(x\), let:

- \(Y_x\) be the space of model completions;
- \(v(x,y) \in \{0,1\}\) indicate whether completion \(y\) is a valid
  hypothesis;
- \(c_x(y)\) be the candidate's observable behavior, obtained by executing the
  candidate on the public probe experiments;
- \(\pi_\theta(y\mid x)\) be the model policy.

The evaluation target for a rollout set \(S\) is coverage of distinct valid
behaviors:

\[
C_x(S)=\left|\{c_x(y):y\in S,\ v(x,y)=1\}\right|.
\]

The method may use \(x\), generated \(y\), the parser, the verifier on visible
evidence, candidate execution on public probes, and model probabilities. It may
not use the private valid-mode set, hidden world, mode cardinality, or a query
that asks whether a desired behavior is feasible.

## 2. The problem has two different failure mechanisms

### 2.1 Reachability

A behavior cannot receive any learning signal if no rollout produces it.
Within-group diversity rewards, IPS, and distribution-matching objectives only
operate on sampled outcomes. They cannot directly reinforce a mode absent from
the batch.

For behavior probabilities \(p_m\) and \(G\) independent rollouts, expected
coverage is:

\[
\mathbb{E}[C@G]=\sum_m \left[1-(1-p_m)^G\right].
\]

This is maximized by spreading rollout probability across modes. Post-hoc
reward shaping does not alter the current batch's \(p_m\).

### 2.2 Retention

Once several valid behaviors are sampled, expected-return optimization can
amplify the most frequent or slightly better rewarded behavior. IPS-GRPO,
forward-distribution matching, and behavior-frequency rewards are possible
retention mechanisms.

A complete method needs both:

1. a proposal mechanism that reaches more of the existing valid support;
2. an update rule that does not immediately collapse that support again.

## 3. Identifiability boundary

Binary validity alone does not identify semantic diversity. Two tasks can have
the same prompts, completions, validity labels, and policy probabilities while
partitioning valid completions into different semantic modes. An algorithm
observing only the former must behave identically on both tasks, although their
coverage objectives differ.

Therefore every semantic-diversity method must state where its behavior
distinction comes from. In the micro-lab, the legitimate source is the
candidate's own executable consequence signature. It is not a hidden answer,
but it is additional task structure. In an open-ended domain, an analogous
behavior descriptor or consequence evaluator would be required.

## 4. Rejected directions

| Direction | Reason for rejection |
|---|---|
| Desired prediction constraints | Changes the task and can become answer steering; feasibility filtering would use private information. |
| Within-batch distance reward alone | Cannot reward an unsampled mode; this is the reachability failure in CD-GRPO. |
| State-keyed archive | Training states are effectively seen once, so a per-state archive has no useful cross-update memory. |
| Global semantic density or inverse frequency | Primarily overlaps IPS-GRPO, semantic density rewards, and quality-diversity methods; still depends on initially sampling a mode. |
| Reverse-KL or forward-distribution matching | Already directly occupied by recent mode-covering RL work, including DMPO. |
| Parameter-space perturbations | Directly occupied by PSN-RLVR. |
| Prefix branching or first-token allocation | Directly occupied by ROSE, REFT, and related tree/prefix exploration methods. |
| Multi-answer generation | A legitimate baseline, but already the subject of Multi-Answer RLVR and not a new single-answer exploration method. |
| Higher temperature or entropy | Necessary baseline; does not distinguish useful hypotheses from arbitrary tail text. |

## 5. Current candidate: representation-orbit exploration

### 5.1 Core idea

The same evidence state can have multiple exactly equivalent presentations.
Although these presentations contain identical information, an LLM is not
perfectly invariant to them. A hypothesis suppressed under one presentation
may be reachable under another.

Let \(T\) be a set of known, validity-preserving transformations. For the first
micro-lab pilot, use only transformations that leave the answer language
unchanged:

- permutations of the visible evidence rows;
- permutations of operator-description order;
- permutations of other explicitly unordered prompt sections.

Do not initially use variable renaming. Although it is mathematically
invertible, it complicates canonicalization and can cross the current
hidden-mode data split.

Define the orbit proposal:

\[
q_\theta(y\mid x)=\frac{1}{|T|}\sum_{\tau\in T}
\pi_\theta(y\mid \tau(x)).
\]

Sampling from \(q\) requires no new model:

1. sample a presentation transform \(\tau\);
2. generate from \(\pi_\theta(\cdot\mid\tau(x))\);
3. verify the result against the original evidence state.

No observations or requested outcomes are added. The method changes where the
rollout budget is spent, not what counts as correct.

### 5.2 Why it could improve reachability

Suppose a state has two valid modes. Under the canonical prompt their
probabilities are \((0.99,0.01)\); under an equivalent presentation they are
\((0.60,0.40)\). Mixing the two presentations gives
\((0.795,0.205)\).

With eight rollouts, the probability of observing both modes changes from:

\[
1-0.99^8-0.01^8 \approx 0.077
\]

to:

\[
1-0.795^8-0.205^8 \approx 0.840.
\]

This gain comes from accessing a different representation-conditioned path,
not from revealing which second mode should be produced.

There is no guaranteed gain. If the model is already invariant or every
presentation favors the same mode, \(q=\pi\) or remains equally concentrated.
That makes the proposal cheaply falsifiable before RL.

### 5.3 Training variants

Use \(G=16\) total completions per state in every arm.

**Orbit-GRPO: reachability only**

- Render four deterministic evidence-order views.
- Allocate four rollouts to each view.
- Keep the existing `0 / 0.2 / 1.0` validity reward and length shaping.
- Normalize rewards across all 16 rollouts belonging to the original state.
- Apply the policy loss to each trajectory under the presentation that
  generated it. This remains on-policy.

**Orbit-IPS-GRPO: reachability plus retention**

- Use the same orbit proposal.
- Canonicalize valid candidates by their executable consequence signature.
- Apply IPS-GRPO's frequency correction across the 16 canonical behaviors.
- Do not add CD-GRPO's log-det reward or archive.

This decomposition is important experimentally:

| Arm | Proposal | Retention objective |
|---|---|---|
| Validity GRPO | canonical i.i.d. | expected validity |
| IPS-GRPO | canonical i.i.d. | inverse probability scaling |
| Orbit-GRPO | equivalent prompt orbit | expected validity |
| Orbit-IPS-GRPO | equivalent prompt orbit | inverse probability scaling |

The \(2\times2\) design tests whether reachability and retention are distinct,
complementary bottlenecks. It also prevents attributing an IPS improvement to
the new proposal mechanism.

## 6. Required falsification before implementation

### Gate 1: zero-training orbit gain

Evaluate the same base model and states with:

- 16 canonical i.i.d. rollouts;
- four evidence-order views with four rollouts each.

Use identical total generation tokens and decoding settings. Measure paired
differences in exact mode coverage, validity, and response length.

Stop if orbit sampling does not improve mean exact coverage or if any apparent
gain is explained only by lower validity.

### Gate 2: genuinely new semantic modes

For each state, count modes:

- found by both canonical and orbit sampling;
- canonical only;
- orbit only.

Inspect examples from the orbit-only cell. Stop if differences are merely
syntactic programs with the same consequence signature.

### Gate 3: source of gain

Compare evidence-order permutations individually. The gain must not come from
one accidentally easier fixed ordering. A useful orbit should show
complementary mode support across views.

### Gate 4: short training transfer

Run at most ten matched updates of Orbit-GRPO and validity GRPO. Evaluate both
on the canonical prompt only. Stop if orbit-only training diversity fails to
transfer to canonical evaluation.

### Gate 5: literature collision

Before making a novelty claim, specifically compare against:

- permutation self-consistency and prompt ensembling;
- stochastic beam sampling without replacement;
- REFT and ROSE rollout diversification;
- IPS-GRPO and DMPO retention objectives.

The claim cannot be “prompt permutations create diverse outputs.” The possible
contribution is the reachability-retention decomposition and an on-policy,
compute-matched orbit proposal for plural-answer RLVR.

## 7. Leakage and fairness audit

The method is admissible only if all of the following hold:

- every transformed prompt is a bijective presentation of exactly the same
  visible evidence;
- no transform is selected using private valid modes or observed model
  correctness;
- transforms are fixed before generation from `state_id` and rollout index;
- all arms use the same number of completions and maximum generated tokens;
- evaluation uses the same canonical states and seeds;
- private mode information is used only for evaluation metrics;
- candidate consequence signatures used by IPS are computed from the
  candidate program, not looked up in the private valid-mode table.

## 8. Generality claim

The method is not universal. It applies when a task has known nuisance
symmetries: evidence order, variable names, unit systems, coordinate frames, or
other invertible representations. This is common in structured scientific
reasoning, program synthesis, theorem proving, and molecular representations.

The general claim would be:

> Equivalent representations expose different parts of an LLM's valid
> hypothesis support. Treating those representations as a rollout proposal,
> and separately correcting outcome-frequency amplification during learning,
> improves plural-answer coverage under fixed generation compute.

If the zero-training test fails, this candidate should be rejected rather than
rescued with private feasibility information or additional desired outcomes.

## 9. Nearest literature

- [Expected Return Causes Outcome-Level Mode Collapse and IPS-GRPO](https://arxiv.org/abs/2601.21669)
- [DMPO: Distribution Matching for Diverse Reasoning](https://arxiv.org/abs/2605.19461)
- [PSN-RLVR: Parameter-Space Noise](https://arxiv.org/abs/2602.02555)
- [ROSE: Semantically Diverse Exploration](https://arxiv.org/abs/2601.05053)
- [REFT: First-Token Diversification](https://arxiv.org/abs/2605.28295)
- [Stochastic Beams and Gumbel-Top-k Sampling](https://arxiv.org/abs/1903.06059)
