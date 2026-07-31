# V0 Synthetic Environment Specification: Boolean Causal Micro-Lab

## 0. Purpose

This environment represents a complete scientific loop:

\[
\text{evidence} \rightarrow \text{hypotheses} \rightarrow \text{experiment} \rightarrow \text{new evidence} \rightarrow \cdots
\]

Hypothesis-generator training uses frozen evidence snapshots:

\[
E_t \rightarrow h
\]

so no multi-turn environment rollouts are required during training.

The environment supports:

- exact hypothesis validity;
- exact semantic hypothesis identities;
- exact version-space coverage;
- controllable underdetermination;
- controllable predictive separation;
- offline contextual-bandit training;
- closed-loop evaluation of scientific utility.

---

## 1. World definition

Each world contains six binary variables:

```text
X1, X2, X3     exogenous variables
Z1, Z2         intermediate variables
Y              outcome
```

The causal order is fixed:

```text
X1, X2, X3 -> Z1 -> Z2 -> Y
```

Edges may skip levels. For example, `X1` may directly affect `Y`.

Each world is a deterministic structural causal model containing one rule for each endogenous variable:

\[
Z_1 := f_1(X_1,X_2,X_3)
\]

\[
Z_2 := f_2(X_1,X_2,X_3,Z_1)
\]

\[
Y := f_3(X_1,X_2,X_3,Z_1,Z_2)
\]

### 1.1 Allowed operators

Unary:

```text
COPY(A)
NOT(A)
```

Binary:

```text
AND(A, B)
OR(A, B)
XOR(A, B)
```

Constraints:

- unary rules have exactly one input;
- binary rules have exactly two distinct inputs;
- binary inputs are sorted canonically;
- no constants;
- no cycles;
- every endogenous variable has exactly one rule.

### 1.2 Syntactic hypothesis-space size

For `Z1`:

\[
3\times2+\binom{3}{2}\times3=15
\]

For `Z2`:

\[
4\times2+\binom{4}{2}\times3=26
\]

For `Y`:

\[
5\times2+\binom{5}{2}\times3=40
\]

Total syntactic hypotheses:

\[
15\times26\times40=15{,}600
\]

This is small enough to enumerate completely.

---

## 2. Experiment space

An experiment specifies:

1. the values of `X1`, `X2`, and `X3`;
2. optionally, an intervention on `Z1` or `Z2`.

Allowed intervention choices:

```text
OBSERVE
DO_Z1_0
DO_Z1_1
DO_Z2_0
DO_Z2_1
```

There are eight exogenous assignments:

\[
(X_1,X_2,X_3)\in\{0,1\}^3
\]

Therefore, the complete experiment space contains:

\[
8\times5=40
\]

experiments.

### 2.1 Intervention semantics

For `DO_Z1_0`, replace the structural equation for `Z1` with:

```text
Z1 := 0
```

Then recompute `Z2` and `Y`. The same semantics apply to interventions on `Z2`.

### 2.2 Observation returned

Every experiment reveals:

```text
Z1, Z2, Y
```

Example:

```json
{
  "inputs": {"X1": 1, "X2": 0, "X3": 1},
  "intervention": "DO_Z1_0",
  "observation": {"Z1": 0, "Z2": 1, "Y": 0}
}
```

---

## 3. Hypothesis representation

The model emits exactly one complete world hypothesis per completion.

```json
{
  "rules": [
    {
      "target": "Z1",
      "operator": "AND",
      "inputs": ["X1", "X3"]
    },
    {
      "target": "Z2",
      "operator": "NOT",
      "inputs": ["Z1"]
    },
    {
      "target": "Y",
      "operator": "XOR",
      "inputs": ["X2", "Z2"]
    }
  ]
}
```

No free-form reasoning is required in V0.

The parser should reject:

- missing targets;
- repeated targets;
- invalid operators;
- incorrect arity;
- duplicate binary inputs;
- unavailable parents;
- noncanonical binary-input ordering;
- additional fields, if strict parsing is enabled.

Suggested internal representation:

```python
from dataclasses import dataclass
from typing import Literal

Operator = Literal["COPY", "NOT", "AND", "OR", "XOR"]


@dataclass(frozen=True)
class Rule:
    target: str
    operator: Operator
    inputs: tuple[str, ...]


@dataclass(frozen=True)
class Hypothesis:
    z1_rule: Rule
    z2_rule: Rule
    y_rule: Rule
```

---

## 4. Semantic hypothesis identity

Do not treat the program string as the hypothesis identity.

For every hypothesis, execute all 40 experiments and concatenate the resulting observations:

\[
\sigma(h)=\left[(Z_1,Z_2,Y)_{a_1},\ldots,(Z_1,Z_2,Y)_{a_{40}}\right]
\]

This 120-bit vector is the hypothesis's complete intervention signature.

Two programs are semantically equivalent when:

\[
h_i\sim h_j \iff \sigma(h_i)=\sigma(h_j)
\]

Group all 15,600 syntactic programs by signature. Each unique signature is one **semantic hypothesis mode**.

```python
import hashlib

mode_id = hashlib.sha256(pack_bits(prediction_signature)).hexdigest()
```

Select one canonical program for each mode using:

1. minimum total number of inputs;
2. then lexicographic rule ordering.

The model may output any equivalent program, but evaluation maps it to the same `mode_id`.

---

## 5. Evidence states and valid hypotheses

A hidden world is one semantic mode:

\[
h^\star
\]

An evidence state contains the outcomes of a subset of the 40 experiments:

\[
E=\{(a_j,o_j)\}_{j=1}^{t}
\]

The exact valid version space is:

\[
V(E)=\left\{h:\sigma(h)[a_j]=o_j\ \forall(a_j,o_j)\in E\right\}
\]

The model sees only `E`.

The verifier has access to the full precomputed mode table and can determine:

- whether the generated hypothesis is valid;
- its semantic mode;
- the complete valid-mode set;
- exact mode coverage;
- predictions under every unobserved experiment.

The model sees data, never the candidate hypothesis set.

---

## 6. Diversity controls

The dataset generator should control two axes exactly and annotate two others.

### 6.1 Primary control A: number of valid modes

Generate states with:

```text
M in {4, 8, 12, 16}
```

where:

\[
M=|V(E)|
\]

This is exact underdetermination.

### 6.2 Primary control B: predictive separation

Mode identity continues to use the complete interventional signature over
`(Z1, Z2, Y)`. Predictive separation instead measures disagreement on a
designated scientific prediction target. The default target is `Y`; target
channels remain configurable so that the metric is a property of predictions,
not of a hard-coded variable name.

Let `Q(E)` be the experiments that have not yet been observed. For two valid
modes, calculate target-outcome disagreement:

\[
d_Y(h_i,h_j)=\frac{1}{|Q(E)|}\sum_{q\in Q(E)}
\mathbf 1\!\left[Y_{h_i}(q)\neq Y_{h_j}(q)\right]
\]

For a state:

\[
\bar d_Y(E)=\frac{2}{M(M-1)}\sum_{i<j}d_Y(h_i,h_j)
\]

Also store:

\[
d_{\min}(E)=\min_{i\neq j}d_Y(h_i,h_j)
\]

For binary predictions, the maximum mean pairwise disagreement at one query is

\[
B_M=\frac{\lfloor M^2/4\rfloor}{\binom{M}{2}}.
\]

Also report the cross-`M` normalized value

\[
\widetilde d(E)=\frac{\bar d_Y(E)}{B_M}.
\]

Treat separation as a continuous control variable. Store the exact value
`mean_separation` for every state and do not assign semantic labels such as
low, medium, or high. To construct a finite evaluation set, choose states near
evenly spaced target values across the empirically supported interval for each
`M`. These target points are a sampling device only; evaluation and statistical
analysis use the raw continuous value. Report the selected range, percentiles,
and largest adjacent gap so that coverage of the axis is auditable. Retain
zero-separation states in the characterization bank, but exclude them from the
primary comparison set because no method can recover predictive diversity when
all valid hypotheses make identical target predictions.

### 6.3 Secondary annotation C: mechanism-family diversity

Assign each canonical program a family based on the rule for `Y`.

Source family:

```text
DIRECT       Y uses only X variables
MEDIATED     Y uses only Z variables
MIXED        Y uses one X and one Z
UNARY        Y has one parent
```

Operator family:

```text
COPY_NOT
AND
OR
XOR
```

Complete family label:

```python
family = (source_family, operator_family)
```

For each evidence state, store:

```text
within-family: 1 family
mixed:         2 families
cross-family:  3+ families
```

### 6.4 Secondary annotation D: base-policy skew

Before RL, sample the SFT model many times on each state and estimate:

\[
p_0(m\mid E)
\]

Store:

```text
balanced
moderately skewed
severely skewed
unreached valid modes
```

Measure this from the policy rather than trying to force it through environment generation.

---

## 7. Evidence-state generation

Precompute:

```python
predictions[mode_id, experiment_id] -> 3-bit outcome
```

For a hidden mode `h*`, an evidence subset determines the compatible version space.

Use beam search to find evidence states with exact target cardinality.

### 7.1 Beam-search interface

```python
def find_states(
    hidden_mode: int,
    target_mode_count: int,
    max_evidence: int = 8,
    beam_width: int = 256,
) -> list["EvidenceState"]:
    ...
```

Initial state:

```text
E = empty
V(E) = all semantic modes
```

At each depth:

1. append one unused experiment;
2. reveal the hidden world's outcome;
3. recompute the version space;
4. discard states where `|V(E)| < target_mode_count`;
5. rank remaining states by closeness to target;
6. retain the top `beam_width`.

Suggested ranking:

\[
\operatorname{score}(E)=\left|\log_2|V(E)|-\log_2M\right|+0.05|E|
\]

Whenever `|V(E)| = M`, save the state.

After collecting a large candidate bank, calculate continuous predictive
separation and select states to cover its supported range for each `M`.

### 7.2 Avoid duplicate states

Canonical state identity:

```python
state_id = hash((hidden_mode, tuple(sorted(experiment_ids))))
```

Two states with the same evidence experiments and observations are duplicates even if reached by different trajectories.

---

## 8. Dataset record

Each state should be serialized approximately as:

```json
{
  "state_id": "world_00491_state_0007",
  "visible_experiments": [
    {
      "inputs": {"X1": 0, "X2": 1, "X3": 1},
      "intervention": "OBSERVE",
      "observation": {"Z1": 1, "Z2": 0, "Y": 1}
    },
    {
      "inputs": {"X1": 1, "X2": 1, "X3": 0},
      "intervention": "DO_Z1_0",
      "observation": {"Z1": 0, "Z2": 1, "Y": 0}
    }
  ],
  "available_experiment_ids": [0, 1, 4, 7],
  "metadata": {
    "valid_mode_count": 8,
    "separation_bucket": "continuous",
    "mean_separation": 0.43,
    "minimum_separation": 0.17,
    "family_bucket": "cross_family",
    "evidence_size": 4
  },
  "private": {
    "valid_mode_ids": ["..."],
    "hidden_mode_id": "...",
    "prediction_signatures": {}
  }
}
```

The prompt builder must never expose the `private` fields.

---

## 9. Model prompt

```text
You are studying a deterministic causal system with six binary variables.

X1, X2, and X3 are externally controlled inputs.
Z1 and Z2 are intermediate variables.
Y is the final outcome.

The system contains exactly one rule for each of Z1, Z2, and Y.

Allowed rule forms:
- COPY(A)
- NOT(A)
- AND(A, B)
- OR(A, B)
- XOR(A, B)

Parent constraints:
- Z1 may use X1, X2, or X3.
- Z2 may use X1, X2, X3, or Z1.
- Y may use X1, X2, X3, Z1, or Z2.
- A binary rule must use two distinct inputs.

The following experiments have already been performed:

Experiment 1
Inputs: X1=0, X2=1, X3=1
Intervention: none
Observed: Z1=1, Z2=0, Y=1

Experiment 2
Inputs: X1=1, X2=1, X3=0
Intervention: set Z1=0
Observed: Z1=0, Z2=1, Y=0

Propose one complete causal hypothesis that is consistent with every
experiment shown above.

The evidence may permit more than one valid hypothesis. Return one hypothesis.

Return only JSON:

{
  "rules": [
    {
      "target": "Z1",
      "operator": "<COPY|NOT|AND|OR|XOR>",
      "inputs": ["<input>", "..."]
    },
    {
      "target": "Z2",
      "operator": "<COPY|NOT|AND|OR|XOR>",
      "inputs": ["<input>", "..."]
    },
    {
      "target": "Y",
      "operator": "<COPY|NOT|AND|OR|XOR>",
      "inputs": ["<input>", "..."]
    }
  ]
}
```

One completion produces one hypothesis. For a training group, sample `G` independent completions from the same prompt.

---

## 10. Verifier output

The verifier should return structured components:

```json
{
  "parse_valid": true,
  "syntax_valid": true,
  "evidence_consistent": true,
  "semantic_mode_id": "mode_0317",
  "is_currently_valid_mode": true,
  "prediction_signature": "packed bits",
  "mechanism_family": ["MEDIATED", "XOR"]
}
```

Group-level evaluation:

```json
{
  "num_samples": 8,
  "num_parse_valid": 8,
  "num_evidence_consistent": 7,
  "num_unique_valid_modes": 5,
  "available_valid_modes": 8,
  "exact_coverage": 0.625,
  "budget_normalized_coverage": 0.625,
  "predictive_diversity_recovery": 0.71,
  "effective_mode_count": 4.2,
  "family_coverage": 0.75
}
```

Definitions:

\[
\text{ExactCoverage@}G=\frac{|\{\text{valid generated mode IDs}\}|}{|V(E)|}
\]

Because this is bounded by `G/M`, also report:

\[
\text{BudgetCoverage@}G=\frac{|\{\text{valid generated mode IDs}\}|}{\min(G,M)}
\]

The primary diversity metric is Predictive Diversity Recovery. For a generation
budget `K`, invalid hypotheses and repeated semantic modes contribute zero
pairwise mass:

\[
\operatorname{PDR@K}=
\frac{
\sum_{i<j}d_Y(\hat h_i,\hat h_j)
\mathbf 1[\hat h_i,\hat h_j\in V(E),\,\hat h_i\neq\hat h_j]
}{
\max_{T\subseteq V(E),\,|T|\leq K}
\sum_{\{u,v\}\subseteq T}d_Y(u,v)
}.
\]

The denominator is computed exactly. With `M <= 16`, exhaustive subset search
is small enough for local evaluation. Primary rankings use cells with `K <= M`
so that the hypothesis budget is genuinely constrained.

---

## 11. Cold-start SFT

SFT is only for:

- learning the JSON schema;
- learning the causal DSL;
- generating evidence-consistent programs at a nontrivial frequency.

For each evidence state:

1. uniformly sample a semantic mode from `V(E)`;
2. use its canonical program as the target;
3. create separate examples for several different valid modes.

Do not always use the simplest mode.

Suggested dataset:

```yaml
sft:
  train_states: 10000
  targets_per_state: 2
  total_examples: 20000
  epochs: 1
  max_prompt_length: 2048
  max_response_length: 256
```

SFT exit criteria:

```text
parse validity:       > 99%
syntax validity:      > 98%
evidence validity:    > 40-50% initially
nonempty hypotheses:  ~100%
```

SFT does not need to produce uniform diversity. It only needs to produce a usable policy.

---

## 12. RL training protocol

This is contextual-bandit RL.

Per update:

```text
1. Sample B evidence states.
2. Generate G independent hypotheses for each state.
3. Parse every hypothesis.
4. Map every valid hypothesis to an exact semantic mode.
5. Calculate individual and set-level rewards.
6. Update the policy.
```

No experiment is selected or executed during training.

Suggested first configuration:

```yaml
model: Qwen/Qwen3-4B

batch:
  states_per_update: 32
  samples_per_state: 8

generation:
  max_prompt_length: 2048
  max_response_length: 256
  temperature: 1.0
  top_p: 1.0

training:
  train_states: 20000
  validation_states: 2000
  epochs: 1
```

### 12.1 Baseline reward

Validity-only GRPO:

\[
r_i=\mathbf 1[h_i\in V(E)]
\]

### 12.2 Environment reward signals to expose

The environment should expose, but not hard-code into one objective:

```text
format validity
syntax validity
evidence consistency
semantic mode ID
number of duplicate samples in the same mode
marginal exact coverage
prediction distance from other samples
family identity
```

A simple first diversity reward baseline:

\[
r_i=\mathbf 1[h_i\in V(E)]\left(1+\lambda\frac{1}{n(m_i)}\right)
\]

where `n(m_i)` is the number of generated samples in the same semantic mode.

Also expose exact marginal coverage:

\[
\Delta_i=C(S)-C(S\setminus\{h_i\})
\]

The final method may use richer credit assignment, but the environment should provide both.

---

## 13. Required baselines

Run all trainable comparisons from the same SFT checkpoint:

1. SFT only;
2. validity-only GRPO;
3. entropy-regularized GRPO;
4. higher-temperature inference;
5. verbalized-diversity prompting;
6. generate-many-then-select;
7. output-identity balancing;
8. the proposed method.

Essential inference-time baseline:

```text
Generate 64 ordinary samples.
Discard invalid hypotheses.
Greedily select 8 unique semantic modes.
```

Compare the trained method:

- at equal generation budget;
- at equal returned-set size.

---

## 14. Synthetic closed-loop evaluation

Training does not require environment rollouts. Evaluation does.

At each step:

1. the generator produces `G` hypotheses;
2. invalid outputs are discarded;
3. a fixed planner selects the unobserved experiment with maximum disagreement;
4. execute it in the hidden world;
5. append the observation;
6. repeat.

### 14.1 Fixed disagreement planner

For each available experiment `a`, inspect generated hypotheses' predicted outcomes:

\[
\{\sigma_a(h):h\in S\}
\]

Select:

\[
a^\star=\arg\max_a H(\{\sigma_a(h):h\in S\})
\]

where `H` is empirical entropy over the possible three-bit outcomes.

Use the same planner for every hypothesis generator.

### 14.2 Closed-loop metrics

Report:

- hidden-world identification success after `T` experiments;
- mean remaining version-space size;
- information gained per experiment;
- number of experiments required to reach `|V(E)| = 1`;
- regret relative to an oracle planner using the full true version space;
- hypothesis coverage at every evidence stage.

This connects diversity to scientific utility.

---

## 15. Data splits

Primary split:

```text
70% hidden semantic modes: train
15% hidden semantic modes: validation
15% hidden semantic modes: test
```

All evidence states derived from a hidden mode remain in the same split.

Also create an optional compositional OOD split by holding out selected `Y` motifs, for example:

```text
XOR with one direct and one mediated parent
NOT of Z2
OR of Z1 and Z2
```

OOD performance is not required for the first pilot.

---

## 16. Repository layout

```text
src/scattered_discovery/envs/causal_micro_lab/
    dsl.py
    enumerate_hypotheses.py
    simulator.py
    interventions.py
    signatures.py
    canonicalize.py
    state_generator.py
    prompt_builder.py
    parser.py
    verifier.py
    rewards.py
    planner.py
    episode.py

scripts/
    build_hypothesis_table.py
    generate_evidence_states.py
    build_sft_dataset.py
    run_closed_loop_eval.py

data/causal_micro_lab/
    modes.parquet
    experiments.parquet
    states_train.parquet
    states_val.parquet
    states_test.parquet
    sft_train.parquet
```

---

## 17. Build order

Implement in this order:

1. DSL parser and canonical renderer.
2. Deterministic simulator.
3. Forty-experiment enumerator.
4. Enumerate all 15,600 programs.
5. Group programs by prediction signature.
6. Exact version-space lookup.
7. Evidence-state beam search.
8. Prompt and output parser.
9. Exact verifier and evaluation metrics.
10. Dataset writer.
11. SFT smoke test.
12. Validity-only GRPO.
13. Diversity reward.
14. Closed-loop disagreement planner.

---

## 18. Pilot acceptance tests

Before serious training, verify:

```text
✓ all syntactic programs enumerate deterministically
✓ semantic-equivalence grouping is stable
✓ states exist for M = 4, 8, 12, 16
✓ continuous separation coverage is broad and has no large unsupported gaps
✓ random valid-mode targets pass the verifier
✓ invalid modes fail the verifier
✓ oracle mode-conditioned generation reaches full budget coverage
✓ closed-loop oracle planner identifies the world within the expected budget
```
