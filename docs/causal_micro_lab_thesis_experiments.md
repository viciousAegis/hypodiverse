# Causal Micro-Lab Thesis Experiment Tracker

Last updated: 2026-07-25

## Thesis Claims

1. Validity-only RLVR can improve the probability of producing a valid
   hypothesis without recovering the full set of plausible semantic modes.
2. Consequence-Diversity GRPO (CD-GRPO) can improve semantic mode coverage at
   matched training and inference compute.
3. Consequence-level credit and cross-step novelty are responsible for the
   improvement, rather than textual variation or temperature alone.
4. Broader valid hypothesis coverage improves downstream experiment selection.

## Status Legend

- `[ ]` Not started
- `[~]` In progress
- `[x]` Complete
- `[!]` Blocked or needs revision

## Experimental Invariants

Unless an experiment explicitly varies one of these settings:

- Model: Qwen3-4B
- Training data: frozen `data/causal_micro_lab/trainable`
- Training ambiguity levels: `M in {4, 8, 16}`
- Response limit: 6,000 tokens
- Length shaping starts at 3,072 tokens for CD-GRPO
- Temperature: 1.0
- Top-p: 1.0
- Training generation budget: 128 completions per update
- Evaluation generation budget: one `K=16` run, with `K=4` and `K=8`
  calculated from prefixes
- Final comparisons use the same evaluation states and random seeds
- Confidence intervals are calculated over states, not individual generations

## Active Run

- [~] **E04: Full CD-GRPO**
- W&B:
  <https://wandb.ai/akshitsinha3/scattered-discovery/runs/a2hptg9i>
- Experiment name: `causal_micro_lab_cluster_cd_grpo_logdet_100step_r1`
- Hardware: 2 x H100 80 GB
- Updates: 100
- Batch: 8 states x 16 rollouts = 128 completions per update
- Dataset: 6,144 train rows and 128 validation rows
- Checkpoint frequency: every 10 updates
- Validation frequency: every 10 updates
- Retained checkpoints: 1
- Method: log-det consequence diversity, archive enabled, `beta=0.3`

## Summary Tracker

| ID | Experiment | Priority | Status | Main output |
|---|---|---:|---:|---|
| E01 | Environment characterization | Essential | [ ] | Benchmark statistics and oracle limits |
| E02 | Base Qwen3-4B evaluation | Essential | [x] | Pre-training validity and coverage |
| E03 | Matched validity-GRPO | Essential | [ ] | Clean `beta=0` control |
| E04 | Full CD-GRPO | Essential | [~] | Main proposed-method model |
| E05 | Final paired model evaluation | Essential | [ ] | Main result table |
| E06 | Archive ablation | High | [ ] | Effect of cross-step novelty |
| E07 | Count-versus-log-det ablation | High | [ ] | Effect of consequence geometry |
| E08 | Diversity-weight sweep | Medium | [ ] | Validity-coverage frontier |
| E09 | Held-out `M=12` evaluation | Essential | [ ] | Cardinality interpolation |
| E10 | Inference diversity baselines | High | [ ] | Training versus decoding comparison |
| E11 | Multi-answer RLVR baseline | Medium | [ ] | Related learned-set baseline |
| E12 | Closed-loop scientific utility | Essential | [ ] | Downstream identification result |
| E13 | Training-seed robustness | High | [ ] | Uncertainty across runs |
| E14 | Qualitative case studies | Essential | [ ] | Interpretable examples |

## E01: Environment Characterization

**Question:** Does the Boolean Causal Micro-Lab provide controlled variation in
ambiguity and semantic separation?

**Reasoning:** The benchmark contribution must be established independently of
model performance. `M`, evidence depth, and separation should be shown to
measure related but non-identical properties.

**Run:**

- Enumerate states for `M in {4, 8, 12, 16}`.
- Summarize evidence depth, separation, family composition, and mode counts.
- Compute oracle achievable coverage for each `K` and `M`.

**Outputs:**

- State counts by `M` and separation bucket
- Evidence-depth distribution
- Mean and maximum pairwise consequence distance
- Mechanism-family composition
- Oracle exact and budget-normalized coverage

**Plots:**

- Violin plot of separation by `M`
- Evidence depth versus `M`
- Separation versus evidence depth
- Heatmap of support by `M` and separation bucket

## E02: Base Qwen3-4B Evaluation

**Question:** What validity and diversity are available before training?

**Reasoning:** Establishes whether RL is needed and identifies format,
reasoning, and response-length failure modes.

**Run:**

- Evaluate the base model on the canonical held-out set.
- Generate `K=16` completions per state.
- Derive `K=4` and `K=8` from the same ordered completions.

**Outputs:**

- Nonempty, parse-valid, syntax-valid, and evidence-valid rates
- Pass@4, pass@8, and pass@16
- Exact and budget-normalized coverage
- Effective mode count and dominant-mode mass
- Duplicate valid-mode rate
- Response-length cap rate

**Plots:**

- Validity funnel
- Coverage@K by `M`
- Response-length distribution
- Mode-frequency concentration

## E03: Matched Validity-GRPO

**Question:** Does validity-only GRPO improve validity without improving
semantic coverage?

**Reasoning:** This is the main control for CD-GRPO. It must match group size,
initialization, data, token budget, and compute.

**Configuration:**

```text
beta = 0
8 states per update
16 rollouts per state
100 updates
same Qwen3-4B base checkpoint
same data order and decoding settings as E04
```

**Expected testable hypothesis:** Validity and pass@K increase more strongly
than exact coverage and effective mode count.

**Outputs and plots:**

- Validity, coverage, dominant-mode mass, and effective mode count by step
- Final canonical evaluation
- Validity-coverage trajectory
- Generated tokens, GPU-hours, and cost

## E04: Full CD-GRPO

**Question:** Does consequence-diversity credit increase semantic coverage
without materially reducing validity?

**Reasoning:** This is the proposed method.

**Configuration:**

```text
variant = logdet
archive = on
beta = 0.3
8 states per update
16 rollouts per state
100 updates
```

**Expected testable hypothesis:** CD-GRPO reaches higher exact coverage,
effective mode count, and pairwise consequence distance than E03 at similar
validity.

**Outputs:**

- Validity and coverage by training step
- Diversity-signal activation rate
- Pairwise consequence distance
- Archive size, novelty rate, and scaling
- Diversity contribution relative to validity advantage
- Final canonical evaluation

**Plots:**

- GRPO versus CD-GRPO training curves
- Diversity activation and archive novelty over training
- Pairwise consequence distance over training
- Validity-coverage trajectory

## E05: Final Paired Model Evaluation

**Question:** At equal inference compute, which model returns the best
hypothesis set?

**Models:**

- Base Qwen3-4B
- Matched validity-GRPO from E03
- CD-GRPO from E04

**Protocol:**

- Use identical canonical states and generation seeds.
- Generate `K=16` once per state.
- Calculate prefix metrics at `K=4` and `K=8`.
- Use paired state-level bootstrap confidence intervals.

**Primary endpoint:** Budget-normalized exact coverage@16.

**Secondary endpoints:**

- Evidence-valid completion rate
- Exact coverage@4 and coverage@8
- Pass@K
- Effective mode count
- Dominant-mode mass
- Duplicate valid-mode rate
- Family coverage

**Plots:**

- Coverage@K grouped by model
- Coverage by `M`
- Coverage by separation bucket
- Validity-coverage scatterplot
- Mode-frequency heatmaps

## E06: Archive Ablation

**Question:** Is within-group diversity sufficient, or is cross-step novelty
necessary?

**Arms:**

```text
logdet, archive off
logdet, archive on
```

**Reasoning:** Without the archive, the policy may produce diversity within a
batch while repeatedly returning the same small set across updates.

**Outputs and plots:**

- Coverage over training
- Dominant-mode mass
- Cumulative unique behaviors
- Archive novelty rate
- Final paired coverage difference

## E07: Count Versus Log-Det Ablation

**Question:** Does consequence geometry help beyond penalizing exact
duplicates?

**Arms:**

```text
count novelty with archive
logdet consequence diversity with archive
```

**Reasoning:** Count-based novelty treats all distinct behaviors equally.
Log-det should favor sets that span meaningfully different predictions.

**Outputs and plots:**

- Exact coverage by separation bucket
- Pairwise consequence distance
- Unique valid-mode count
- Validity-coverage frontier

## E08: Diversity-Weight Sweep

**Question:** How does diversity pressure trade off against validity?

**Arms:**

```text
beta in {0.0, 0.1, 0.3, 0.6}
```

**Reasoning:** Establishes that the selected coefficient is not arbitrary and
tests whether excessive diversity pressure damages validity.

**Outputs and plots:**

- Coverage versus beta
- Validity versus beta
- Validity-coverage Pareto frontier
- Beta-guard activation
- Diversity contribution magnitude

Shorter, equal-compute runs are acceptable for this sweep.

## E09: Held-Out M=12 Interpolation

**Question:** Does training on `M in {4,8,16}` generalize to an unseen
ambiguity cardinality?

**Run:**

- Generate 128 held-out `M=12` evaluation states.
- Do not add `M=12` to training.
- Evaluate the same models and seeds as E05.

**Reasoning:** Tests whether the method learns a cardinality-independent
diversity strategy.

**Outputs and plots:**

- Coverage, validity, and effective mode count versus `M`
- Highlight `M=12` as unseen during training
- Coverage by separation within `M=12`

## E10: Inference Diversity Baselines

**Question:** Can decoding changes recover the same coverage without
diversity-aware training?

**Arms:**

- Standard-temperature independent sampling
- Higher-temperature independent sampling
- Verbalized-diversity prompting
- Generate-many-then-select using consequence distance

**Fairness:** Match generated-token budget and returned-set size.

**Outputs and plots:**

- Coverage versus generation budget
- Coverage per generated token
- Validity-coverage frontier
- Duplicate rate by method

## E11: Multi-Answer RLVR

**Question:** How does independent consequence-diversity training compare with
training one trajectory to emit several answers?

**Run:**

- Follow Puri et al.'s Multi-Answer RLVR setup as closely as possible.
- Begin with `K=4`.
- Match total generated training tokens and returned-set size.

**Outputs and plots:**

- Set validity and exact coverage
- Within-set duplicate rate
- Coverage per generated token
- Comparison with independent CD-GRPO sampling

This experiment is secondary if implementation or compute is constrained.

## E12: Closed-Loop Scientific Utility

**Question:** Does broader hypothesis coverage improve experiment selection and
hidden-world identification?

**Protocol:**

1. Generate 16 hypotheses.
2. Discard invalid outputs.
3. Choose the unobserved experiment with maximum empirical outcome entropy.
4. Execute it in the hidden world.
5. Add the result to evidence and repeat.

Use the same planner for every generator.

**Models:**

- Base Qwen3-4B
- Validity-GRPO
- CD-GRPO
- Oracle version-space planner

**Outputs:**

- Hidden-world identification success after `T` experiments
- Remaining version-space size
- Information gain per experiment
- Experiments to singleton
- Oracle regret

**Plots:**

- Identification success versus experiment count
- Remaining version-space size versus experiment count
- Cumulative oracle regret
- Coverage at each evidence stage

## E13: Training-Seed Robustness

**Question:** Are the main differences stable across optimization randomness?

**Run:**

- At least two, ideally three, seeds for E03 and E04.
- Keep all non-seed settings fixed.

**Outputs and plots:**

- Mean training curves with seed ranges
- Per-seed final metrics
- Paired bootstrap intervals over evaluation states
- Across-seed uncertainty for the primary endpoint

## E14: Qualitative Case Studies

**Question:** What does useful semantic diversity look like in concrete states?

**Select:**

- One low-separation state
- One medium-separation state
- One high-separation state

**Show:**

- Visible evidence
- Available valid modes
- GRPO-generated hypotheses
- CD-GRPO-generated hypotheses
- Predicted outcomes on informative unobserved experiments
- Experiment selected by the fixed planner

**Plots:**

- Consequence-signature matrix
- Hypothesis-to-mode mapping
- Selected experiment and predicted outcome partition

## Main Thesis Tables

1. Benchmark statistics and oracle limits
2. Base, GRPO, and CD-GRPO aggregate evaluation
3. Coverage@K by `M`
4. Coverage by separation bucket
5. Method ablations
6. Closed-loop scientific utility
7. Compute, token, GPU-hour, and cost comparison

## Main Thesis Figures

1. Environment separation and evidence-depth characterization
2. Validity and coverage over training
3. Validity-coverage frontier
4. Coverage@K by model and `M`
5. Mode-frequency heatmaps
6. Coverage improvement by separation bucket
7. Closed-loop identification curves
8. Qualitative consequence-signature example

## Execution Order

1. Finish E04.
2. Run E03 with exactly matched compute.
3. Generate the E09 `M=12` evaluation slice.
4. Run E05 for base, GRPO, and CD-GRPO.
5. Produce the main tables and plots.
6. Run E06 and E07.
7. Implement and run E12.
8. Add E10.
9. Add E08 and E13 as budget permits.
10. Run E11 if time and compute remain.
11. Select and render E14 case studies.

## Run Record Template

Copy this block for every training or evaluation run:

```text
Experiment ID:
Date:
Git commit:
Config:
Model initialization:
Dataset manifest:
Seed:
Hardware:
W&B URL:
Artifact path:
Checkpoint:
Generated tokens:
GPU-hours:
Estimated cost:
Completion status:
Key metrics:
Notes or failures:
```
