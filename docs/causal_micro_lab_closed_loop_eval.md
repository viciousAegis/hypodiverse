# Causal Micro-Lab Closed-Loop Evaluation

## Purpose

The one-step evaluation measures whether a generator produces a valid and
semantically diverse hypothesis bank at a frozen evidence state. The
closed-loop evaluation asks the downstream question:

> Does broader valid hypothesis coverage lead to more informative experiments
> and faster identification of the hidden causal world?

The evaluation should isolate the effect of hypothesis generation. The primary
protocol therefore uses each trained model only for hypothesis generation and
uses the same deterministic experiment planner, simulator, and evidence
updater for every model.

## Closed-Loop Process

At experiment step \(t\), the loop is

\[
E_t
\rightarrow \text{hypothesis generator}
\rightarrow H_t
\rightarrow \text{fixed planner}
\rightarrow a_t
\rightarrow \text{hidden world}
\rightarrow o_t
\rightarrow E_{t+1},
\]

where:

- \(E_t\) is the currently visible evidence;
- \(H_t\) is the generated hypothesis bank;
- \(a_t\) is the selected experiment; and
- \(o_t\) is the outcome returned by the hidden world.

The loop terminates when the exact version space becomes a singleton, the
experiment budget is exhausted, or no unobserved experiments remain.

## Primary Controlled Protocol

For every held-out initial state and model:

1. Render a prompt containing only the evidence currently visible in \(E_t\).
2. Generate \(K\) independent hypotheses with the model being evaluated.
3. Parse and verify every completion against \(E_t\).
4. Discard invalid hypotheses.
5. Preserve duplicate valid hypotheses in the planner input.
6. Simulate each valid hypothesis under every unobserved experiment.
7. Compute empirical outcome entropy for each experiment.
8. Select the experiment with maximum entropy.
9. Break entropy ties by the lowest experiment ID.
10. Execute the selected experiment in the hidden true mode.
11. Append the resulting observation to the visible evidence.
12. Recompute the private exact version space for evaluation.
13. Repeat from the updated prompt.

Each step uses a fresh bank of eight generations. The primary protocol does
not carry an archive of earlier hypotheses forward: after new evidence arrives,
old hypotheses may no longer be valid, and archive maintenance would introduce
an additional algorithmic choice that differs across models.

The planner selects

\[
a_t
=
\arg\max_{a\notin E_t}
H\left(
    \left\{
        \sigma_a(h):h\in H_t^{\mathrm{valid}}
    \right\}
\right),
\]

where \(\sigma_a(h)\) is the three-bit outcome predicted by hypothesis \(h\)
under experiment \(a\).

### Why Duplicates Remain

Duplicate hypotheses should not be removed in the primary protocol. If the
generator repeatedly samples one mode, that mode receives greater empirical
mass in the prediction distribution. This can reduce outcome entropy and
produce a less informative experiment. The resulting loss of scientific
utility is precisely the consequence that the evaluation is intended to
measure.

A secondary diagnostic may deduplicate valid modes before planning. Comparing
the two variants separates:

- support coverage: which modes were reached at least once; and
- distribution quality: how generation probability is allocated among them.

## Compared Generators

Use the same held-out states and inference settings for all available
conditions:

1. base Qwen3-4B;
2. GRPO;
3. LIFPO.

All trained checkpoints must use the same output schema, parser, verifier,
temperature, maximum response length, and hypothesis-bank size.

## Planner Conditions

### Fixed Generated-Hypothesis Planner

This is the primary planner. It computes empirical entropy from the hypotheses
generated at the current evidence state. It is deterministic, uses no private
answer information, and is identical across generators.

### Oracle Version-Space Planner

This is an upper-bound reference. It computes experiment entropy using every
mode in the private true version space \(V(E_t)\). The oracle is never used to
generate model-facing information.

### Random Planner

This is a lower-bound reference. It selects uniformly from unobserved
experiments using a fixed seed. The same random choices should be coupled
across generator conditions whenever their evidence trajectories permit it.

### Synthetic Generator References

In addition to the oracle and random-experiment planners, report three
references that isolate the quality of the generated bank:

- sample \(K\) valid modes uniformly without replacement;
- sample \(K\) valid modes uniformly with replacement; and
- sample one valid mode and repeat it \(K\) times.

## Failure Rules

Failure behaviour must be fixed before evaluation.

- If no completion is valid, select an unobserved experiment using the seeded
  random fallback planner.
- If only one valid mode is generated, every experiment has zero empirical
  entropy; use the standard lowest-ID tie-break.
- If a model request fails, record the request error and treat the bank as
  containing no valid hypotheses.
- If no unobserved experiment remains, terminate the trajectory.

Do not use the oracle planner as a fallback for model failures.

## Evaluation States

Use hidden worlds held out from all training data. All methods must start from
the same initial evidence states and hidden modes.

Recommended primary configuration:

| Setting | Value |
|---|---:|
| Held-out trajectories | 128: 64 at \(M_0=16\), 64 at \(M_0=32\) |
| Hypotheses per step \(K\) | 8 |
| Maximum experiments \(T\) | 8 |
| Early stopping | Singleton version space |
| Planner | Fixed empirical-entropy planner |

The 128 trajectories should be balanced as closely as possible across:

- initial valid-mode count \(M\);
- predictive-separation tertile.

At worst, one model requires

\[
128 \times 8 \times 8 = 8{,}192
\]

hypothesis generations. Early identification reduces the realised total.
Trajectories are sequential within a hidden world but can run concurrently
across worlds.

Results for different \(K\) values cannot be extracted from one closed-loop
trajectory after the first step because different bank sizes may select
different experiments and therefore produce different future evidence. The
primary budget is \(K=8\); any other \(K\) requires separate,
compute-accounted trajectories.

## Metrics

### Primary Metrics

- hidden-world identification success after each experiment;
- mean and median remaining version-space size;
- mean \(\log_2 |V(E_t)|\);
- experiments required to reach a singleton;
- mean capped experiments used, where failures consume all \(T\) experiments;
- identifications per 100 experiment turns;
- area under the remaining-version-space curve;
- cumulative information gain.

Mean capped experiments used measures expected experiment cost per attempted
world. Identifications per 100 experiments measures successful discoveries per
unit of scientific interaction:

\[
100\frac{\#\text{identified worlds}}
{\sum_i \left(t_i\mathbf{1}[\text{success}_i]
+T\mathbf{1}[\text{failure}_i]\right)}.
\]

### Hypothesis-Bank Diagnostics

At every evidence stage, report:

- parse validity;
- syntax validity;
- evidence consistency;
- number of distinct valid hypotheses;
- budget-normalised exact coverage;
- duplicate rate;
- effective mode count;
- generated-mode separation;
- empirical prediction entropy of the selected experiment.

### Oracle Regret

At the model trajectory's current state, define per-step entropy regret as

\[
\mathcal{R}_t
=
\max_{a\notin E_t} H_{V(E_t)}(a)
-
H_{V(E_t)}(a_t),
\]

where the private version space is used only for evaluation. Report both
per-step and cumulative regret.

Also compare identification success and remaining version-space size directly
against the oracle trajectory after the same experiment budget.

## Statistical Comparisons

Use paired comparisons because every generator is evaluated on the same hidden
worlds.

- Report means with bootstrap confidence intervals over hidden worlds.
- Plot identification success as a function of experiment count.
- Plot mean \(\log_2 |V(E_t)|\) with confidence bands.
- Report paired differences from GRPO.
- Report all primary curves separately for \(M_0=16\) and \(M_0=32\).
- Use current \(\log_2|V(E_t)|\) directly; do not introduce stage bins.
- Slice secondary diagnostics by predictive-separation tertile.
- State the number of supported trajectories in every slice.

If only one trained seed is available, bootstrap uncertainty describes
evaluation-state variability, not training-seed variability. This limitation
must be stated explicitly.

## Recommended Figures

1. **Closed-loop schematic:** evidence, generated hypotheses, fixed planner,
   experiment, observation, and evidence update.
2. **Identification curve:** fraction of hidden worlds identified versus
   experiment count.
3. **Version-space curve:** mean \(\log_2 |V(E_t)|\) versus experiment count.
4. **Oracle-regret curve:** cumulative entropy regret versus experiment count.
5. **Coverage-utility relationship:** initial or per-step hypothesis coverage
   versus subsequent information gain.
6. **Qualitative trajectory:** one evidence state showing generated modes,
   predicted outcomes, selected experiment, observation, and version-space
   reduction.

## Implementation

The model-backed runner reuses the exact parser, verifier, simulator, and
version-space engine used by frozen evaluation. It:

1. load private held-out `EvidenceState` records;
2. maintain one mutable trajectory state per hidden world;
3. call the existing OpenAI-compatible evaluation backend;
4. reuse the existing prompt renderer, parser, verifier, and simulator;
5. parallelise across trajectories while preserving sequential steps within
   each trajectory;
6. write one append-only JSONL trace record per trajectory step;
7. support restart from completed trace records;
8. write aggregate summaries and plots;
9. log live progress and closed-loop metrics to W&B; and
10. store enough metadata to reproduce every selected experiment.

### Trace Schema

Each step record should include:

- run and model identifiers;
- initial state ID and hidden mode ID;
- trajectory and experiment-step indices;
- visible evidence before selection;
- raw completions and thinking text;
- parser and verifier outputs;
- valid and unique mode IDs;
- predicted outcome distribution per available experiment;
- selected experiment and its empirical entropy;
- observed hidden-world outcome;
- version-space size before and after the experiment;
- information gain;
- oracle-selected experiment;
- oracle entropy and selected-experiment true entropy;
- entropy regret;
- request timings, token counts, and errors.

Private fields such as the hidden mode and full version space must remain in
the trace and evaluator only; they must never appear in model prompts.

## Cluster Invocation

The Slurm launcher accepts one of the three public model conditions and uses
the released Hugging Face checkpoints:

```bash
sbatch scripts/cluster/sbatch_causal_micro_lab_closed_loop_eval.slurm base
sbatch scripts/cluster/sbatch_causal_micro_lab_closed_loop_eval.slurm grpo
sbatch scripts/cluster/sbatch_causal_micro_lab_closed_loop_eval.slurm lifpo
```

## Interpretation

The strongest supported claim would be:

> Under an identical fixed experiment-selection rule, a generator with broader
> valid semantic coverage identifies hidden causal worlds using fewer
> experiments.

This connects output diversity to measurable scientific utility without
claiming that the micro-lab reproduces open-ended real-world discovery.

Failure to improve closed-loop utility would also be informative. It would
show that higher offline coverage does not automatically yield better
experiments, or that the generated differences occur on consequences that are
not useful along the realised identification trajectories.
