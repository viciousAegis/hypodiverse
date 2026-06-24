# Scattered Discovery Environment

The scattered discovery environment is a synthetic active-discovery task for studying whether RL training makes a model collapse onto one answer or preserve a distribution over multiple valid answers. It is intentionally smaller and more controlled than HypoSpace: we know the full hidden world, can generate many worlds cheaply, and can independently vary the number of valid hypotheses, their overlap, evidence noise, and interaction budget.

The current implementation is `env_type: scattered_causal`. It lives in:

```text
src/scattered_discovery/envs/scattered_world.py
src/scattered_discovery/envs/scattered_causal.py
src/scattered_discovery/envs/scattered_dsl.py
src/scattered_discovery/envs/scattered_evidence.py
```

The veRL dataset rows store the full environment instance as `env_spec_json`, so the same generated tasks can be used for training, validation, local evaluation, and cluster evaluation.

## What The Task Is

Each world contains a hidden directed causal graph made from several true branches. A branch is currently a directed path:

```text
x00 -> x01 -> x02 -> ...
```

There are several such branches in the same world. Some branches may share an early prefix, depending on the dispersion setting. The model does not see the hidden graph. It interacts with the environment using a small DSL:

```text
ACTION: INTERVENE x00
ACTION: TEST edge(x00,x01)
ACTION: COMMIT path(x00,x01,x02)
ACTION: COMMIT [path(...); path(...)]
```

The model starts with only the initial measurable variables. Intervening on a known variable reveals downstream candidate variables and noisy measurements for each outgoing candidate edge. Testing an edge collects noisy evidence for that edge. Committing ends the episode and scores the final hypothesis or final set.

The model-facing problem is therefore:

1. Explore the world under a fixed experiment budget.
2. Gather enough evidence to support a final answer.
3. Commit a terminal hypothesis, or a set of terminal hypotheses depending on protocol.

For the current path task, a terminal hypothesis is a complete true branch path. Intermediate edges or partial paths may be true, but they are not final answers.

## Why This Environment Exists

The environment is designed to separate several things that are confounded in real benchmarks:

- Hypothesis correctness: whether a committed object is actually in the hidden world.
- Evidence support: whether the model gathered evidence for the committed object during the episode.
- Answer diversity: whether a rollout group covers one or many distinct valid terminal hypotheses.
- Syntactic control: whether failures are due to malformed actions rather than reasoning.
- Search pressure: how much budget is available relative to graph size and distractors.

This makes it useful for GRPO-style experiments because the terminal reward can be made clean and sparse, while the world still has multiple correct answers. We can ask whether standard single-answer GRPO collapses toward one branch and whether set protocols, diversity rewards, or world-modeling losses recover broader coverage.

## Current Path-Based World Generator

`WorldGenerator` currently constructs worlds from paths. The important controls are:

```yaml
world:
  num_branches: 3
  branch_depth: 2
  distractors_per_node: 1
  base_budget: 10
  test_cost: 1
  intervene_cost: 2
```

`num_branches` controls how many terminal answers exist. If there are three branches, there are three true terminal paths.

`branch_depth` controls path length in edges. A depth of 2 means each terminal path has 3 nodes.

`distractors_per_node` controls how many false outgoing candidates are attached to true parent nodes. This increases the branching factor and makes intervention observations less trivial.

`base_budget` controls the number of experimental actions available. `INTERVENE`, `TEST`, and invalid actions have configurable costs.

The hidden graph is generated as:

1. Choose a shared prefix length from the dispersion value.
2. Create `num_branches` directed paths with that shared prefix.
3. Add all true path edges to the hidden graph.
4. Add false distractor targets to parent nodes.
5. Shuffle outgoing candidates so the answer is not position-coded.

The model sees only measurements, known variables, and evidence summaries. It does not see branch ids, target paths, version-space counts, or accept/reject labels unless debugging flags explicitly expose them.

## How Diversity Is Controlled

Diversity is controlled primarily through `dispersion`.

`dispersion` is a number between `0.0` and `1.0` that controls how much early structure the true branches share.

Low dispersion means branches share more prefix nodes:

```text
x00 -> x01 -> x02
         |-> x03
         |-> x04
```

High dispersion means branches diverge earlier, up to fully separate roots:

```text
x00 -> x01 -> x02
x03 -> x04 -> x05
x06 -> x07 -> x08
```

In code, the shared prefix length is approximately:

```text
round((1 - dispersion) * branch_depth)
```

with dispersion near `1.0` forcing no shared prefix.

This matters because diversity can be controlled without changing the number of correct answers. For example, two worlds can both have `num_branches: 4`, but:

- At low dispersion, one early intervention may put the model near several valid terminal paths.
- At high dispersion, covering multiple answers requires exploring multiple separate starts.

So `dispersion` controls overlap among valid hypotheses, not just graph size. That is the main knob for diversity pressure.

Dataset configs usually balance this explicitly:

```yaml
dispersion_values: [0.0, 0.25, 0.5, 0.75, 1.0]
```

The dataset builder assigns these values round-robin across generated rows. That gives balanced train and validation files rather than separate single-dispersion files.

## Other Difficulty Controls

Diversity is not only dispersion. Current dataset YAMLs also support `world_values`, which sweep structural difficulty:

```yaml
world_values:
  num_branches: [3, 4, 5]
  branch_depth: [2, 3]
  distractors_per_node: [1, 2]
base_budget_from_branch_depth_overhead: 2
```

These are not conflicts with `task.world`. The `task.world` block sets fixed defaults, and `world_values` overrides selected fields per generated row. `base_budget_from_branch_depth_overhead: 2` then ties budget to depth with `base_budget = 2 * branch_depth + 2`. This lets a single train or validation file contain a balanced mixture of graph sizes, budgets, and dispersion values.

The main difficulty knobs are:

- More branches: more valid terminal answers and more possible diversity.
- Greater depth: longer final hypotheses and more evidence required.
- More distractors: more false edges mixed into interventions.
- More noise: more repeated tests needed before evidence is reliable.
- Lower budget: greater exploration pressure.
- Higher dispersion: less shared evidence across answers.

## Evidence And Correctness

The environment uses noisy Gaussian measurements. True claims sample around `true_mean`; false claims sample around `false_mean`; `noise_sigma` controls overlap.

Evidence is accumulated as log odds. The model sees measurement summaries, not the hidden truth.

A committed final path is evidence-backed only when every adjacent edge in that path has accepted evidence from the episode. The parser may still accept diagnostic path tests, but a path-level test is not sufficient for final credit in the current pilot setup. This keeps the interactive contract realistic: the agent must discover and support the causal chain edge by edge, rather than query the whole final answer directly.

A committed hypothesis is only valid if all of these are true:

- It is syntactically parseable.
- It is true in the hidden world.
- It is terminal for the current task.
- It is evidence-backed from the episode.
- It is not a duplicate if set mode is being evaluated for unique coverage.

This distinction is important. An intermediate true edge is not "wrong" as a causal statement, but it is incorrect as a final answer because the task asks for complete terminal discoveries. This mirrors real settings where partial results can be scientifically useful but are not the requested final object.

## Reward Profiles

The clean baseline uses `reward_profile: terminal_only`.

In that profile, the important learning signal is terminal validity: reward comes from valid final hypotheses. This is the cleanest setup for testing GRPO behavior because it does not give much shaped reward for well-formed but wrong behavior.

The practical scattered GRPO baseline uses
`reward_profile: terminal_clean_invalid_bonus`. It keeps valid final hypotheses
at reward `1.0`, but gives reward `0.2` for a parseable final commit that is
invalid after an otherwise clean rollout. This should help action discipline
without directly rewarding coverage.

There is also support for shaped reward profiles, where format, admissibility, false commits, unsupported commits, duplicates, and budget can contribute to the reward. Those are practical guardrails when small models struggle to use the DSL, but they make the experimental story less clean.

For collapse/diversity experiments, use terminal-only first.

## Single Protocol Versus Set Protocol

The same environment supports two final-answer protocols.

`protocol: single` means the model must commit exactly one hypothesis:

```text
ACTION: COMMIT path(x00,x01,x02)
```

This is the vanilla single-rollout setting. If the world has many valid terminal paths, one rollout can still only get credit for one of them. Diversity is then measured across rollout groups, pass@K samples, or repeated samples from the same prompt.

`protocol: set` means the model may commit a set:

```text
ACTION: COMMIT [path(x00,x01,x02); path(x00,x03,x04)]
```

This turns the episode into a coverage problem. The reward vector records which branch ids were recovered, and aggregate metrics can measure coverage across the final set.

For vanilla GRPO comparison, keep the veRL objective unchanged and vary the environment protocol/reward surface. The environment owns the final reward semantics; veRL still optimizes action tokens with GRPO.

## What "Path-Based" Means

The current implementation uses paths as the terminal motif because they are the simplest nontrivial case:

- A full answer has multiple parts.
- Partial true claims exist.
- Multiple final answers can overlap.
- Diversity can be controlled by prefix sharing.
- Evidence can be gathered at edge or path level.

This gives us a controlled first test of hypothesis diversity. It is not meant to claim that real scientific hypotheses are always paths. It is a scaffold: useful because it lets us know exactly what the answer space is and how answer overlap changes.

The DSL already contains expression types beyond paths:

```text
edge(x00,x01)
path(x00,x01,x02)
fork(x00,[x01,x02])
collider([x00,x01],x02)
```

At present, `edge`, `fork`, and `collider` can be parsed and classified as true or false motifs, but the generated terminal targets are full paths. That means forks and colliders are currently not the primary final-answer target family.

## Extensions To Other Motifs

The natural generalization is to make the terminal target type configurable.

Instead of every terminal answer being a path, a world could define terminal motifs such as:

```text
fork(A,[B,C])
collider([A,B],C)
chain(A,B,C)
diamond(A,B,C,D)
backdoor(A,Z,B)
frontdoor(A,M,B)
```

Each motif family needs four pieces:

1. A generator that plants several true motifs in the hidden graph.
2. A distractor generator that creates plausible false motifs.
3. A classifier that maps a DSL expression to true, false, terminal, or intermediate.
4. An evidence policy that says what observations support the motif.

For example, a fork task might ask the model to find complete common-cause motifs:

```text
fork(x00,[x01,x02])
```

Evidence could be accepted if both outgoing edges are supported. Intermediate true edges would again be true but non-terminal.

A collider task might ask for common-effect motifs:

```text
collider([x00,x01],x02)
```

Evidence could require support for both incoming edges and possibly an intervention pattern that distinguishes direct causes from correlated distractors.

A diamond task could require two paths that split and rejoin:

```text
x00 -> x01 -> x03
x00 -> x02 -> x03
```

This would be a stronger test because the model must assemble a compositional structure rather than a single linear branch.

## Preserving Diversity Controls For Motifs

The same diversity idea can transfer beyond paths, but the overlap definition changes.

For paths, dispersion controls shared prefix length.

For forks, dispersion could control shared causes or shared effects:

- Low dispersion: many valid forks share a common cause.
- High dispersion: valid forks use separate causes and effects.

For colliders, dispersion could control shared effects or shared causes:

- Low dispersion: many causes converge on the same effect.
- High dispersion: colliders are disjoint.

For diamonds or other larger motifs, dispersion could control shared subgraphs:

- Low dispersion: motifs share one arm or a central node.
- High dispersion: motifs are disjoint.

The general abstraction is:

```text
dispersion = 1 - normalized overlap among valid terminal hypotheses
```

The implementation should expose this through motif-specific generators while keeping dataset-level controls the same:

```yaml
env_type: scattered_causal
task:
  motif_type: fork
  dispersion: 0.5
  world:
    num_targets: 4
    distractors_per_node: 2
```

We do not have this motif-type config yet. The current code supports the DSL pieces and path terminal generator; motif terminal generation is the next extension.

## Recommended Experimental Use

For clean GRPO baselines:

```yaml
defaults:
  protocol: single
  reward_profile: terminal_only

datasets:
  - env_type: scattered_causal
    dispersion_values: [0.0, 0.25, 0.5, 0.75, 1.0]
```

Use balanced validation files with the same dispersion values. Report metrics by dispersion bucket as well as aggregated means.

For diversity experiments:

- Keep the train file mixed across dispersion values.
- Evaluate single-rollout reward and pass@K coverage.
- Compare `protocol: single` and `protocol: set`.
- Track `valid_unique_count`, `recovery`, `reward_vector`, and duplicate counts.
- Stratify results by graph size and dispersion.

For harder follow-up experiments:

- Increase `num_branches`.
- Increase `branch_depth`.
- Increase `distractors_per_node`.
- Reduce `base_budget`.
- Increase `noise_sigma`.
- Move from path terminals to fork/collider/diamond terminals.

## Current Limitations

The environment is synthetic and deliberately stylized. That is useful for controlled RL experiments, but it is not a replacement for HypoSpace or real-world discovery tasks.

Current limitations:

- Terminal targets are currently path branches.
- Motif expressions are parsed but not yet generated as terminal families.
- Evidence is Gaussian and independent, not a realistic experimental simulator.
- Interventions reveal all outgoing candidates for a known variable, so difficulty must come from budget, noise, distractors, graph size, and dispersion.
- The variable names are abstract `xNN` symbols, not semantic scientific concepts.

These are acceptable for the first experiments because the goal is not realism. The goal is to test how RL objectives behave when there are multiple valid discoveries and controllable overlap among them.

## Useful Commands

Generate the default mixed datasets:

```bash
UV_CACHE_DIR=.uv-cache uv run --extra verl scattered-discovery-make-dataset \
  --config configs/verl/datasets/all_envs.yaml
```

Generate scattered-only smoke data:

```bash
UV_CACHE_DIR=.uv-cache uv run --extra verl scattered-discovery-make-dataset \
  --config configs/verl/datasets/scattered_smoke.yaml
```

Open the interactive viewer:

```bash
UV_CACHE_DIR=.uv-cache uv run scattered-discovery-viewer
```

Generate a graph gallery across dispersion values:

```bash
UV_CACHE_DIR=.uv-cache uv run scattered-discovery-graph-gallery \
  --dispersions 0,0.25,0.5,0.75,1 \
  --samples-per-dispersion 2
```
