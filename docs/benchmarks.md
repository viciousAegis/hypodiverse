# Benchmark Notes

These are candidate environments or datasets for our causal-discovery and hypothesis-diversity work.

## HypoSpace

Source: <https://arxiv.org/pdf/2510.15614> and <https://github.com/CTT-Pavilion/_HypoSpace>

What it does:

- Evaluates set-valued hypothesis generation under underdetermination.
- Domains: causal graphs from perturbation observations, gravity-constrained 3D voxel reconstruction, and Boolean genetic interactions.
- Metrics: Validity, Uniqueness, and Recovery against exactly enumerable admissible sets.
- The GitHub repo provides generators and benchmark runners rather than a hosted Hugging Face dataset.

Why it helps us:

- It directly targets the diversity/coverage issue we care about.
- It gives deterministic validators and exact admissible-set sizes, so coverage is measurable without LLM judging.
- It is a clean comparison point for vanilla GRPO, set-reward GRPO, and future diversity-aware objectives.

Difficulty statistics:

- Difficulty is scaled through admissible-set size `|H_O|`: causal varies node/intervention count, 3D varies view/top-projection settings, and Boolean varies operator/depth/observation coverage.
- The paper reports that causal is easiest at their original scales, 3D is intermediate, and Boolean is most discriminative.
- In the paper's larger-scale checks, `|H_O|=160` for causal and `|H_O|=125` for 3D. GPT-5, Gemini-2.5-Pro, and Grok-4 retain 100% validity, but Recovery drops to roughly 57-72% on causal and 65-79% on 3D.
- Boolean hard settings show the sharpest coverage collapse. The paper reports hard-level Recovery around 48% for GPT-5, 47% for Gemini-2.5-Pro, 36% for DeepSeek-R1/Grok-4, 24% for Claude-Opus-4, 14% for GPT-4o, and 11% for Llama-3.3-70B-Instruct.

Failure points:

- Static HypoSpace is hypothesis selection/generation, not interactive experimental design.
- Small prompts can leak answer-shaped examples if examples overlap with task variables.
- Models can maintain high validity while repeatedly proposing the same simple hypotheses.
- Boolean canonicalization can hide superficial diversity and expose true mechanistic collapse.

Remedies:

- Use held-out generated splits with target-shaped examples removed.
- Report Validity, Uniqueness, Recovery, duplicates, invalids, and unsupported commits together.
- Add complexity-stratified baselines because the paper finds strong simplicity bias.
- For our interactive version, keep version-space counts diagnostic-only, not agent-facing.

## DiscoveryWorld

Source: <https://arxiv.org/abs/2406.06769>

What it does:

- Interactive science-game environment where agents form hypotheses, run experiments, analyze evidence, and draw conclusions.

Why it helps us:

- Strong match for long-horizon scientific-discovery behavior.
- Useful once we want more realistic multi-step interaction beyond enumerable toy hypothesis spaces.

Failure points:

- More complex environment integration.
- Harder to isolate whether improvements come from better exploration, tool use, memory, or language priors.

Remedies:

- Use it after validating methods on HypoSpace/scattered causal.
- Preserve structured logs of actions, observations, and final hypotheses.

## ScienceWorld

Source: <https://arxiv.org/abs/2203.07540>

What it does:

- Text-based interactive science tasks with household/lab-like procedures.

Why it helps us:

- Good generalization test for interactive scientific reasoning.

Failure points:

- Less focused on hypothesis-set diversity.
- Rewards are task-completion oriented.

Remedies:

- Use as a transfer benchmark, not as the core diversity benchmark.

## CausalBench / Causal Discovery Datasets

Sources: <https://arxiv.org/abs/2404.06349> and established causal datasets such as Tuebingen cause-effect pairs and bnlearn networks.

What they do:

- Evaluate causal reasoning or causal structure discovery on fixed datasets.

Why they help us:

- Useful external validity check for causal reasoning.
- Can provide real or semi-real graphs beyond synthetic branches.

Failure points:

- Often single-answer or static.
- May not support interactive interventions or hypothesis-set recovery.

Remedies:

- Wrap them as static eval baselines first.
- Only add interaction where interventions/observations are well-defined.

## Terminal-Bench

Source: <https://arxiv.org/abs/2601.11868>

What it does:

- Evaluates terminal agents on software/CLI tasks.

Why it helps us:

- ECHO was evaluated there, so it is a useful reference for environment-token prediction.

Failure points:

- Far from causal discovery.
- Improvements may reflect shell/tool dynamics rather than scientific hypothesis diversity.

Remedies:

- Use it only to validate ECHO implementation mechanics.
- Keep our primary claim on HypoSpace/scattered causal diversity metrics.
