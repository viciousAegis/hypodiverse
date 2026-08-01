# Geometry-aware Causal Micro-Lab v3 evaluation

## Purpose

`final_v3` keeps the Boolean Causal Micro-Lab task, verifier, prompts, and
trained checkpoints fixed. It changes the diversity analysis from counting
only distinct semantic modes to measuring how well the generated modes
represent the complete valid hypothesis space.

For evidence state `E`, the distance between valid modes `h` and `h'` is the
fraction of unobserved experiments for which their complete deterministic
outcomes `(Z1, Z2, Y)` differ:

```text
d_E(h, h') = mean_{q not in E} 1[f_h(q) != f_h'(q)]
```

For a generated valid set `S`, its representation error and predictive
coverage AUC are:

```text
Q_E(S) = mean_{h in V(E)} min_{s in S} d_E(h, s)
AUC_E(S) = 1 - Q_E(S)
```

The report keeps four questions separate:

1. Did the model produce any valid answer? (`pass_at_k`)
2. How many valid modes did it recover? (`num_unique_valid_modes`)
3. How much of the predictive space do those modes represent?
   (`predictive_coverage_auc`)
4. Given the same number of recovered modes, did it choose good
   representatives? (`predictive_placement_regret`, lower is better)

Placement regret compares the generated set with the exact best subset of the
same cardinality. It therefore does not conflate finding more modes with
placing those modes well in the hypothesis space.

## Frozen evaluation set

The committed set is under:

```text
eval_sets/causal_micro_lab/final_v3/
```

It contains 192 held-out states, 48 each for `M={4,8,12,16}`. Selection is
continuous over the empirical representative-coverage opportunity within each
`M`; no low/medium/high labels are used. `manifest.json` records hashes,
selection ranges, and the overlap audit against all existing train,
validation, debug, and legacy evaluation rows.

Rebuild and audit it on CPU with:

```bash
PYTHONPATH=src python scripts/build_causal_micro_lab_final_eval_v3.py
PYTHONPATH=src python scripts/audit_causal_micro_lab_representative_coverage.py
```

The reference-policy audit must report `"rankability_passed": true`. Its
current K=4 ordering is collapsed < concentrated sampling < uniform sampling <
uniform distinct < pairwise-distance oracle < representative oracle.

## Cluster preflight

From the cluster repository checkout:

```bash
git pull
test -f eval_sets/causal_micro_lab/final_v3/verl_test.jsonl
bash -n scripts/cluster/sbatch_causal_micro_lab_v3_base_eval.slurm
bash -n scripts/cluster/sbatch_causal_micro_lab_v3_checkpoint_eval.slurm
bash -n scripts/cluster/submit_causal_micro_lab_v3_evals.sh
```

The checkpoint roots expected by the frozen comparison are:

```text
causal_micro_lab_cluster_validity_grpo_r1/global_step_90
causal_micro_lab_cluster_ips_grpo_v1_eps02_r1/global_step_60
causal_micro_lab_cluster_latent_ips_grpo_v2_fulltraj_k8_r1/global_step_55
```

The checkpoint launcher verifies FSDP shards, merges them with
`python -m verl.model_merger merge --backend fsdp`, and reuses an existing
complete merge.

## Submit evaluations

Submit all four independent one-GPU jobs:

```bash
bash scripts/cluster/submit_causal_micro_lab_v3_evals.sh
```

Equivalent individual commands are:

```bash
sbatch scripts/cluster/sbatch_causal_micro_lab_v3_base_eval.slurm

sbatch scripts/cluster/sbatch_causal_micro_lab_v3_checkpoint_eval.slurm \
  causal_micro_lab_cluster_validity_grpo_r1 standard 90

sbatch scripts/cluster/sbatch_causal_micro_lab_v3_checkpoint_eval.slurm \
  causal_micro_lab_cluster_ips_grpo_v1_eps02_r1 standard 60

sbatch scripts/cluster/sbatch_causal_micro_lab_v3_checkpoint_eval.slurm \
  causal_micro_lab_cluster_latent_ips_grpo_v2_fulltraj_k8_r1 latent 55
```

Every job uses one A100, 128 request workers, SGLang static memory fraction
0.82, a 6000-token primary response, thinking enabled, and the deterministic
256-token non-thinking fallback. It samples one K=16 bank per state and derives
K=4,8,12 from stable prefixes. W&B live logging is enabled under project
`scattered-discovery`.

Expected W&B and artifact run names are:

```text
causal_micro_lab_final_v3_k16_qwen3_4b_base
causal_micro_lab_cluster_validity_grpo_r1_global_step_90_final_v3_k16
causal_micro_lab_cluster_ips_grpo_v1_eps02_r1_global_step_60_final_v3_k16
causal_micro_lab_cluster_latent_ips_grpo_v2_fulltraj_k8_r1_global_step_55_final_v3_k16
```

## Compare completed runs

After all jobs complete:

```bash
ROOT=artifacts/causal_micro_lab_final_eval_v3

PYTHONPATH=src python scripts/analyze_causal_micro_lab_v3_results.py \
  --report base="$ROOT/causal_micro_lab_final_v3_k16_qwen3_4b_base/latest/report" \
  --report validity="$ROOT/causal_micro_lab_cluster_validity_grpo_r1_global_step_90_final_v3_k16/latest/report" \
  --report ips="$ROOT/causal_micro_lab_cluster_ips_grpo_v1_eps02_r1_global_step_60_final_v3_k16/latest/report" \
  --report latent="$ROOT/causal_micro_lab_cluster_latent_ips_grpo_v2_fulltraj_k8_r1_global_step_55_final_v3_k16/latest/report" \
  --baseline validity
```

The analysis refuses to compare reports with different state/K support. It
writes:

```text
artifacts/causal_micro_lab_final_eval_v3/comparison/headline_by_k.csv
artifacts/causal_micro_lab_final_eval_v3/comparison/paired_differences.csv
artifacts/causal_micro_lab_final_eval_v3/comparison/performance_by_opportunity_state.csv
artifacts/causal_micro_lab_final_eval_v3/comparison/performance_by_opportunity_bin.csv
docs/figures/causal_micro_lab_v3/method_comparison_by_k.png
```

Per-run report directories also contain state-level predictive coverage curves
and aggregate curves by K and M. Diversity metrics are reported conditionally
on at least one valid mode; invalid generations are handled separately by
pass@K and valid-mode rate.
