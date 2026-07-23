# CD-GRPO Implementation Spec (v1)

Consequence-Diversity GRPO on the Boolean Causal Micro-Lab (V0 env spec is a companion document; this spec assumes it). Training framework: **verl**. Model: Qwen3-4B, **thinking mode ON** with a thinking-token budget cap.

---

## 0. One-paragraph summary

Standard GRPO, plus a second advantage term computed only among evidence-consistent (valid) completions in each group. That term rewards a completion in proportion to how much its hypothesis's **predicted consequences on unobserved experiments** differ from those of the other valid completions (set-level, via marginal log-det gain of a kernel over consequence signatures), scaled down for behaviors the policy has already produced on this evidence state in past steps (decayed archive). Invalid, unparseable, or truncated completions never touch the diversity term or the archive.

## 1. Hard rule: oracle separation

Two module boundaries, enforced by code structure (separate packages, no cross-imports):

- **Reward path (train-time)** may use ONLY: the strict parser, the DSL executor (runs a *candidate's own program* on chosen inputs/interventions), and the visible evidence `E` of the state. It must NEVER import or read: the mode table, `private.valid_mode_ids`, `private.hidden_mode_id`, `metadata.valid_mode_count`, or any precomputed signature of any hypothesis other than the sampled completions themselves.
- **Eval/logging path** may use all ground truth (mode table, valid-mode sets) for metrics only. Metrics must not feed back into rewards, sampling, or scheduling.

Any PR that makes the reward path depend on `private.*` or the mode table is wrong by definition.

## 2. Per-completion pipeline (train-time)

For each completion in a group of `G` rollouts on evidence state `E`:

1. **Extract answer**: strip any completed Qwen thinking block and retain the
   final three rule lines. If generation reaches the response-token cap, mark
   `status=TRUNCATED`.
2. **Strict parse** per V0 spec §3 (reject missing/repeated targets, bad operators, bad arity, duplicate binary inputs, unavailable parents, noncanonical input order). Failure → `status=PARSE_FAIL`.
3. **Validity**: execute the parsed hypothesis with the DSL executor on every experiment in `E` (same inputs + intervention); valid iff all predicted `(Z1, Z2, Y)` match the recorded observations. → `status=VALID` or `INVALID`.
4. **Consequence signature** (VALID only): execute the hypothesis on the **probe set** `P` and concatenate predicted `(Z1, Z2, Y)` triples into a bit vector `s`.
   - Default `P` = all experiments in the full 40-experiment space not present in `E`.
   - Estimation-ablation hook: `probe_fraction ρ ∈ (0, 1]`; when `ρ < 1`, `P` is a random subset of size `⌈ρ·|A∖E|⌉`, sampled once per state with a seed derived from `state_id` (fixed across the whole run, so signatures are comparable across steps).
5. **Behavior key** `b = sha256(s)` — computed from the sampled completion's own signature; used for grouping duplicates and archive keying. This is not a mode-table lookup.

## 3. Rewards and advantages

### 3.1 Validity advantage (unchanged GRPO)

- `r_val = 1.0` for an evidence-consistent hypothesis, `0.2` for a
  strictly syntax-valid but evidence-inconsistent hypothesis, and `0.0`
  otherwise, plus the shared overlength penalty. The penalty is zero through
  3072 response tokens, then decreases linearly to `-0.2` at the 6000-token
  cap; cap-hit responses are masked from the policy loss.
- `A_val` = verl's standard GRPO group normalization of `r_val`.

### 3.2 Diversity advantage

Let `V` = the VALID completions in the group. If `|V| < 2`, skip (all `A_div = 0`).

1. Group `V` by behavior key; let `U = {u_1..u_m}` be unique signatures, `c_j` = count of completions with signature `u_j`.
2. Distance `d(u_i, u_j)` = Hamming disagreement rate between signatures over `P` (normalize by `3·|P|`).
3. Kernel `K_ij = exp(−d(u_i, u_j) / ℓ)`, add `ε·I` (`ε = 1e-6`) for conditioning.
4. Per-unique-behavior credit: let
   `q_j = logdet(K) − logdet(K_{−j})`, then use
   `g_j = exp(q_j)`. Here `q_j` is the log conditional variance of behavior
   `j` given the others and is non-positive for the unit-diagonal kernel;
   exponentiation produces a positive marginal contribution in `(0, 1]`.
   Using `q_j` directly would make duplicate credit splitting reverse its
   intended effect because the raw credits are negative.
5. Archive scaling (see §4): `g̃_j = g_j · (1 + N(state_id, u_j))^(−1/2)`.
6. Per-completion raw diversity reward: `r_div,i = g̃_j / c_j` for completion `i` carrying signature `u_j` (duplicates split the credit — this is the intra-group duplicate tax).
7. `A_div`: normalize `r_div` (mean-subtract, divide by std with `max(std, 1e-4)` guard) **across `V` only**. Invalid/truncated/parse-fail completions get `A_div = 0`.

**Count variant** (config `variant=count`, ablation): replace steps 2–6 with `r_div,i = c(b_i)^(−1/2) · (1 + N(state_id, b_i))^(−1/2)`, where `c` is the within-group count of that behavior key. Same normalization.

### 3.3 Combination

`A_i = A_val,i + β · A_div,i`, applied per token as in standard GRPO.

- Do **not** fold `r_div` into the scalar reward before group normalization — the two streams are normalized separately, then summed. In verl this means a custom advantage step after the built-in group-relative computation (implementation detail left to the agent; the requirement is the separation).
- `β`: constant, default `0.3`. Optional guard (config flag, default on): if the running validity rate over the last `W=50` steps drops more than 10 points below its running max, halve `β`; never raise it back automatically.

## 4. Archive

- Store: `N[(state_id, behavior_key)] → float`, in-memory dict, checkpointed with the trainer state.
- Update: after each group's rewards are computed, `N[(state_id, b)] += 1` for every VALID completion (post-update, i.e., the scaling in §3.2 uses pre-update counts).
- Decay: at each epoch boundary, `N ← γ·N` with `γ = 0.7` (config).
- Config flag `archive=off` for the ablation (scaling factor ≡ 1, no writes).
- TRUNCATED / PARSE_FAIL / INVALID completions never write to the archive.

## 5. Sampling / generation config

- `G = 16` per state (config; must satisfy `G ≥ max M` in the dataset for interpretable coverage).
- Thinking ON; thinking budget cap ~1500 tokens (config), total max new tokens sized so a complete JSON answer always fits after the cap.
- Temperature and top-p: match across all method arms; log them.

## 6. Logging (eval path — ground truth allowed)

Per training step, per state, and aggregated per `(M, separation_bucket)` cell:

- validity rate; truncation rate; parse-failure rate (three separate numbers, not one).
- `coverage@G` = |distinct valid modes emitted in group| / `M`, using the true mode table to canonicalize.
- duplicity = 1 − (unique valid modes / valid completions).
- dominant-mode mass and effective mode count `exp(H(p̂(m|E)))` from rollout frequencies.
- mean token-level entropy of the policy on answer tokens (to plot against mode entropy — the decoupling figure).
- diversity-term diagnostics: mean `g_j`, kernel condition number, fraction of groups with `|V| < 2`, current `β`.
- archive stats: size, max count, mean scaling factor applied.
- CoT probe (cheap version): regex-count how many distinct parseable rule objects appear *inside* the thinking trace; log mean per completion. (Mechanism observable: deliberation vs sampling diversity.)

## 7. Config surface (single YAML block)

`variant {logdet|count}`, `archive {on|off}`, `beta`, `beta_guard {on|off}`, `ell` (kernel bandwidth, default 0.25), `gamma` (archive decay), `probe_fraction`, `G`, `think_budget`, plus standard verl GRPO settings. Every experiment arm in the thesis grid must be reachable by config alone — no code edits between arms. Baseline GRPO = `beta: 0`.

## 8. Tests (acceptance criteria)

Unit:
1. Parser rejects each malformed case in V0 spec §3 (one test per case); accepts a canonical valid hypothesis.
2. Executor: hand-computed signature for one known hypothesis on 3 named experiments matches.
3. Signature determinism: same hypothesis, same state, same seed → identical `P` and `s` across calls.
4. Kernel: symmetric, PSD after jitter; `d=0` duplicates collapse into one unique entry with correct `c_j`.
5. Duplicate tax: a group with 4 copies of behavior A and 1 of behavior B gives each A-copy strictly less `r_div` than B.
6. Archive: decay applies at epoch boundary; TRUNCATED completions produce no writes; `archive=off` reproduces `N ≡ 0` behavior bit-exactly.
7. Advantage separation: with `beta=0`, produced advantages are bit-identical to vanilla verl GRPO on the same rollouts.
8. Oracle firewall: reward-path package has no import path to the mode table / `private.*` (enforce with an import-linter rule or equivalent).

Integration:
9. Fixed synthetic batch (hardcoded 8 completions, known statuses/signatures): end-to-end advantages match hand-computed values to 1e-6.
10. Smoke run, ~200 steps on M=8/high-separation states: `beta=0` arm shows dominant-mode mass increasing; `variant=logdet, archive=on` arm shows `coverage@G` ≥ the `beta=0` arm at the same step count. (Direction check, not a significance test.)

## 9. Out of scope for this build

Multi-answer RLVR baseline (separate spec), the oracle-K skyline arm, real-world (program synthesis) transfer, VPO comparison. Don't build hooks for them beyond the config surface above.
