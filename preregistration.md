# Pre-Registration — Action Lock-In

**Frozen:** 2026-06-18, *before any real model has been run* (hypotheses + analysis + kill-criteria
committed to git prior to data collection). The only quantity left PENDING is the numeric power
threshold δ, which is computed from a single-model smoke run's within-proposition SD (Task 9 Step 2)
and **frozen before the confirmatory grid run**.

Scope: **clean-phenomenon** (pure inference, no training, no real long-horizon agent task). Headline
rides on H2; main-conf is conditional on H2 being decisively positive, else honest poster / pivot to
backup B (Hedge Laundering).

---

## 1. Hypotheses (H1–H5 confirmatory; H6 exploratory)

Each hypothesis names the exact contrast key produced by `actionlockin/analyze.py::contrasts`.

- **H1 — masked commitment delta (DEMOTED to expected-secondary; sign contested).**
  `belief(action-commitment, dose 3) − belief(none) > 0`. Key: `h1_masked_commit_delta`.
  Sign is *not* asserted: self-readback confidence drift is negative in 2603.01239 (Claude CDS −0.032)
  and positive in 2501.09775 / CoT-overconfidence. We describe it; we do not claim it. H2 is a
  within-item residual that is robust to this background term's sign.

- **H2 — action-specific increment (SOLE HEADLINE).**
  `belief(action-commitment, dose 3) − belief(content-matched inline assertion) > 0`.
  Key: `h2_action_specific`. **Both pure self-conditioning (reading back ANY of one's own words raises
  confidence; 2603.01239 / 2501.09775) AND pure user-stance sycophancy (SWAY 2604.02423) predict
  H2 = 0.** The assertion arm carries the SAME factive lexicon as dose-3 ("proven … is true"), in
  declarative (non-action) form, same author (model-self), so H2 isolates *declarative-vs-action form*.
  This is the load-bearing novelty; no prior work has measured it.

- **H3 — commitment-vs-query asymmetry (DEMOTED to confirmatory mechanism check).**
  `belief(action-commitment, dose 3) − belief(action-query) > 0` after removing surface confounds.
  Keys: `h3_commit_vs_query` (raw, diagnostic only) and `h3_commit_vs_query_adj` (confound-adjusted
  `is_commit` coefficient — **the kill-gate quantity**). The commitment>query *direction* was already
  found by SWAY in the user-stance setting; we do **not** claim it as a discovery. Our increment =
  relocating it to a spontaneous-action locus, pre-result / result-masked, measured as logprob on the
  presupposed P (not stance-agreement).

- **H4 — dose-response.** Posterior rises monotonically with presupposition strength within the
  commitment framing. Keys: `h4_dose_slope` (raw) and `h4_dose_slope_adj` (dose coefficient controlling
  the covariates). The adjusted slope must remain positive to count.

- **H5 — placebo (negative control).** An action presupposing an *unrelated* claim Q does not raise
  belief(P). Key: `h5_placebo` = `belief(action-placebo) − belief(none)`; expected ≈ 0.

- **H6 — accuracy harm (EXPLORATORY).** On items where P is actually false (`gold == False`), the
  commitment action shifts the answer *toward wrong*. Key: `h6_accuracy_shift` = `commit3 − none` on
  `gold == False` items. Exploratory only; uncorrected.

---

## 2. Primary measurement

- **Primary readout (open-weight models):** token logprob of the True/False answer token at a fixed,
  byte-identical belief probe appended after the manipulated middle, rendered with the model's **native
  chat template** (`tokenizer.apply_chat_template`); the middle is injected as a genuine **assistant
  turn** (teacher-forced), then a fixed answer cue elicits True/False. `prob_true` = softmax over
  `" True"` vs `" False"` first-token logits.
- **Moat statement:** the logprob-over-choices probe is itself **not novel** (2512.23184 already defines
  model belief this way). Novelty = the *combination*: masked result + a presupposition P that is never
  asserted + content-matched assertion subtraction + native-chat-template self-readback. Related work
  must not sell the probe as the contribution.
- **Invariant:** the belief probe is byte-identical across all conditions for a given proposition; only
  the injected middle differs. The answer-token position is therefore locked by the fixed probe.
- **Secondary (generalization/closed models):** verbalized 0–100 confidence on 1–2 frontier models;
  noisier, robustness only; **must not gate H1–H5.**

## 3. Conditions (within-item)

Per proposition, the injected assistant middle is one of (keys = `condition/framing/dose`):

| cell | condition/framing/dose | role |
|---|---|---|
| none | `none/na/0` | baseline neutral reflection |
| assertion | `assertion/na/0` | content-matched declarative ("It is proven that P is true.") |
| action-query | `action/query/0` | non-presuppositional action ("whether it is true that P") |
| action-placebo | `action/placebo/0` | action presupposing unrelated Q (result masked) |
| commitment d1–d3 | `action/commitment/{1,2,3}` | presuppositional action, graded factive; result masked |

All action cells return a **masked result** (`Result: (no results returned)`) — measured strictly
*before* any informative result arrives.

## 4. Confound control (as actually computed in code)

Two lines of defense: **length-match first, regress second.**

- Length-match: within each proposition the condition middles are padded to a common per-model token
  count (`pad_to_match` with the model tokenizer) before scoring.
- Regression covariates — exactly the three columns `confounds.text_features` produces, no others:
  - `mid_n_tokens` — number of middle tokens (in-context).
  - `mid_mean_logprob` — mean **in-context** per-token logprob of the middle (fluency/perplexity proxy,
    conditioned on system+user prefix — NOT the middle scored in isolation).
  - `mid_claim_salience` — relative end-position of the last mention of the claim inside the middle.

## 5. Analysis (confirmatory, matches `analyze.py`)

- Per-proposition paired deltas + bootstrap 95% CI, **computed separately per model**; **per-model
  GO/NULL verdicts are the primary result.**
- H3 gate = the **confound-adjusted** `is_commit` coefficient from
  `p_of_P ~ 1 + is_commit + mid_n_tokens + mid_mean_logprob + mid_claim_salience` with a
  cluster(proposition) bootstrap CI.
- Pooling = **cluster bootstrap over models** (the model is the replication unit); never average a
  proposition across models. With only 3–5 models, model is **NOT** a variance-component random effect
  (not identifiable); an optional secondary uses model as a FIXED effect + per-prop random intercept.

## 6. Kill-criteria (frozen — verdict GO requires BOTH)

`evaluate_kill_criteria`: **GO iff** `h2_action_specific.ci_low > 0` **AND**
`h3_commit_vs_query_adj.ci_low > 0`. Otherwise report a **clean NULL** (an honest, publishable finding /
graceful downgrade). The gate reads the *adjusted* H3, not the raw delta.

## 7. Multiple comparisons

- Headline = **intersection-union test** (H2 ∧ H3); requiring both to be significant already controls
  the family error rate — **no extra correction** (adding a conjunct only makes it harder).
- Secondary H1 / H4 / H5 → **Holm** correction. H6 exploratory → uncorrected.

## 8. Power / minimum detectable effect δ  — **PENDING (frozen before the confirmatory run)**

> Computed in Task 9 Step 2 from the smoke run's within-proposition paired SD of `(commit3 − assertion)`,
> using the **same** per-proposition paired bootstrap estimator as `analyze.py::_boot_ci`. Sweep small
> effects in P-space (0.01–0.05) at N ≈ 250, report the smallest δ with power ≥ 0.8, and **freeze δ here
> before any confirmatory grid run.**

- **δ = TBD** (fill from smoke SD; do not hard-code before measuring).
- **Null-interpretation rule (pre-registered):** a per-model H2 CI lying entirely within [−δ, +δ] reads
  as "no effect larger than δ"; otherwise "inconclusive / underpowered" — NOT taken as evidence of
  absence.
- **Generalization N:** cross-model conclusions rest on per-model replication across 3–5 models
  (descriptive); no population inference is claimed.

---

*Stopping rule:* hypotheses, conditions, primary metric, covariate list, analysis, kill-criteria, and
multiple-comparison plan are frozen as of this commit. Only δ (§8) is filled post-smoke, pre-confirmatory.
Backup: if H2 fails under the inline-assertion control (commit3 not reliably > assertion), pivot to
backup idea B (Hedge Laundering).

---

# Revision v2 (pilot-driven, 2026-06-18) — supersedes the operationalization & primary metric above

The §1–§8 plan above was frozen **before any model was run** (its value as a record stands). A
single-model pilot (Qwen2.5-1.5B, then a 7B probe) then surfaced three issues that **invalidate the v1
operationalization and primary metric** but leave the hypotheses' substance intact. This section freezes
the corrected design. The honest v1→v2 trail is intentional and lives in git history
(`results/gate_v2_*.md`, `results/gate_multimodel.md`).

**Why v1 failed (pilot evidence):**
1. The v1 commitment action `search("documents that prove P")` *lowered* belief: searching for proof
   pragmatically implies P is unproven, and the masked `"(no results returned)"` was read as
   disconfirming evidence. → action LOWERED P(P), opposite to the hypothesis.
2. The v1 inline assertion `"It is proven that P is true."` saturated belief (≈0.96), making
   `H2 = commit − assertion` structurally negative regardless of any real action effect.
3. The probability readout has a 0/1 ceiling; capable models (7B+) are near-deterministic on factual
   claims, so almost no propositions land in the [0.2,0.8] mid-band → **n≈1 on 7B** (no generalization).

**v2 operationalization (frozen):**
- **Commitment action = premise-style**, files P as an established fact, graded factive dose:
  `log_fact(fact=P)` → `log_fact(fact=P, status="established")` → `log_fact(fact=P, status="confirmed and verified")`.
  **No result line** — belief is probed strictly *before* any result (the core construct).
- **Inline assertion = content-matched to dose-3, non-saturating**: `"It is confirmed and verified that P."`
- **Query (non-presuppositional)**: `check(query="whether P is true")`. **Placebo**: dose-3 commitment on an unrelated claim.

**v2 primary metric (frozen): log-odds.** Primary readout = `logit(True) − logit(False)`
(`scorer.logit_diff`), recorded as `logodds_P`. Log-odds is unbounded, so confident items keep headroom
and **the FULL proposition set contributes on every model — mid-band calibration is no longer required**
(it is dropped from the confirmatory run; probability `p_of_P` is retained only for the answer-flip
accuracy-harm metric). All §5 analysis machinery is unchanged; it now runs on `logodds_P`
(`contrasts(df, value="logodds_P")`).

**Revised hypothesis emphasis (the pilot inverted the headline):** across 1.5B and a 7B probe, in both
probability and log-odds space, **H1 (action raises pre-result belief), H3 (commit > query), H4
(dose-response), and H6 (accuracy harm on false items) are robustly POSITIVE; H2 (action > content-matched
assertion) is NEGATIVE (≈0 at scale).** Interpretation: the lock-in is driven by the **presupposed
content**, not the syntactic action form. Therefore:
- **Headline (confirmatory):** presupposition-carrying self-authored content (including tool calls)
  raises an agent's posterior on P *before any result*, graded by factive strength, more than a
  non-presuppositional query, and **shifts answers on false claims toward wrong (accuracy harm).**
- **H2 is reframed** from headline to a **mechanistic dissociation**: it tests whether the action *form*
  adds lock-in beyond equal content. Finding H2 ≤ 0 is reported as the result (form does not add; content
  drives it), pre-empting the "it's just self-conditioning over the action tokens" critique.

**Revised success criteria (frozen, replaces §6 kill-criteria):** the phenomenon is established iff,
**per model**, in log-odds: `h1_masked_commit_delta.ci_low > 0` AND `h3_commit_vs_query.ci_low > 0` AND
`h4_dose_slope.ci_low > 0`, with `h5_placebo` not positive (specificity). H6 (accuracy harm) is the
exploratory impact result. H2 is reported as a signed dissociation, not gated. If H1/H3/H4 fail to
replicate across the open-model set, pivot to backup B.

**Pending:** the confirmatory multi-model log-odds run (`scripts/gate_multimodel.py`) over the cached
open models; a fresh scoop re-check for the reframed headline (vs self-anchoring 2603.01239,
SWAY 2604.02423, presupposition-acceptance 2505.22354) emphasizing the masked/pre-result/log-odds/
form-vs-content-dissociation/accuracy-harm combination.

---

## Completion addendum (appended 2026-07-08; no text above this line was modified)

**Freeze trail (git):** `8d4e6b5` (2026-06-18) froze v1 before any model run (δ pending smoke);
`b055bdf` (2026-06-18) recorded the pilot-driven v2 revision (log-odds primary, headline reframed,
H2 demoted to signed dissociation) before the confirmatory multi-model run; `a450510` (2026-07-07)
removed two dead doc pointers only. The operative gate is the **revised success criteria** block
above (which explicitly replaced the §6 kill-criteria, including the earlier adjusted-H3 gate).

**Confirmatory runs executed:**
- 2026-06-18, local: `scripts/gate_multimodel.py` over Qwen2.5-1.5B/7B-Instruct and
  Mistral-7B-Instruct-v0.3 (120 propositions, log-odds).
- 2026-06-24, rented 2×RTX PRO 6000: `scripts/run_gpu_all.py` over Qwen2.5-32B/72B-Instruct, plus
  recomputation of 1.5B/7B under the corrected gold labels (p030/a151/a161), on all three claim sets.
  Retained per-proposition parquets: `results/gate_*_{propositions,arith_props,arith_hard}.parquet`
  (see `results/README.md` for the canonical manifest).

**Outcomes against the frozen criteria (per model, log-odds):**
- `h1_masked_commit_delta.ci_low > 0`, `h3_commit_vs_query.ci_low > 0` (raw), and
  `h4_dose_slope.ci_low > 0`: **pass on all five models**; they also pass under Bonferroni k=20
  (99.75% CIs, `scripts/bonferroni_frontier.py`).
- `h5_placebo` not positive: point estimates ≤ 0 at 1.5B/7B/32B/Mistral. **At 72B the point
  estimate is +0.02 (99.75% CI [−0.88, +0.88], spanning zero)** — near-zero rather than strictly
  non-positive; the paper reports this exception openly (§Results: a gold-dependent cancellation
  masking a positive placebo push on false claims at 7B/72B/Mistral).
- H2: negative on all five models — the v1 headline (H2 > 0) is **refuted** and reported as such.
- H6 (exploratory): accuracy/decision harm reported in the paper's downstream sections.

**Deviations disclosed:**
1. §8 (power / minimum detectable effect δ) was never computed; the confirmatory run proceeded
   without the planned δ freeze. Observed corrected lower bounds (smallest overall +0.23) make the
   study's sensitivity moot for the reported effects, but the planned step itself was skipped and is
   disclosed as such, not patched post hoc.
2. The "fresh scoop re-check" listed as pending was folded into the paper's Related Work rather than
   performed as a separate documented step.

No analysis choices were altered after seeing confirmatory data beyond the documented v1→v2 trail.
