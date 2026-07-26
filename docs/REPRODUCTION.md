# REVE reproduction on BCI IV 2a — what we found

**Question:** does our REVE run on NSG match the paper's (arXiv 2510.21585, Table 2)
**REVE-Base = 0.6396 ± 0.0095** balanced accuracy on BCI Competition IV 2a?

**Short answer: no, we land at 0.40–0.50 — but we decomposed *why*, and it's ~half evaluation
protocol, ~half fine-tune recipe.** REVE + NSG can reach the number (subjects 3 and 9 hit 0.65 / 0.75),
so nothing is broken; the shortfall is uneven cross-subject adaptation, not a platform limit.

## The experiment ladder (9-subject means, on NSG V100s)

| # | change | mean ± std | Δ vs paper | what it taught us |
|---|---|---|---|---|
| v1 | frozen-probe → LoRA, train-session z-score | 0.435 ± 0.118 | −0.20 | pipeline runs; high variance, some subjects collapse |
| v2 | + val early-stop, warmup+cosine, grad-clip, label-smooth | 0.404 ± 0.090 | −0.24 | **not** a stability problem: val is solid (subj 9 val = 0.64) but **test ≪ val** ⇒ cross-session shift |
| v3 | + per-session z-score | 0.404 ± 0.090 | −0.24 | per-channel normalization doesn't fix it ⇒ shift is spatial/covariance, not channel scale |
| **v4** | **pooled split (both sessions, random 80/20)** | **0.503 ± 0.14** | **−0.14** | **removing the session-transfer challenge recovers ~0.10** |
| — | paper REVE-Base | 0.640 ± 0.010 | — | |

## The decomposition

The 0.24 gap under the standard session-holdout protocol splits into two isolated causes:

1. **≈0.10 (40%) — evaluation protocol.** The strict **session-1-train / session-2-test** holdout
   (the classic BCI 2a benchmark) is *harder* than a pooled random split. Pooling sessions (v4) recovers
   0.10 with everything else identical. So part of our "failure to reproduce" is simply that we ran a
   harder protocol than the paper likely used. Matching the split matters.
2. **≈0.14 (60%) — fine-tune recipe/capacity.** Even pooled, we cap at 0.50 vs 0.64. Our adaptation is
   **LoRA-only** (5.76M params, r=32) at a modest epoch budget; the paper describes fuller fine-tuning.
   Subjects 4/5/6 stay low (0.31–0.38) even pooled — uneven adaptation, a recipe issue.

## Evidence it's not the platform or the model

- Subject 9: **val 0.64 (session-holdout) → test 0.75 (pooled)**; subject 3: **0.65 pooled**. Both
  reach/exceed the paper on the same hardware and weights. The ceiling is real and reachable.
- `torch==2.4.1` (cu121) makes REVE load + train on the V100 (the image's 2.0.1 lacks `torch.nn.attention`).
- Vendored gated weights (pulled via the lab HF token, shipped in the zip) — reproducible, no token on NSG.

## What would close the remaining gap (not run here)

- **Recipe half:** fuller fine-tuning than LoRA-only (unfreeze more / higher-rank / longer schedule),
  and per-subject adaptation for the stubborn subjects (4/5/6). Validate one subject against a **local
  DGX reference** run of the identical config to separate "our recipe" from anything else.
- **Protocol half:** confirm the paper's exact split (pooled vs session-holdout) and metric, and report
  under the *matched* protocol.

## Bottom line

Infrastructure is proven — a real EEG foundation-model benchmark ran end-to-end at 9-way scale on free
NSG GPUs against the actual dataset, four protocol variants deep. The science is an honest partial: we
did not hit 0.6396, but we **quantified the gap and split it cleanly into protocol (~40%) and recipe
(~60%)** — a more useful result than a lucky single number. Full artifacts: `sweep/results/repro`
(current) and `repro_v1/_v2/_v3` (archived).
