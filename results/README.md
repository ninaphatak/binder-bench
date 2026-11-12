# What the numbers say

Step 3 ran once, on 2026-11-12, against the analysis pre-registered in
`ANALYSIS_PLAN.md` and committed before any outcome was examined. The
population is the 378 expressed designs from Adaptyv Bio's EGFR competition
Round 2, of which 53 bind, a base rate of 14.0%. Nine of the 53 have KD pinned
at the 1e-5 M assay ceiling and are handled as censored.

## The headline

**Interface confidence did not predict binding in this dataset.**

All four metrics sit at chance on binary discrimination, and all four sit at
zero on affinity ranking. Every confidence interval includes the null value.

| metric | role | AUC | 95% CI | Spearman vs KD (n=44) | 95% CI |
|---|---|---|---|---|---|
| `pae_interface_mean` | primary | 0.460 | [0.384, 0.541] | −0.158 | [−0.449, 0.161] |
| `pae_interface_contacts_mean` | S1 | 0.478 | [0.394, 0.565] | −0.080 | [−0.383, 0.247] |
| `iptm` | S2 | 0.484 | [0.402, 0.568] | +0.110 | [−0.226, 0.409] |
| `ipsae` | S3 | 0.512 | [0.424, 0.604] | −0.012 | [−0.340, 0.306] |

Nothing survives Benjamini-Hochberg across the metric family. The decile
calibration curves are non-monotone and flat for all four, so these scores
behave neither as rankers nor as thresholds across their range.

This **contradicts** BindCraft's claim that interface confidence is a powerful
binary predictor of binding, and **replicates** its claim that it does not
track affinity. Both were asserted from box plots without interval estimates;
both now have intervals.

The achieved precision matches what SPEC predicted in advance: the
Hanley-McNeil SE is 0.042 against a forecast of 0.040 to 0.044, so this is a
real null at moderate effect sizes, not an underpowered shrug. An effect large
enough to be useful would have been detected.

## The one exception, and it is a real one

ipSAE carries information in its **extreme tail only**, which a global rank
statistic like AUC averages away:

| top k by ipSAE | binders | rate | hypergeometric p | enrichment |
|---|---|---|---|---|
| 10 | 8 | 80% | 3.3e-6 | 5.7x |
| 20 | 10 | 50% | 8.0e-5 | 3.6x |
| 50 | 12 | 24% | 0.03 | 1.7x |

k ∈ {10, 20, 50} was fixed in advance, and only one design ties the
tenth-ranked value, so the cut is not an artifact. The two standard metrics
show no such tail: `pae_interface_mean` puts 1 binder in its top 10 and `iptm`
puts 2, both at or below the 14% base rate.

So the practical reading is narrow but genuine: **do not use these scores to
rank a library, but the very top of an ipSAE ranking is worth ordering.**
That ipSAE alone shows this, while the two metrics the field actually
thresholds on do not, is a point in favour of correcting ipTM's chain-size
dependence (Dunbrack, bioRxiv 2025.02.10.637595) rather than a point about
confidence in general.

## What the positive controls say about the metrics

Two known binders were folded identically to the designs. Both reproduce their
crystallographic binding mode, at 6.6x to 11.1x enrichment over chance against
epitopes computed from PDB 1IVO and 1YY9, so the pipeline is sound.

Both nevertheless score as non-binders:

| control | KD | `pae_interface_mean` | `ipsae` | beats |
|---|---|---|---|---|
| Human EGF | 7.6 nM | 21.92 | 0.0123 | 55% / 52% of designs |
| Cetuximab scFv | 6.6 nM | 26.58 | 0.0000 | 7% / 0% of designs |

Cetuximab, a licensed drug, scores below **every one** of the 378 designs on
ipSAE. A confidence filter applied to these two molecules would have discarded
both.

## Caveats that belong in any writeup

1. **Absolute values are not comparable to published ones.** `pae_interaction`
   averages over every binder-target residue pair, so a 621-residue target
   inflates it. No literature threshold applies to these numbers. Only the
   ranks are meaningful, which is why every statistic here is rank based.
2. **Both controls are in AlphaFold's training data.** They validate the
   pipeline, never the metric. Their low scores are more striking for it, not
   less.
3. **This tests AlphaFold2-multimer**, the model whose metrics gated which
   designs Adaptyv chose to assay. AlphaFold3 and the Boltz line are untested
   here and are the obvious follow-up.
4. **The tested 400 were not a random sample** of the 1,131 submissions: 100 by
   a three-metric rank average and 300 hand-curated for diversity. The
   truncation analysis in `ANALYSIS_PLAN.md` section 2a addresses this and is
   not yet run.
5. **65% of designs have binder pLDDT below 50.** AF2 does not believe two
   thirds of these binders fold at all, which is context for what the interface
   scores are describing.

## Files

- `tables/claim_a_discrimination.csv` — AUC, CI, precision at k, per metric
- `tables/claim_b_affinity_ranking.csv` — Spearman, both censoring populations
- `tables/calibration_by_decile.csv` — bind rate per metric decile
- `tables/step3_summary.json` — everything above, machine readable

Reproduce with `uv run python scripts/run_step3.py --folds <dir>`.
