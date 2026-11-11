# Pre-registered analysis plan

**Status: STUB, NOT FINAL.** Opened 2026-11-09, during Step 0. No statistic in
this project has been computed at the time of this commit, and the git timestamp
on this file is the evidence for that.

This file must be complete and committed **before** anything in Step 3 is
computed. Sections marked TO BE WRITTEN are open. Anything already stated below
is fixed and may only be changed by an appended, dated, reasoned entry in
`DECISIONS.md`, never by a silent edit.

---

## 1. Step 3: two co-primary claims

Revised 2026-11-09 per HANDOFF-02. BindCraft's Discussion asserts two things
from box plots with no interval estimate: that interface confidence is a
powerful binary predictor of binding activity, and that it does not correlate
with interaction affinity. Both are separately interval-estimable here, so both
are pre-specified as **co-primary**. Each carries its own claim. This is not a
primary plus an exploratory follow-up.

- **Primary metric:** `pae_interface_mean`, for both claims. Every other metric
  is secondary and exploratory.
- **Primary population:** the **378 expressed designs** (`tested`). The
  `filter_pass` / `unbiased` arm split is deleted; those arms do not exist in
  Round 2 and the labels are not recoverable. Nothing in this plan is
  conditioned on an arm.

**Claim A, binary discrimination.** AUC separating `binds` from does-not-bind
over the 378, bootstrap 95% CI over 10,000 resamples, and whether that CI
excludes 0.5. Supported by precision at top k for k in {10, 20, 50} against the
**14.0%** base rate (53/378), and the calibration curve by metric decile. A
ranker gives a monotone calibration curve; a threshold gives a flat one.

**Claim B, affinity ranking.** Among the **53 binders only**, correlation
between the metric and KD. **Spearman primary**, Pearson on log10 KD secondary,
bootstrap CI. The 53 KD values span 3.92 log10 decades (1.21 nM to 10 uM), so
the dynamic range is real. Expected CI half-width is about 0.27 at this n
(Fisher z, 1.96/sqrt(n-3)). **n = 53 goes in the sentence, not a footnote, and
the CI is reported rather than hidden.**

### Claim B: the KD ceiling, pre-specified before any correlation is computed

**9 of the 53 binders carry KD exactly 1.000e-05 M.** The assay's highest
antigen concentration was 1000 nM, which is 1e-6 M, so 1e-5 sits a full decade
above anything actually flowed over the probe, and nine identical values at
that exact point is a reporting ceiling rather than nine measurements. Treating
them as real numbers would tie 17% of the data at the extreme of the range and
distort the correlation.

Fixed now, before computing anything:

- **Primary for Claim B:** Spearman on the **44 binders with KD below the
  ceiling**, since a censored value carries no rank information beyond "worse
  than everything measured". Report n = 44 in the sentence.
- **Secondary:** Spearman on all 53 with the 9 ceiling values retained as tied
  at the maximum, reported alongside so the effect of the choice is visible.
- Report both regardless of which is more favourable. Choosing between them
  after seeing the numbers is exactly the tuning the hard rules forbid.
- The CI will widen at n = 44 relative to n = 53, to roughly plus or minus 0.30
  (Fisher z). That is the honest interval and it gets reported, not hidden.

**Multiplicity.** A and B are co-primary and reported as such. Everything else,
the remaining six metrics and the six baselines, gets Benjamini-Hochberg
correction and is reported as exploratory.

Still to specify before Step 3: the exact bootstrap scheme (stratified by
outcome or not), the AUC tie-handling rule, and the random seed.

## 1a. Pre-specified secondary metrics

AMENDED 2026-11-11, before any metric-versus-outcome relationship was computed
and before any design's binding status was examined. Nina approved the addition.
What follows is an addition of secondary metrics only. The primary metric and
both primary claims above are **unchanged**.

### Why this amendment exists

The 12-design pilot showed `pae_interface_mean` running 16.76 to 26.70 with a
median of 23.16, and **not one** design below 10, which is the field's
conventional cutoff for a credible interface. Both positive controls, genuine
single-digit nM binders, sat inside that same range at 21.75 and 26.51.

The cause is in the metric's definition, not in the pipeline. Adaptyv's
`pae_interaction` averages over **every** binder-to-target residue pair:

    (pae[:b, b:].mean() + pae[b:, :b].mean()) / 2

With a 621-residue target, the overwhelming majority of those pairs are nowhere
near the interface, so the mean is dominated by distant residues and scales with
target size. Two consequences, both load bearing:

1. Our absolute values are **not** comparable to published values computed on
   smaller constructs, and **no absolute threshold from the literature may be
   applied to them**. This must not appear in the paper as though the scales
   matched.
2. If the primary metric fails, dilution by target size and genuine
   uninformativeness would be indistinguishable without a sharper variant.

The ranks are unaffected: the target is identical for all 378 designs, so a
constant offset cannot change AUC, precision-at-k, a decile calibration curve,
or a Spearman correlation. The primary claims stand as written.

### The two secondary metrics, pre-specified

**S1. `pae_interface_contacts_mean`.** The same PAE matrix, restricted to
binder-target residue pairs that are **actually in contact**, contact being the
5.0 Angstrom heavy-atom criterion already used by `geometry.py` and by the
control gate. Averaged over contacting pairs only, then averaged across the five
models exactly as the primary is. This removes the target-size dilution while
changing nothing else about the definition.

**S2. `iptm`.** Mean across the five models, already computed and stored for
every prediction. It is the other number the field thresholds on, so having it
pre-registered forecloses a reviewer asking whether the result was specific to
one metric.

**S3. `ipsae`.** ADDED 2026-11-11 on Nina's instruction, still before any
outcome was examined.

The published fix for exactly the defect S1 was invented to work around.
Dunbrack, *Rēs ipSAE loquuntur: What's wrong with AlphaFold's ipTM score and
how to fix it* (bioRxiv 2025.02.10.637595), shows that ipTM depends on the size
of the chain pair, so different constructs give different scores for the same
interface. ipSAE restricts the calculation to residue pairs whose interchain
PAE passes a cutoff, and recomputes the `d0` normalisation from the count of
residues that pass rather than from total chain length.

Two reasons it belongs here. It is the field's own answer to the target-size
dilution we measured, so omitting it leaves the obvious reviewer question
unanswered. And a BindCraft community guide already recommends ranking designs
by ipSAE over ipTM, so it is not merely a methods paper, it is in use.

Implementation commitment, fixed now: the **reference implementation** from the
Dunbrack Lab repository is used, not a reimplementation of ours, so what gets
reported is the published metric rather than our reading of it. Any deviation
forced by input format is documented in `DECISIONS.md`. Cutoffs are the
reference defaults; if the defaults are ambiguous, the value chosen is recorded
before computation, never selected by comparing outcomes.

S3 is computed **after** the fold, from the saved PAE matrices and structures.
It needs no GPU time and no re-folding. Our own S1 is retained alongside it,
and if the two disagree that is reported rather than resolved in favour of
whichever looks better.

### How they will be used, fixed now

- Both get the **identical** Claim A and Claim B treatment: AUC with a 10,000
  resample bootstrap CI, precision at top k for k in {10, 20, 50}, decile
  calibration curve, Spearman against KD among binders with a bootstrap CI.
- They are reported **alongside** the primary **whatever the outcome**, in one
  table, with the primary identified as primary. Not reported only if they win.
- No metric is promoted to primary after the fact. If a secondary discriminates
  better than the primary, that is stated as a secondary result and as a
  methodological finding about the metric's definition, never swapped in.
- Four metrics means four families of tests per claim. Benjamini-Hochberg
  correction across the metric family, as already specified elsewhere in this
  plan, applies.

### What would make this amendment illegitimate

It is legitimate only because it was committed before any outcome was examined.
The 12 pilot designs' `binds` values were deliberately not read; only the
distribution of the predictions themselves was inspected, which is quality
control on predictions and uses no outcome data. If any metric-outcome
relationship had been computed first, this amendment would be a forking path and
the secondary metrics would carry no inferential weight.

## 2. Step 3 power, for the record

The detectability floor is set by the number of positives. Hanley-McNeil
standard errors at the actual Step 3 split, **53 binders against 325
non-binders**:

| True AUC | SE | 95% CI | Excludes 0.5 |
|---|---|---|---|
| 0.60 | 0.044 | [0.514, 0.686] | yes |
| 0.65 | 0.043 | [0.565, 0.735] | yes |
| 0.70 | 0.042 | [0.617, 0.783] | yes |
| 0.75 | 0.040 | [0.671, 0.829] | yes |

So Step 3 resolves a moderate effect comfortably and a weak one not at all. A
weak effect is operationally useless anyway, so that limit costs nothing. The
Step 0 item 7 gate is recorded as **passed** on this basis.

Step 4 is the tighter constraint, because its sets are about 20 per side:

| n per group | AUC 0.65 | AUC 0.70 | AUC 0.75 |
|---|---|---|---|
| 20 | CI [0.479, 0.821], does not exclude 0.5 | CI [0.537, 0.863] | CI [0.597, 0.903] |
| 40 | CI low 0.530 | CI low 0.585 | CI low 0.643 |

## 2a. The truncation analysis, two tiers kept separate

Replaces the deleted `unbiased` secondary population.

**Tier 1, measured, and this is the tier that carries the argument.** All 1131
submitted sequences are public (`embeddings.zip`, `esm2_650m.csv`, keyed by raw
sequence; all 400 tested match exactly, leaving exactly 731 untested). Compare
tested against untested on sequence length, GRAVY, net charge at pH 7, and ESM2
pseudo-log-likelihood. The last is one of the three real components of the
selection rule. Report distribution overlap. This is a measurement.

**Tier 2, model-based, labelled as such wherever it appears.** The untested 731
**were never assayed and have no binding labels**, and their AF2 metrics would
cost about 2.9x the Step 2 fold budget, so no AUC can be computed on them. Tier
2 therefore states, under an explicitly declared extrapolation of
`p(binds | metric)` beyond the observed range plus an assumed shift in the
untested metric distribution, how far the untested would have to differ before
Claim A overturns. Deliverable is one sentence of the form *the result survives
unless the untested designs differ from the tested ones by more than X*. It must
never be presented as a measurement.

## 3. Step 4 variance rule

**Pre-specified now, before any trajectory is run.** This is the part of the
plan that item 7 of this file's eventual final version must not quietly relax.

**Triage tier:** 5 independent replicas x 10 ns per complex. Three replicas is
the absolute floor. Each replica gets its own initial velocity seed, and
independent equilibration where affordable; replicas sharing a velocity seed are
not independent samples.

**Equilibration discard:** the first 20% of each production run is discarded,
so 2 ns of a 10 ns triage replica, leaving 8 ns of usable signal. Taken from
UnibindMD's convention of averaging over the last 80 ns of 100 ns. Fixed in
advance because choosing it after seeing the traces would be tuning a threshold
after seeing the outcome it affects.

**The variance decomposition, and the escalation rule.** On the triage tier,
decompose the candidate scalar's variance into between-design and within-design
(replica) components and report the intraclass correlation
ICC = var_between / (var_between + var_within), with its interval.

- **ICC >= 0.5:** between-design signal dominates. The scalar is usable at 10 ns.
  Proceed to the matched-set AUC on the triage tier.
- **ICC in [0.2, 0.5):** marginal. Escalate by **adding replicas before adding
  nanoseconds**, because the Step 4 observable is early relaxation within a
  basin rather than a rare crossing event, so independent restarts sample it
  more efficiently than one longer trajectory at equal cost.
- **ICC < 0.2:** within-design noise swamps the between-design signal. The
  scalar is not usable at this trajectory length. Report that as the finding for
  that candidate scalar and do not extend it to 100 ns hoping the number moves.

Report per-replica numbers always. Never report a pooled mean alone.

**What the Step 4 claim is allowed to say.** Neither 10 ns nor 100 ns observes
unbinding: for a micromolar binder, k_off puts the complex lifetime in the
millisecond to second range, five to nine orders of magnitude beyond either
window. The admissible claim is that the predicted interface is not a free
energy minimum under CHARMM36m. The claim that the complex dissociates is not
available at these trajectory lengths.

## 4. Step 4 set definitions

TO BE WRITTEN BEFORE STEP 4, and before looking at any MD output. From SPEC.md:
confident-wrong is high primary metric and did not bind, with the threshold
chosen by percentile within `filter_pass` and written down first;
confident-right is the score-matched control that did bind. Target about 20 per
set, which the table in section 2 shows is the real constraint on the whole step.

## 5. Things that would invalidate this plan if done

- Computing any Step 3 statistic before this file is final and committed.
- Changing either co-primary claim, the primary metric, the primary population,
  the `binds` threshold or the denominator after seeing any outcome.
- Presenting the model-based truncation bound (tier 2) as a measurement, or
  merging it into tier 1.
- Demoting Claim B to exploratory after seeing its CI.
- Choosing between the censored and uncensored Claim B variants after seeing
  which is more favourable. Both get reported.
- Choosing the Step 4 percentile threshold or the equilibration discard after
  seeing the data they affect.
- Reporting a pooled-arm headline number (moot: no arm labels exist).
- Reporting a single-replica MD result.
