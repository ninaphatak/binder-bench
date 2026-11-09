# Pre-registered analysis plan

**Status: STUB, NOT FINAL.** Opened 2026-11-09, during Step 0. No statistic in
this project has been computed at the time of this commit, and the git timestamp
on this file is the evidence for that.

This file must be complete and committed **before** anything in Step 3 is
computed. Sections marked TO BE WRITTEN are open. Anything already stated below
is fixed and may only be changed by an appended, dated, reasoned entry in
`DECISIONS.md`, never by a silent edit.

---

## 1. Step 3 primary test

TO BE WRITTEN IN FULL BEFORE STEP 3. Fixed already, from SPEC.md:

- **Primary metric:** `pae_interface_mean`. Every other metric is secondary and
  exploratory.
- **Primary population:** the pAE-selected arm (`filter_pass`).
- **Primary statistic:** AUC separating `binds` from does-not-bind, with
  bootstrap 95% CI over 10,000 resamples, and whether that CI excludes 0.5.
- **Multiplicity:** roughly thirty tests across seven metrics, two arms and six
  baselines. Secondary metrics get Benjamini-Hochberg correction and are
  reported as exploratory. Only the primary test carries the claim.
- **Never** pool the two arms into a headline number.

Still to specify: the exact bootstrap scheme (stratified by outcome or not), the
tie-handling rule for AUC, the random seed, and the precision-at-k comparison
against the arm's own base rate.

## 2. Step 3 power, for the record

The detectability floor is set by the number of positives, not by anything in
the pipeline. Hanley-McNeil standard errors, balanced groups:

| n per group | AUC 0.65 | AUC 0.70 | AUC 0.75 |
|---|---|---|---|
| 20 | CI [0.479, 0.821], does not exclude 0.5 | CI [0.537, 0.863] | CI [0.597, 0.903] |
| 40 | CI low 0.530 | CI low 0.585 | CI low 0.643 |

This is why the Step 0 item 7 gate is set at 25 positives. A null result below
that count is uninterpretable rather than informative, which is the one outcome
with no publishable reading in either direction.

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
- Changing the primary metric, the primary population, the `binds` threshold,
  the denominator or the arm definitions after seeing any outcome.
- Choosing the Step 4 percentile threshold or the equilibration discard after
  seeing the data they affect.
- Reporting a pooled-arm headline number.
- Reporting a single-replica MD result.
