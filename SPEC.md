# Experiment 01: Is interface confidence a threshold or a ranker?

Spec for a Claude Code agent. Read this whole file before writing any code.
Compute backend: **Modal**.

---

## Working context

You are working for **Nina Phatak**, a researcher in **Giulia Palermo's
computational chemistry lab at UCLA**. This matters operationally in four ways.

**MD protocol.** Use the published binder-validation convention (Step 4), not
a protocol invented here and not the Palermo lab's. The lab's conventions are
tuned for protein-nucleic acid complexes; this is protein-protein, so most of
what transfers is solvent and ion treatment. If Nina can get the lab's setup
cheaply it is worth recording for comparability, but it is not a dependency and
must not block Step 4.

**Compute options beyond Modal.** UCLA provides institutional HPC (Hoffman2),
and the Palermo lab has its own allocation and trajectory-generation capacity.
Modal is the default here because it needs no queue and is fast to stand up,
but for Step 4's long MD an institutional allocation may be much cheaper. Raise
this with Nina before assuming Modal for Step 4.

**Model weights eligibility.** Academic affiliation affects access. AlphaFold 2
and ColabFold weights are openly licensed and fine to use. AlphaFold 3 weights
require an application and are restricted to non-commercial academic use, which
the UCLA affiliation satisfies should AF3 ever be wanted. Record which weights
were used, and under which license, in the run provenance.

**Attribution.** Figures, preprint and repo carry the UCLA affiliation. This is
Nina's own first-author project, independent of the lab's S15/CAST work, so do
not describe it as a Palermo lab output. It is her project, conducted while in
the lab.

---

## Context you need

**The filter under test.** Every de novo binder pipeline ends the same way:
fold the designed complex with a structure predictor, keep the designs with
good interface confidence (ipTM, interface pAE, pLDDT). This experiment asks
what that filter is actually worth.

**Why it might be worth little.** Those metrics are trained to predict
agreement with a deposited crystal structure. They estimate how likely a
prediction is to match a structure someone could solve. Binding is a free
energy difference, and a factor of ten in affinity is about 1.4 kcal/mol, less
than a hydrogen bond. There is no reason a geometric confidence score resolves
that.

**The question, stated precisely.** Not "does interface confidence separate
binders from non-binders across all designs." That question is not answerable
on this dataset (see the selection-bias section). Instead:

> Among designs that already pass the confidence filter the field uses, does
> interface confidence carry any **residual** information about whether a
> design binds?

If the answer is no, the metric is a **threshold, not a ranker**. Every
GPU-hour spent optimising an already-passing design toward higher confidence is
wasted, and the designs that survive the filter and still fail are
indistinguishable by it. That is an actionable finding for anyone running a
design pipeline, and it is immune to the selection-bias objection because the
dataset *is* the post-filter population.

**Relationship to the existing repo.** There is an older unfinished repo,
`enrich-bench`, containing a working ProteinMPNN sequence-generation step. That
code is **not needed here** — this data already has sequences with measured
outcomes. Do not wire it in. Experiment 01 is an evaluation pipeline: external
sequences in, scores and statistics out.

Recommendation: start a clean `binder-bench` repo. The data flow is inverted
relative to `enrich-bench` (observed sequences rather than generated ones), and
grafting will muddle both. Keep `enrich-bench` as a sibling; its generation
code becomes the input to a later experiment.

---

## THE SELECTION-BIAS PROBLEM, read before touching the data

REVISED 2026-11-09 (HANDOFF-02, after Step 0 item 2). The earlier version of
this section described **Round 1's** arms and labelled them Round 2's. What
follows is from primary text: the competition README, `adaptyvbio/competition_metrics`,
and the lessons preprint (10.1101/2025.04.17.648362).

Competitors submitted far more designs than Adaptyv could assay. What actually
happened:

- **Round 2, 400 tested out of 1131 submitted.** 100 selected by a **rank
  average of three metrics**: iPTM, PAE interaction and ESM2 pseudo-log-likelihood,
  with fractional ranking for ties. Plus **300 chosen by hand** by Adaptyv staff
  for novelty, creativity and method diversity, under the constraint that every
  designer got at least one design tested.
- **Round 1, 201 tested out of 726.** 100 by **iPAE alone** plus 101 hand-selected.
- **There is no pseudo-random arm in either round.**

Consequences, as revised:

1. **Only 100 of 400 had any confidence metric in the selection objective,** and
   there it was one of three components alongside a language-model likelihood.
   There is no pAE-only arm in Round 2.
2. **The resulting enrichment is mild and attenuating, not inflating.** Any
   pull toward the low-pae end biases *against* finding an effect, so a null
   result cannot be dismissed as an artifact of selection, and a positive result
   is conservative.
3. **Per-design arm labels are unpublished and the ranking is not
   reconstructible.** The three metrics are published only for the 400 tested,
   not the 1131 submitted. Reconstruction would need 731 extra folds, about 2.9x
   the Step 2 budget. Do not attempt it.
4. **The 14% success rate is conditional** on having been chosen for assay. Do
   not quote it as the field's base rate.

**How this spec handles it:** the analysis population is the 378 expressed
designs, described plainly as *designs Adaptyv chose to assay*, score-based for
a quarter of them and human-curated for the rest. No correction is applied. The
truncation is **measured** instead, see Step 1.

**Status of the threshold-versus-ranker framing.** Retained for now. The spread
of `pae_interaction` among tested designs is wide (6.80 to 27.89, median 16.17,
binders 7.57 to 27.46), which is consistent with mild truncation, but spread
among the tested says nothing on its own about the 731 untested. The framing is
robust to not yet knowing, so it stays until the Step 1 truncation measurement
is in hand. Revisit then, not before.

---

## Step 0: Inventory and the go/no-go gate

Report back after each numbered item. Do not proceed past item 7 without Nina.

0. **Prior-art read, before any code.** Three things must be read and
   summarised in `PRIOR_ART.md`, one paragraph each, with an explicit sentence
   on what this experiment does that the work does not:
   - **BindCraft** (Nature 2025) Extended Data, where experimentally validated
     binders *and non-binders* from earlier pipelines are repredicted and false
     positives are shown. This is the closest published thing to Step 3.
   - **EvoBind2** (Commun Chem 2025), which names the phenomenon
     **"adversarial designs"** (high in-silico confidence, no binding) and
     reports an AlphaFold-multimer adversarial check improving success rate
     2.5x. This is a cheap baseline that our MD signal must beat or complement.
   - **UnibindMD** (bioRxiv, Oct 2026), a five-stage pipeline ending in 100-ns
     GROMACS/CHARMM36m MD validation. Peptide binders, pipeline paper, not a
     benchmark. Source of our MD protocol; also the nearest competitor.

   **Use the field's vocabulary.** The target population is "adversarial
   designs," not "confidently wrong designs." Reviewers expect the term.

1. Inventory `enrich-bench`: what exists, what runs, what is half-written.
   Three lines. Do not refactor it.
2. Clone `github.com/adaptyvbio/egfr_competition_2`. Confirm per-design
   `selection_arm` labels exist. **DONE 2026-11-09: they do not exist.** No arm,
   batch or group field in any of the six result files, and the README does not
   describe the selection. Denominators confirmed: 400 designs plus 2 controls,
   22 expressed none, 378 expressed, 53 bound.
3. ~~Inventory Round 3~~ **REVISED 2026-11-09: there is no Round 3.** 601 is the
   two-round total (201 + 400). The real question is whether **Round 1's 201
   designs** can be pooled with Round 2. ANSWERED, item 3 complete: **no.**
   Round 1 contributes about 5 binders (202 rows, only 8 carrying a KD, one of
   those the Cetuximab control, 3 of the rest pinned at the 1e-5 to 2.3e-5
   assay ceiling; preprint hit rate 5/201). Against roughly +9% on the
   positives: Round 1's `result_summary.csv` has no `binding` or `expression`
   columns at all, labels exist only at replicate level with 162 of 461
   replicates "unknown", expression was 73% against 94 to 95%, the selection
   rule differed, and its `pae_interaction` was computed without templates.
   Round 2 alone. Round 1 is an out-of-sample descriptive check at most.
4. Check whether either repo ships predicted structures or confidence metrics.
   Note them as a comparison point, **not** as our predictions.
5. Confirm `modal` is installed and authenticated.
6. Confirm Nina's **Modal budget** in dollars and record it in
   `DECISIONS.md`. New-account credits do not cover 378 GPU folds. Every cost
   estimate later in this spec is checked against that number, not against a
   vague sense of "reasonable."
7. **THE GATE. PASSED 2026-11-09.** Revised: there is no pAE-selected arm to
   count, so the gate is evaluated on the pooled population of 378 expressed
   designs. **53 binders against 325 non-binders.** Hanley-McNeil SE on AUC at
   that split is 0.042 to 0.044 across AUC 0.60 to 0.75, giving a 95% CI of
   about plus or minus 0.083. That resolves a moderate effect but not a weak
   one. A weak effect is operationally useless anyway, so the limit costs
   nothing. Proceed with Round 2 alone.

## Step 1: Build the dataset

Produce `data/designs.parquet`:

| column | notes |
|---|---|
| `design_id` | |
| `sequence` | binder amino acid sequence |
| `design_method` | self-reported by competitor; messy, keep verbatim |
| ~~`selection_arm`~~ | DROPPED 2026-11-09: unpublished and not reconstructible |
| `expressed` | bool |
| `binds` | bool |
| `kd` | nullable float |
| `neutralisation_coefficient` | nullable; Round 2 ran a neutralisation assay |

### Decisions to make explicitly and log in `DECISIONS.md`

**Denominator.** Round 2: 400 tested, 378 expressed, 53 bound. Restrict to the
378 that expressed. A design that failed to express says nothing about whether
the structure prediction was right — that is a different failure mode
(solubility, aggregation, codon usage). Log the exclusion count.

**Label definition.** Define `binds` from the assay, not from a leaderboard
rank. Write down the threshold and its source.

**Analysis populations.** REVISED 2026-11-09. There is one, not two. The
`filter_pass` / `unbiased` split assumed arms that do not exist in Round 2 and
cannot be recovered.

- **Primary (`tested`)**: the 378 expressed designs. Described as *designs
  Adaptyv chose to assay*, score-based for 100 of 400 and human-curated for the
  other 300.
- **`selection_arm` is dropped from the schema.** It is unpublished and not
  reconstructible. If Adaptyv supply labels later they feed a robustness check
  only, never the primary analysis.

### Also build: the truncation picture. NOW REQUIRED, not optional.

This replaces the deleted secondary population and is the load-bearing answer
to the selection-bias objection. It is constructible from published data today.

**All 1131 submitted sequences are public**, in the competition's
`embeddings.zip` as `esm2_650m.csv`, 1131 rows keyed by raw sequence. All 400
tested sequences match exactly, 0 missing, leaving exactly **731 untested**.

Two tiers, reported separately:

1. **Measured, and this is the one that carries the argument.** Compare the 378
   tested against the 731 untested on features computable without any structure
   prediction: sequence length, GRAVY, net charge at pH 7 (all free) and **ESM2
   pseudo-log-likelihood** (a few CPU-hours, and it is one of the three real
   components of the selection rule). Heavy overlap means the selection was not
   strongly score-driven; sharp separation quantifies the truncation. Either way
   it is a measurement.
2. **Model-based bound, reported separately and labelled as model-based.** How
   far would the untested designs have to differ before the primary result
   overturns. State the extrapolation explicitly: it requires an assumed
   `p(binds | metric)` extended past the observed range and an assumed shift in
   the untested metric distribution, because **the untested designs were never
   assayed and have no binding labels, and never will.** Deliverable is one
   sentence of the form *the result survives unless the untested designs differ
   from the tested ones by more than X*. Do not present it as a measurement.

The AF2 metrics for the untested 731 are out of reach: 731 extra folds is about
2.9x the Step 2 budget. Do not plan around having them.

### Target construct

Predictions need the exact EGFR construct used in the assay. Competitors
targeted a specific domain, not the whole ectodomain. Pin it down from the
competition materials.

**Do not guess the construct.** A wrong target makes every prediction
meaningless, and it is the single cheapest way to waste the entire GPU budget.
**Stop and confirm with Nina before Step 2.**

---

## Step 2: Re-predict everything, uniformly, on Modal

**Do not use any confidence score reported by the competition or by
competitors.** Those came from different models, versions and configs. That
inconsistency is real — one competitor reported designs in the predicted top
100 by ColabFold scoring much worse under AlphaFold 3 — and inheriting it means
measuring version noise instead of the thing we care about.

### Evaluator choice

**Primary: AF2-multimer via ColabFold.** Rationale: this is the metric the
field actually filters on. BindCraft hallucinates against AF2 weights, and the
competition's own selection used `pAE_interaction`. Putting the real-world
filter on trial is the point.

**Secondary, DEFERRED: Boltz-2.** Cross-model agreement is a genuine result
(if two models disagree about which designs are confident, the metric is not
self-consistent, independent of any binding data). But it doubles this step's
compute and is not on the critical path for any version of the paper. **Do not
run it in the first pass.** Revisit after Step 3 if budget and timeline allow.

### MSA strategy — important, and it simplifies the Modal build

The binder is de novo and has no meaningful MSA. Standard practice for binder
prediction is **target MSA + single-sequence binder**. So you need **exactly
one MSA, for the EGFR construct**, computed once and cached in a Modal Volume,
reused across all 378 predictions.

Do **not** call the public MMseqs2 API per design. 378 calls from a cloud IP
will be rate-limited or blocked, and it would introduce run-to-run variation in
the MSA, which is precisely the inconsistency this step exists to eliminate.

### Config to fix and record

Recycles, number of models/seeds, MSA mode (target-only), templates off,
relaxation off, random seed policy. Record the model weights version/hash and
**the GPU type the container ran on** — GPU architecture can affect numerics,
and you want that in the provenance.

### Provenance

Mirror `enrich-bench`'s `costs.csv` schema, including `n_inputs`, `git_sha` and
`git_dirty`. It is better provenance than the rest of this step asks for:
`n_inputs` is what makes dollars-per-design recoverable later, and a sha is
meaningless if the tree was dirty.

### Metrics to record per design

- `iptm`, `ptm`
- `pae_interface_mean` (mean over interchain residue pairs)
- `pae_interface_min`
- `plddt_binder_mean`
- `plddt_interface_mean`
- `n_contacts_5A` (interchain heavy-atom contacts)
- `bsa` (buried surface area)
- full pAE matrix saved per design — Step 4 may need it

**Named check: chain ordering.** Interface metrics are computed over
*interchain* residue pairs, so they depend on knowing which chain is the binder
and which is the target. If chain order flips between designs, or between your
parser and the model's output convention, you will compute the metric on the
wrong block of the pAE matrix and get numbers that look plausible and mean
nothing. Assert chain identity and length explicitly per design, and fail loudly
on a mismatch rather than defaulting.

Save predicted structures; Step 4 uses them as MD starting coordinates.

### Modal architecture

Verify every API detail against current Modal docs before writing code. The
SDK is at `modal>=1.0`; `mounts=` on `@app.function` is superseded by
`Image.add_local_dir()` / `Image.add_local_python_source()`, so check which
pattern is current.

**Look for an existing Modal example for ColabFold or Boltz before building the
image from scratch.** Containerising ColabFold is the largest technical unknown
in this spec, and Modal publishes protein-folding examples. Starting from a
working image instead of debugging CUDA and JAX versions could save a week.

Shape:

```python
import modal

app = modal.App("binder-bench")

weights = modal.Volume.from_name("bb-weights", create_if_missing=True)
data    = modal.Volume.from_name("bb-data",    create_if_missing=True)
results = modal.Volume.from_name("bb-results", create_if_missing=True)

fold_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    .uv_pip_install(...)            # pin every version
    .run_function(download_weights, volumes={"/weights": weights})
)

@app.function(
    image=fold_image,
    gpu=["A100-40GB", "L40S"],      # fallback chain
    timeout=3600,
    volumes={"/weights": weights, "/data": data, "/results": results},
    max_containers=4,               # RAISE ONLY AFTER THE PILOT
)
def fold_one(design_id: str) -> dict:
    ...

@app.local_entrypoint()
def main(pilot: bool = True):
    ids = pilot_ids() if pilot else all_ids()
    for rec in fold_one.map(ids):
        ...
```

Rules for the Modal layer:

- **Bake weights into the image** with `run_function` at build time, or stage
  them into the weights Volume once. Containers must never re-download per
  invocation.
- **Idempotency.** Write per-design output keyed by `design_id` to the results
  Volume, and have `fold_one` return early if its output already exists.
  Containers die; a re-run must resume, not restart.
- **`max_containers` stays low until the pilot is costed.** This is the main
  guard against an accidental large bill.
- **One function, one job.** Folding, MD and analysis are separate functions
  with separate images. Do not build one image that carries everything.
- Volumes auto-commit in the background; call the explicit commit before a
  function returns if the docs still recommend it for the version in use.

### Positive control, before the pilot

REVISED 2026-11-09. The controls are already in the dataset and were assayed on
this exact construct, which is better than an external panel because the assay
matches: **Cetuximab_scFv at KD 6.6 nM and Human_EGF at KD 7.6 nM** (both
single-digit nM, not the ~1e-6 M this spec previously expected). Fold both
before any competition design. They must come out with strong interface scores.

If they do not, something upstream is wrong, most likely the target construct,
possibly chain ordering or the MSA. This control costs minutes and is the only
cheap way to catch a wrong construct before it eats the whole GPU budget. **Do
not proceed to the pilot until it passes.**

Caveat to state in the paper, and it matters **more** now that the controls are
the in-dataset ones: cetuximab and EGF are squarely in the models' training
data, so they will score well partly from memorisation. The control validates
the *pipeline* (construct, chain ordering, MSA) and **never the metric**. State
that explicitly wherever these two appear. Do not present them as evidence the
metric works.

### Compute discipline — pilot first, always

Run `fold_one.map()` over **10 designs** first. Record wall-clock, GPU-seconds
and peak memory per design. Then:

1. Read Modal's **current** pricing page (do not assume a rate from memory).
2. Extrapolate to 378 designs for both evaluators.
3. Write the estimate to `DECISIONS.md` and **report it to Nina before running
   the full set.**

If the estimate exceeds 20 GPU-hours or Nina's stated budget, stop and propose
cuts (smaller target construct, single evaluator, fewer seeds) rather than
proceeding.

---

## Step 3: Measure what the filter is worth

Outputs go in `results/`.

### Pre-specify before running anything in this step

Write to `ANALYSIS_PLAN.md` and commit it **before** computing any statistic,
so the commit is timestamped ahead of the results:

REVISED 2026-11-09. Step 3 now carries **two co-primary claims**, not one.
BindCraft's Discussion asserts both of the following from box plots with no
interval estimate, and both are separately interval-estimable from this
dataset. Putting intervals on the two sentences the field's pipelines rest on
is the paper.

- **One primary metric.** Use `pae_interface_mean`. It is what the field filters
  on and one of the three the competition selected by, so it is the metric whose
  value is actually in question. Every other metric is secondary and exploratory.
- **One primary population**: the 378 expressed designs.
- **Claim A, binary discrimination.** AUC separating binder from non-binder over
  the 378, bootstrap 95% CI, 10k resamples, and whether that CI excludes 0.5.
  Plus precision at top k for k in {10, 20, 50} against the 14.0% base rate, and
  the calibration curve by metric decile.
- **Claim B, affinity ranking.** Among the **53 binders only**, correlation
  between the metric and KD. Spearman primary, Pearson on log KD secondary,
  bootstrap CI. The 53 KD values span 3.92 log10 decades (1.21 nM to 10 uM), so
  there is real dynamic range. The CI will be roughly plus or minus 0.27 at this
  n (Fisher z, verified). **Report n=53 in the sentence, not a footnote, and
  report the CI rather than hiding it.**
- **Multiplicity.** A and B are co-primary and each carries its own claim, since
  BindCraft asserts both; state that split explicitly rather than treating one as
  exploratory. Everything else, seven metrics plus six baselines, gets
  Benjamini-Hochberg correction and is reported as exploratory.

This costs an hour and it is the difference between a result and a fishing
expedition. Given that the paper's entire argument is about rigour in
evaluation, an unregistered analysis plan would be the obvious thing for a
reviewer to attack.

### Primary analysis: Claim A on the `tested` population

On the 378 expressed designs:

- **AUC** for each metric separating `binds` from `does not bind`, with
  bootstrap 95% CI (10k resamples).
- The headline test is whether that AUC's CI **excludes 0.5**. If it does not,
  the metric carries no residual signal among designs that passed the filter:
  it is a threshold, not a ranker.
- **Precision at top k** for k in {10, 20, 50}: having already filtered, if you
  then kept the top k by this metric, how many bound? Compare to the population's
  own 14.0% base rate. This is the operationally meaningful number.
- **Calibration curve.** Bin by metric decile, plot observed binding rate per
  bin. A ranker gives a monotone curve; a threshold gives a flat one. This
  figure is the clearest statement of the finding either way.

### Claim B: affinity ranking among the binders

Among the 53 binders only, metric against KD. Spearman primary with bootstrap
CI, Pearson on log KD secondary. This is the direct test of BindCraft's "does
not correlate with the interaction affinity." Report n and CI in the sentence.

### Secondary analysis: the truncation picture

REPLACES the deleted `unbiased` arm, which does not exist. Run the two tiers
defined in Step 1: the measured tested-versus-untested comparison on length,
GRAVY, net charge and ESM2 pLL, then the model-based bound reported separately
and labelled as model-based.

### Baselines — non-negotiable

If a cheap, dumb feature matches the confidence metric, that is the story. Run
all of these through the identical AUC / precision@k / calibration machinery:

- sequence length
- mean hydrophobicity (GRAVY)
- net charge at pH 7
- `n_contacts_5A` alone
- `bsa` alone
- a random score (sanity check: AUC ~0.5, CI covering it)
- **the EvoBind2-style adversarial cross-check**: re-predict with a second
  model or a second AF2 mode and use disagreement between the two as the score.
  This is the published mitigation with a 2.5x number attached, it costs far
  less than MD, and if it matches what MD achieves in Step 4 then MD is not
  worth the compute. This is the baseline that matters most. (Running it means
  the deferred Boltz-2 fold comes back onto the critical path, for this subset
  at minimum.)

### Stratification

Report by `design_method` where group sizes allow (n >= 20). Different
generators may fail differently and pooling can hide it.

### Acceptance criteria

- `results/figures/calibration.{png,svg}` — observed binding rate by metric
  decile, on the `tested` population
- `results/figures/score_distributions.{png,svg}` — binders vs non-binders,
  on the `tested` population
- `results/figures/truncation.{png,svg}` — submitted vs tested score
  distribution. **REQUIRED as of 2026-11-09.** Submissions are public (all 1131),
  and this figure replaces the deleted secondary population.
- `results/tables/auc.csv`: metric, n, AUC, CI low, CI high (no arm column)
- `results/tables/affinity_correlation.csv`: Claim B, metric, n, Spearman,
  CI low, CI high
- `results/tables/precision_at_k.csv`
- `results/tables/cross_model_agreement.csv` — AF2 vs Boltz-2 rank
  correlation. CONDITIONAL: only if the deferred Boltz-2 run happened.
- every number traceable to a script and a recorded seed
- `results/README.md`, one paragraph, stating what the numbers say

**Do not editorialise toward the hypothesis.** If interface confidence turns
out to rank well among filter-passing designs, say so plainly. That is a
publishable result too, and it redirects the project rather than ending it.

---

## Step 4: MD on the confident failures

Start only after Step 3 is reviewed. This is the research contribution and the
expensive step.

### Define the sets before looking at any MD output

- **Confident-wrong:** high primary metric, did not bind. Choose the threshold
  by percentile (e.g. top quartile within the `tested` population) and **write
  it down first.**
- **Confident-right control:** high primary metric, did bind, **matched on
  score range.** Without this matched control the MD finding means nothing,
  because any complex looks unstable if you squint at it long enough.

Target ~20 per set, budget permitting.

### Protocol

Start from the Step 2 predicted structures. Explicit solvent, physiological
salt, standard minimise / heat / equilibrate, then production.

**Use the published convention and cite it.** UnibindMD (bioRxiv, Oct 2026)
runs 100-ns all-atom MD validation of designed binder complexes with
GROMACS/CHARMM36m. AlphaDesign used Amber ff14SB (Maier et al 2015) in explicit
solvent for the same purpose. Adopt CHARMM36m at the UnibindMD settings as the
default, with the citation. A cited protocol is not a judgment call a reviewer
can question.

Deviate only with a recorded reason in `DECISIONS.md`. Do not invent settings,
and do not block on obtaining the Palermo lab's protocol.

**Engine and compute target. REVISED 2026-11-09: GROMACS on UCLA Hoffman2, not
Modal.** Arithmetic: the 378-design fold is roughly 20 GPU-hours, while this
triage tier is 40 complexes x 5 replicas = **200 jobs** of 10 ns on a solvated
complex of roughly 50,000 to 80,000 atoms, which is about an order of magnitude
more. Whatever covers the fold will not cover the MD at any plausible rate. A
GROMACS job array on institutional HPC is different engineering from a Modal
fan-out and must not be written twice, so decide before building anything for
Step 4. Step 2's folding stays on Modal as specified and is comfortably within
budget. **Raise this with Nina before any Step 4 engineering.**

**Triage first:** REVISED 2026-11-09 to **10 ns x 5 independent replicas** per
complex, three being the absolute floor. The unit of analysis is the design, not
the replica, so at 20 vs 20 the scalar needs a true AUC near 0.70 before its CI
clears 0.5; replicas shrink per-design measurement error rather than changing n,
and five cost the same as one 50 ns run. Each replica gets its own initial
velocity seed. Discard the first 20% of each run, fixed in advance. Extend to
~100 ns only where triage is ambiguous. See ANALYSIS_PLAN.md for the ICC
escalation rule.

**What this tier may and may not claim.** Neither 10 ns nor 100 ns observes
unbinding: for a micromolar binder k_off puts the complex lifetime in the
millisecond to second range. The admissible claim is that the predicted
interface is not a free energy minimum under CHARMM36m. "The complex
dissociates" is not available at these lengths.

Three replicas is the minimum, not a nicety. A single trajectory crossing a
barrier by luck is how you get a result that does not reproduce, and
within-replica variance underestimates true uncertainty. **Report per-replica
numbers, never only the pooled mean.**

### Modal layout for MD

APPLIES ONLY IF Nina chooses Modal over Hoffman2 for Step 4. Fan out over
`(design_id, replica_index)` tuples with `.starmap()`. Each is an independent
container, so 40 complexes x 5 replicas is **200 jobs**, not the 120 this spec
originally assumed. Separate image with OpenMM; separate `max_containers` cap.

Checkpoint trajectories to the results Volume periodically, not only at the
end, and make each job resumable from its last checkpoint. A 100 ns run that
dies at 90 ns and has to restart is pure waste.

Cost the triage tier the same way as Step 2: run 3 jobs, measure, extrapolate,
report before launching all 120.

### Observables

- interface contact persistence (fraction of initial interchain contacts
  retained, vs time)
- interchain RMSD after aligning on the target only
- binder internal RMSD and radius of gyration (is it unfolding?)
- per-contact lifetime distribution
- whether the binder leaves the intended epitope

### Acceptance criterion

A **single scalar per design**, computed from the trajectories, that separates
confident-wrong from confident-right better than the static metric did. Report
its AUC on the matched sets with per-replica spread.

That scalar is the paper's contribution: not "confidence does not predict
binding" but "here is a cheap dynamical signal that does."

---

## Standing guardrails

**Stop and ask Nina before:**
- fixing the EGFR target construct
- raising `max_containers` above the pilot cap
- running the full 378-design fold after the pilot
- spending more than 20 GPU-hours in one go
- changing the `binds` threshold or the denominator (the arm definitions are
  gone; see the selection-bias section)
- starting Step 4

**Never:**
- use a confidence score computed by anyone else as our prediction
- call the public MMseqs2 API once per design
- pool selection arms into a headline statistic (moot as of 2026-11-09: no
  recoverable arm labels exist, so there is nothing to pool)
- drop designs without logging the count and reason
- report an MD result from a single replica
- tune a threshold after seeing the outcome it affects

**Always:**
- pin dependency versions; record model weights version and GPU type per run
- one script per artifact in `results/`, seeded and rerunnable
- key every output by `design_id` so re-runs resume instead of restarting
- append to `DECISIONS.md` as you go: dated, one line each

---

## Known threats to validity, to state in the paper

1. **Selection bias** toward high `pAE_interaction`. Addressed by making the
   post-filter population the intended one, not by correcting it away. State
   this as a scope limit: the claim is about filter-passing designs.
2. **Single target (EGFR).** A reviewer will say "that is one protein." A
   second target is the strongest available upgrade.
3. **Training-data leakage.** Check whether the EGFR construct and any closely
   related binders predate the model training cutoffs.
4. **Force fields** are weaker for some systems than others, and converged
   sampling on a flawed potential gives a precise wrong answer. Be explicit
   that the Step 4 claim is about sampling, not about the potential.
5. **Expression filtering** removes 22 designs whose structural predictions are
   unexamined. Note that the analysis is conditional on expression.
6. **Multiplicity.** State the pre-specified primary test and that everything
   else is exploratory with BH correction. Point at the timestamped
   `ANALYSIS_PLAN.md` commit.
7. **Positive controls are in the training data,** so they validate the
   pipeline rather than the metric. Say so explicitly.
