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

## THE SELECTION-BIAS PROBLEM — read before touching the data

Competitors submitted far more designs than Adaptyv could assay. Round 2 tested
400. Selection into that 400 was **not random**: it was split into arms, one
of which took the best designs **by `pAE_interaction`**, the very metric under
test. Adaptyv's own writeup notes the pseudo-random arm had a different
affinity range from the selected arm, so the arms demonstrably differ.

Consequences:

1. **Range restriction.** The observable score axis is a narrow slice near the
   good end. A score needs variance to rank anything; restricting the range
   attenuates AUC toward 0.5 even for a metric that is informative across the
   full range.
2. **The bias runs in the direction of the hypothesis.** A naive pooled AUC
   would come out weak, and a reviewer answers: of course it did, you only
   tested designs that already scored well. Unanswerable after the fact.
3. **Collider effects.** Conditioning on "got tested" when testing depended on
   pAE induces spurious associations inside the tested set. Designs admitted
   through a non-pAE arm got in for some other reason, which may itself
   correlate with binding.
4. **The 14% success rate is conditional.** It is success among designs
   pre-enriched for high confidence, not the field's base rate. Do not quote it
   as "binder design succeeds 14% of the time."

**How this spec handles it:** by making the restricted population the
*intended* population (the residual-information question above), and treating
the full-range question as a clearly-labelled secondary analysis on the
pseudo-random arm only.

`selection_arm` must be recoverable per design. **If it cannot be recovered
from the competition repo, stop and report — this is a blocker, not something
to work around.**

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
   `selection_arm` labels exist. **Report each arm's name, its design count,
   and its binder count.**
3. Inventory **Round 3** as well (the round that characterised 601 proteins
   against EGFR): is the data public in the same shape, what was its selection
   rule, is the assay comparable to Round 2? Do not download predictions. Just
   establish whether it can be pooled or analysed in parallel.
4. Check whether either repo ships predicted structures or confidence metrics.
   Note them as a comparison point, **not** as our predictions.
5. Confirm `modal` is installed and authenticated.
6. Confirm Nina's **Modal budget** in dollars and record it in
   `DECISIONS.md`. New-account credits do not cover 378 GPU folds. Every cost
   estimate later in this spec is checked against that number, not against a
   vague sense of "reasonable."
7. **THE GATE.** Count the binders in Round 2's pAE-selected arm.
   - **25 or more:** proceed with Round 2 alone, as specified.
   - **Fewer than 25:** stop. An AUC on fewer than roughly 25 positives has an
     interval too wide to exclude anything, and an underpowered null is the one
     outcome with no publishable reading in either direction. Report the count
     alongside Round 3's numbers and let Nina decide: pool the rounds, analyse
     them in parallel, or change the question.

## Step 1: Build the dataset

Produce `data/designs.parquet`:

| column | notes |
|---|---|
| `design_id` | |
| `sequence` | binder amino acid sequence |
| `design_method` | self-reported by competitor; messy, keep verbatim |
| `selection_arm` | pAE-selected / rule-based / pseudo-random (see above) |
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

**Analysis populations.** Define two, up front:
- **Primary (`filter_pass`)**: the pAE-selected arm. The post-filter
  population. This is where the residual-information question lives.
- **Secondary (`unbiased`)**: the pseudo-random arm. Smaller n, wider CIs, but
  the closest thing to a full-range sample.

### Also build: the truncation picture

If all *submitted* designs are public (not just the tested 400), record their
self-reported or recomputed pAE distribution. Plot submitted vs tested. This
quantifies exactly how much of the range is missing instead of leaving it as
prose. Do this even though the reframed question does not depend on it — it is
the figure that pre-empts the reviewer.

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

Before folding any competition design, fold **3 to 5 known EGFR binders** with
measured affinity against your chosen construct. Candidates: cetuximab-derived
fragments, or the EGF/TGF-alpha-derived designs Adaptyv flagged as landing near
1e-6 M. They must come out with strong interface scores.

If they do not, something upstream is wrong, most likely the target construct,
possibly chain ordering or the MSA. This control costs minutes and is the only
cheap way to catch a wrong construct before it eats the whole GPU budget. **Do
not proceed to the pilot until it passes.**

Caveat to state in the paper: known binders like cetuximab are in the models'
training data, so they will score well partly from memorisation. The control
validates the *pipeline* (construct, chains, MSA), not the *metric*. Do not
present it as evidence the metric works.

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

- **One primary metric.** Use `pae_interface_mean`. It is what the field filters
  on and what the competition selected by, so it is the metric whose value is
  actually in question. Every other metric is secondary and exploratory.
- **One primary population**: the pAE-selected arm.
- **One primary statistic**: AUC, and whether its 95% CI excludes 0.5.
- **Multiplicity.** Seven metrics across two arms plus six baselines is roughly
  thirty tests. If one comes out significant after that, it is noise, not a
  finding. Secondary metrics get Benjamini-Hochberg correction and are
  reported as exploratory. Only the pre-specified primary test carries the
  claim.

This costs an hour and it is the difference between a result and a fishing
expedition. Given that the paper's entire argument is about rigour in
evaluation, an unregistered analysis plan would be the obvious thing for a
reviewer to attack.

### Primary analysis: residual information within `filter_pass`

On the pAE-selected arm only:

- **AUC** for each metric separating `binds` from `does not bind`, with
  bootstrap 95% CI (10k resamples).
- The headline test is whether that AUC's CI **excludes 0.5**. If it does not,
  the metric carries no residual signal among designs that passed the filter:
  it is a threshold, not a ranker.
- **Precision at top k** for k in {10, 20, 50}: having already filtered, if you
  then kept the top k by this metric, how many bound? Compare to the arm's own
  base rate. This is the operationally meaningful number.
- **Calibration curve.** Bin by metric decile, plot observed binding rate per
  bin. A ranker gives a monotone curve; a threshold gives a flat one. This
  figure is the clearest statement of the finding either way.

### Secondary analysis: the full-range question

Same statistics on the `unbiased` (pseudo-random) arm, reported separately and
explicitly labelled as the smaller, wider-CI, full-range estimate. Never pool
the arms into one headline number.

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
  decile, per arm
- `results/figures/score_distributions.{png,svg}` — binders vs non-binders,
  faceted by arm
- `results/figures/truncation.{png,svg}` — submitted vs tested score
  distribution. OPTIONAL: only if submissions are public, and only once the
  required figures are done. The reframed question does not depend on it.
- `results/tables/auc.csv` — metric, arm, n, AUC, CI low, CI high
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
  by percentile (e.g. top quartile within `filter_pass`) and **write it down
  first.**
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

Engine: OpenMM or GROMACS on Modal, or the lab's own stack on institutional HPC
if that is cheaper at this trajectory volume. Raise the choice with Nina.

**Triage first:** 10 ns x 3 independent replicas per complex. This catches
gross interface failure cheaply. Extend to ~100 ns only where triage is
ambiguous.

Three replicas is the minimum, not a nicety. A single trajectory crossing a
barrier by luck is how you get a result that does not reproduce, and
within-replica variance underestimates true uncertainty. **Report per-replica
numbers, never only the pooled mean.**

### Modal layout for MD

Fan out over `(design_id, replica_index)` tuples with `.starmap()`. Each is an
independent container, so 40 complexes x 3 replicas is 120 jobs. Separate image
with OpenMM; separate `max_containers` cap.

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
- changing the `binds` threshold, the denominator, or the arm definitions
- starting Step 4

**Never:**
- use a confidence score computed by anyone else as our prediction
- call the public MMseqs2 API once per design
- pool the selection arms into a headline statistic
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
