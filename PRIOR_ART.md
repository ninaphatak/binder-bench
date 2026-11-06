# Prior art (Step 0, item 0)

Three required reads. Each paragraph ends with an explicit sentence on what
Experiment 01 does that the work does not. Field vocabulary: the target
population is **adversarial designs**, meaning designs with high in-silico
confidence that do not bind.

Sources were read as primary text (journal full text or preprint full text),
not from secondary summaries. Quotations are verbatim; numbers below were read
off the source, not recalled.

---

## 1. BindCraft (Pacesa et al., *Nature* 646(8084):483-492, 2025)

DOI [10.1038/s41586-025-09429-6](https://www.nature.com/articles/s41586-025-09429-6);
open-access text at [PMC12507698](https://pmc.ncbi.nlm.nih.gov/articles/PMC12507698/).

BindCraft designs binders by hallucinating against AF2 weights and then
filtering on a fixed panel of interface metrics (Methods defaults: complex
pLDDT > 0.8, interface pTM > 0.5, interface pAE < 0.35, Rosetta shape
complementarity > 0.60, at least 3 interface hydrogen bonds, fewer than 4
unsatisfied interface hydrogen bonds, binder surface hydrophobicity < 35%,
bound-vs-unbound binder RMSD < 3.5 Angstrom, fewer than three interface lysines
and methionines), reporting 10 to 100% experimental success across targets. The
analysis closest to our Step 3 is **Extended Data Fig. 7, "Benchmarking
prediction accuracy across design pipelines,"** in which "experimentally
validated binders and non-binders from previously published binder design
pipelines have been repredicted" through the BindCraft prediction pipeline under
AF2 monomer (panel a) and multimer (panel b) weights. The caption states that
"EvoPro and RFdiff designs have been already prefiltered by AF2 monomer in their
respective publications, and indicate the presence of false positives," while
RIFdock and MaSIF-seed designs were not prefiltered. In other words the authors
are explicit that the same selection bias we have to handle is present in their
comparison set. Panel c compares AF2 and AlphaFold3 interface pTM on BindCraft
designs, where AlphaFold3 "still found a large proportion of false positive
predictions." The load-bearing claim for us is in the Discussion: interface pTM
"has emerged as a powerful binary predictor of binding activity" but "does not
correlate with the interaction affinity." That is the threshold-not-ranker
hypothesis, asserted qualitatively from box plots. The paper reports no AUC, no
confidence interval and no calibration curve for that separation, and the
comparison pools designs from pipelines with different and partly unknown
selection rules.

**What Experiment 01 does that BindCraft does not:** it turns that qualitative
Discussion sentence into a pre-registered, interval-estimated test. One AUC with
a bootstrap 95% CI for `pae_interface_mean`, on one uniformly re-predicted
dataset with measured affinities, computed *within* a single known selection arm
rather than pooled across pipelines, with a calibration curve and
precision-at-k that distinguish a threshold from a ranker instead of inferring
it from box-plot overlap.

---

## 2. EvoBind2 (Li, Vlachos and Bryant, *Commun. Chem.*, 2025)

"Design of linear and cyclic peptide binders from protein sequence
information." DOI [10.1038/s42004-025-01601-3](https://doi.org/10.1038/s42004-025-01601-3);
open access at [PMC12280060](https://europepmc.org/article/MED/40691301);
preprint [10.1101/2024.06.20.599739](https://www.biorxiv.org/content/10.1101/2024.06.20.599739v2.full).

EvoBind2 designs linear and cyclic peptide binders blind from target sequence
alone, with no specified binding site and no specified binder length, by 1000
rounds of in-silico directed evolution against an AF2 loss (Eq. 1). It then
applies the check that matters to us. This paper is the source of the field's
term: "it is possible to generate sequences that achieve high in silico
confidence (low loss, Eq. 1) but fail to bind in reality. These deceptive
sequences, which 'trick' EvoBind2, are considered **adversarial designs**." The
mitigation is an orthogonal second predictor, a modified AlphaFold-multimer run
with receptor MSA and single-sequence peptide giving a second loss (Eq. 2), on
the reasoning that "it is unlikely that two independently trained structure
prediction networks, using different data, would both agree on an incorrect
solution." To test it they deliberately sampled the disagreement region: for
each peptide length, the lowest-EvoBind2-loss sequence subject to AFM loss > 1,
then measured affinity by SPR against semi-synthetic ribonuclease (PDB 1SSC).
Result: "only two of the sequences in the adversarial selection displayed
micromolar affinity compared to five when both losses were low... improving the
success rate by **2.5x**." Two caveats to carry forward. First, the 2.5x rests
on **5/13 versus 2/13**, thirteen peptides per selection against one target, so
the interval around that fold-change is very wide, and the preprint abstract
described the same experiment as "tripling the success rate." Second, 2 of 13
adversarial-selection peptides still bound, so the check is not clean in either
direction. They also report that among the seven top-selection peptides with
measurable affinity, AFM loss and EvoBind2 pLDDT correlate with Kd at Spearman
R = 0.96 and 0.79, an n = 7 correlation on the selected end of the range.

**What Experiment 01 does that EvoBind2 does not:** EvoBind2 demonstrates that
cross-model disagreement enriches for real binders on 13 peptides against one
target. We run that same disagreement score as **one baseline among six**,
through identical AUC, precision-at-k and calibration machinery, on a larger
protein-binder set against EGFR with pre-registered multiplicity control, and we
use it as the bar the Step 4 MD signal must beat, since if a second fold matches
what MD achieves then MD is not worth the compute. We are also testing the
*confidence metric itself* as a ranker, which EvoBind2 assumes rather than
measures.

---

## 3. UnibindMD (Wu, Huang, Fu, Wang and Fu, bioRxiv, 23 Oct 2026)

"UnibindMD: An Integrated AI and Molecular Dynamics Platform for De Novo
Peptide Binder Design." DOI
[10.64898/2026.10.20.760329](https://www.biorxiv.org/content/10.64898/2026.10.20.760329v1.full),
CC BY-NC-ND, from ShenZhen UnibindMD Computation Biotech Co., Ltd.

A five-stage pipeline paper: BindCraft design, then independent structure
prediction and confidence assessment (AlphaFold3-class or Boltz-2), then
affinity prediction (PRODIGY), then developability (ADMET-AI), then all-atom MD
validation (GROMACS/CHARMM36m). It is motivated by exactly our premise, stated
in their abstract: "confidence metrics such as ipTM and pLDDT are
self-consistent but not independent, so a design predicted to be confident may
still be physically unstable." Applied to three Andes virus intervention points
(Gn-Gc glycoprotein complex; L-protein endonuclease domain, PDB 6Q99; host
VEGFR2 kinase, PDB 2XIR), nine BindCraft campaigns over 4,475 trajectories
yielded 464 accepted binders (acceptance 2.3 to 67.6%), top candidates at
ipTM 0.94 and pLDDT 0.94, and three selected leads at PRODIGY dG of -9.1 to
-11.1 kcal/mol. The MD protocol we are adopting in Step 4, verbatim from their
Methods: GROMACS 2021.3, CHARMM36m, TIP3P water, rhombic dodecahedron box with
at least 1.0 nm padding, neutralised then 0.15 M NaCl, steepest-descent
minimisation, 100 ps NVT then 100 ps NPT with position restraints on protein
heavy atoms, then **100 ns unrestrained NPT production** with v-rescale
thermostat at 300 K, Parrinello-Rahman barostat at 1 bar, 2 fs timestep, LINCS
on hydrogen bonds, PME with 1.2 nm cutoff, force-switched van der Waals, on a
single V100. Observables, averaged over the last 80 ns: receptor and peptide
C-alpha RMSD, interface heavy-atom contacts at 4.5 Angstrom and contact
retention, polar contacts, salt bridges, hydrogen bonds, buried SASA, peptide
radius of gyration and end-to-end distance, DSSP secondary structure, PCA
variance. All three leads "remained stably bound," with receptor CRMSD at or
below 2.6 Angstrom, at least 99.5% of frames retaining more than half of initial
interface contacts, and 75 to 82% helical content. Three things are important
about this paper's standing. It is a **pipeline paper, not a benchmark**: every
design it validates is one its own MD stage passed, so there is no negative set
and no measurement of whether the MD stage discriminates. It reports **no
wet-lab data at all**, stating that "All results are computational and await
experimental validation." And the number of independent replicas per complex is
**not stated**; the Methods read as a single 100 ns trajectory per complex,
which our own hard rule on never reporting a single replica forbids us from
copying.

**What Experiment 01 does that UnibindMD does not:** UnibindMD applies MD to
designs it has already decided to keep. We apply it to a **matched negative
set**, meaning adversarial designs (high confidence, measured non-binding)
against score-matched confirmed binders, with experimental labels on both sides,
and we report a single scalar's AUC with per-replica spread across 3 replicas
rather than a stability narrative from one trajectory. Their protocol is our
cited default; their validation design is the gap we are filling.

### Deviation from UnibindMD to log in DECISIONS.md

Their protocol as published is 1 x 100 ns. SPEC Step 4 triage is 3 x 10 ns per
complex, extended to about 100 ns only where ambiguous. This is a deliberate
deviation: replicas over length, because a single trajectory crossing a barrier
by luck does not reproduce. Force field, water model, salt, minimisation,
equilibration, integrator and cutoff settings are taken unchanged from
UnibindMD so the comparison stays citable.
