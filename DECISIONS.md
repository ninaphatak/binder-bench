# Decisions log

One line each, dated. Append only.

- 2026-11-05 — Step 0 item 0 complete: PRIOR_ART.md written from primary text for BindCraft (Nature 2025), EvoBind2 (Commun Chem 2025), UnibindMD (bioRxiv 2026-10-23).
- 2026-11-05 — EvoBind2 "2.5x" in SPEC.md verified against the journal version: 5/13 vs 2/13 μM binders, one target. The 2024 preprint abstract says "tripling" for the same experiment; journal text says 2.5x. Using 2.5x with the counts stated.
- 2026-11-05 — Step 4 MD protocol default adopted from UnibindMD Methods verbatim (GROMACS/CHARMM36m, TIP3P, 0.15 M NaCl, 1.0 nm padding, 100 ps NVT + 100 ps NPT restrained, v-rescale 300 K, Parrinello-Rahman 1 bar, 2 fs, LINCS h-bonds, PME 1.2 nm, force-switch vdW).
- 2026-11-05 — Deviation from UnibindMD recorded: they run 1 x 100 ns per complex (replica count not stated in Methods); we run 3 x 10 ns triage extended to ~100 ns only where ambiguous. Reason: per-replica reporting is a hard rule; a single trajectory does not reproduce.
- 2026-11-05 — Noted for the paper's threats-to-validity section: BindCraft Extended Data Fig. 7 already states qualitatively that interface pTM is "a powerful binary predictor of binding activity" but "does not correlate with the interaction affinity." Our contribution is the interval estimate and calibration curve, not the direction of the claim.
