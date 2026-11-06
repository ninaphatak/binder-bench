# Decisions log

One line each, dated. Append only.

- 2026-11-05: Step 0 item 0 complete: PRIOR_ART.md written from primary text for BindCraft (Nature 2025), EvoBind2 (Commun Chem 2025), UnibindMD (bioRxiv 2026-10-23).
- 2026-11-05: EvoBind2 "2.5x" in SPEC.md verified against the journal version: 5/13 vs 2/13 μM binders, one target. The 2024 preprint abstract says "tripling" for the same experiment; journal text says 2.5x. Using 2.5x with the counts stated.
- 2026-11-05: Step 4 MD protocol default adopted from UnibindMD Methods verbatim (GROMACS/CHARMM36m, TIP3P, 0.15 M NaCl, 1.0 nm padding, 100 ps NVT + 100 ps NPT restrained, v-rescale 300 K, Parrinello-Rahman 1 bar, 2 fs, LINCS h-bonds, PME 1.2 nm, force-switch vdW).
- 2026-11-05: Deviation from UnibindMD recorded: they run 1 x 100 ns per complex (replica count not stated in Methods); we run 3 x 10 ns triage extended to ~100 ns only where ambiguous. Reason: per-replica reporting is a hard rule; a single trajectory does not reproduce.
- 2026-11-05: Noted for the paper's threats-to-validity section: BindCraft Extended Data Fig. 7 already states qualitatively that interface pTM is "a powerful binary predictor of binding activity" but "does not correlate with the interaction affinity." Our contribution is the interval estimate and calibration curve, not the direction of the claim.
- 2026-11-05: git repo initialised, initial commit 0d36ff3 on main (CLAUDE.md, SPEC.md, PRIOR_ART.md, DECISIONS.md, .gitignore). Needed before Step 3: ANALYSIS_PLAN.md must be committed ahead of any statistic and the timestamp is the point.
- 2026-11-05: Step 0 item 1 complete: enrich-bench inventoried at ~/molecules/enrich-bench, read-only, not refactored, not wired in.
- 2026-11-05: OPEN QUESTION for Nina: enrich-bench/PROJECT.md carries a teaching contract (DRIVE mode default; Claude forbidden to write AUC/precision-at-k/EF, filter thresholds, interface-contact and contact-persistence code). SPEC.md Steps 3 and 4 ask for exactly that code. Scope not resolved; not assumed to carry over.
- 2026-11-05: OPEN QUESTION for Nina: enrich-bench total budget is [redacted]. Unclear whether binder-bench draws on the same pot. Item 6 needs her own figure before any cost extrapolation.
- 2026-11-05: RESOLVED by Nina: the enrich-bench teaching contract (DRIVE mode, forbidden-code list) does NOT apply to binder-bench. Claude writes the pipeline, including the Step 3 metrics and the Step 4 scalar, as SPEC.md specifies.
- 2026-11-05: RESOLVED by Nina: the no-em-dashes style rule DOES apply here. PRIOR_ART.md and DECISIONS.md rewritten without them.
