"""The frozen ColabFold configuration, and the command line that expresses it.

Every flag here was read from `colabfold_batch --help` as shipped in the pinned
image (ColabFold 1.6.2), not from documentation or memory. The defaults quoted
in comments are that version's defaults.

The point of Step 2 is a UNIFORM re-prediction. Competitors and the competition
organisers used different models, versions and configurations, and inheriting
that inconsistency would mean measuring version noise instead of the thing we
care about. So this configuration is fixed once, recorded, and applied to every
design including the controls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Frozen configuration
# ---------------------------------------------------------------------------

#: AF2-multimer v3. Set explicitly rather than relying on "auto", which would
#: pick multimer_v3 for complexes anyway but leaves the record ambiguous.
MODEL_TYPE = "alphafold2_multimer_v3"

#: All five models. The primary estimator is the MEAN across them, not the best
#: of them. See metrics.aggregate_across_models for why.
NUM_MODELS = 5

#: Matches the competition's three recycles, so that difference is not a
#: confound when comparing our numbers to their published ones.
NUM_RECYCLE = 3

#: One seed, not the competition's three. Seeds are replicas of the same
#: parameterisation, whereas the five models are different parameterisations.
#: Fifteen predictions per design would have cost several times the budget, and
#: averaging the five models already captures most of the variance reduction.
NUM_SEEDS = 1

#: Fixed and recorded. Everything in results/ must be rerunnable from a seed.
RANDOM_SEED = 0

#: Rank by ipTM explicitly. "auto" would also choose ipTM for multimers, but
#: the record should not depend on knowing that. Note that the rank_001 value
#: is SECONDARY in this project, kept only for comparability with the
#: competition's published numbers.
RANK_BY = "iptm"

#: The binder is de novo and single-sequence, so cross-chain MSA pairing has
#: nothing to pair. The default is "unpaired_paired", which would build a
#: paired block of one row at some cost. Flagged as a pilot check: fold one
#: design both ways and confirm the interface metric does not move materially.
PAIR_MODE = "unpaired"

#: Relaxation off. It improves side-chain geometry at the cost of runtime and
#: would make the structures a function of an OpenMM minimisation as well as of
#: AF2. Step 4 does its own explicit-solvent MD from these coordinates.
NUM_RELAX = 0

#: Templates off. This is a deliberate DIVERGENCE from the competition, which
#: ran with templates. Two reasons. Templates would leak solved EGFR complex
#: structures into the prediction, which is a training-data leakage pathway for
#: exactly the binders most likely to resemble known ones. And ColabFold's own
#: help warns that --templates "can result in the MSA server being queried with
#: A3M input", which would defeat the single-cached-MSA strategy.
USE_TEMPLATES = False

#: Leave at the version default of 100. The flag computes models only until
#: pLDDT or pTM exceeds the threshold, so a lower value would silently run
#: FEWER than five models and quietly break the mean-across-models estimator.
#: The parser asserts that five models were actually produced.
STOP_AT_SCORE = 100

#: Deterministic ordering. ColabFold's default is to sort by length, which
#: reduces JAX recompilation when a batch has mixed lengths. We keep that
#: default deliberately: within a chunk it is a real saving, and it does not
#: affect any individual prediction.
SORT_QUERIES_BY = "length"


@dataclass(frozen=True)
class FoldConfig:
    """Everything that could change a number, in one recorded object."""

    model_type: str = MODEL_TYPE
    num_models: int = NUM_MODELS
    num_recycle: int = NUM_RECYCLE
    num_seeds: int = NUM_SEEDS
    random_seed: int = RANDOM_SEED
    rank_by: str = RANK_BY
    pair_mode: str = PAIR_MODE
    num_relax: int = NUM_RELAX
    use_templates: bool = USE_TEMPLATES
    stop_at_score: int = STOP_AT_SCORE
    sort_queries_by: str = SORT_QUERIES_BY

    def as_record(self) -> dict:
        return asdict(self)


def build_command(
    input_path: Path | str,
    output_dir: Path | str,
    weights_dir: Path | str,
    config: FoldConfig | None = None,
) -> list[str]:
    """Construct the colabfold_batch argument list.

    ``input_path`` may be a single A3M file or a DIRECTORY of them. A directory
    is strongly preferred for a chunk of designs: ColabFold then sorts by
    length and reuses its compiled kernels across the batch, so both the weight
    loading and the JAX compilation are amortised rather than paid per design.

    Returns a list, never a shell string, so no argument can be
    reinterpreted by a shell.
    """
    cfg = config or FoldConfig()

    cmd = [
        "colabfold_batch",
        str(input_path),
        str(output_dir),
        # Weights read from the staged Volume. Without this ColabFold falls back
        # to its internal default directory and would re-download 3.8 GiB into
        # every container.
        "--data",
        str(weights_dir),
        "--model-type",
        cfg.model_type,
        "--num-models",
        str(cfg.num_models),
        "--num-recycle",
        str(cfg.num_recycle),
        "--num-seeds",
        str(cfg.num_seeds),
        "--random-seed",
        str(cfg.random_seed),
        "--rank",
        cfg.rank_by,
        "--pair-mode",
        cfg.pair_mode,
        "--num-relax",
        str(cfg.num_relax),
        "--stop-at-score",
        str(cfg.stop_at_score),
        "--sort-queries-by",
        cfg.sort_queries_by,
    ]

    # Templates are off by DEFAULT in this version, so the flag is simply never
    # added. Asserting rather than adding a negative flag, because there is no
    # --no-templates to add.
    if cfg.use_templates:
        cmd.append("--templates")

    # Deliberately NOT passed: --overwrite-existing-results. Its name and its
    # help text contradict each other in 1.6.2, where the help reads "Do not
    # recompute results, if a query has already been predicted." A flag whose
    # meaning is ambiguous has no place in a run we need to reproduce.
    # Idempotency is handled one level up, by checking the results Volume for a
    # per-design output before folding it at all.

    return cmd


def assert_no_ambiguous_flags(cmd: list[str]) -> None:
    """Guard against flags that would silently change what gets computed."""
    forbidden = {
        "--overwrite-existing-results": (
            "name and help text contradict each other in ColabFold 1.6.2"
        ),
        "--templates": (
            "would leak solved EGFR complexes into predictions and may query "
            "the MSA server with A3M input"
        ),
        "--amber": "relaxation is off; Step 4 does its own MD",
        "--use-dropout": "would make predictions non-deterministic",
        "--initial-guess": (
            "would seed the prediction with designed coordinates, which is a "
            "different experiment"
        ),
    }
    present = [f for f in forbidden if f in cmd]
    if present:
        raise ValueError(
            "Refusing to run. Forbidden flags present: "
            + "; ".join(f"{f} ({forbidden[f]})" for f in present)
        )


def expected_prediction_count(config: FoldConfig | None = None) -> int:
    """How many predictions one design should yield.

    Used to verify that ColabFold actually produced what we asked for.
    --stop-at-score can silently cut this short, which would break the
    mean-across-models estimator while leaving plausible numbers behind.
    """
    cfg = config or FoldConfig()
    return cfg.num_models * cfg.num_seeds
