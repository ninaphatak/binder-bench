"""Parse a ``colabfold_batch`` output directory into one record per AF2 model.

Pure: numpy and the standard library only. No Modal, no network, importable and
testable offline.

Why this module is not a one-liner
----------------------------------
The headline metric of this project is the MEAN across all five AlphaFold
models, with the spread across models recorded separately (see
``metrics.aggregate_across_models``). The competition we compare against read
its numbers off ``rank_001``, the best of five, which is an optimistic
estimator: a design only has to fool one network out of five to post a good
score. That makes two things load-bearing here that a casual parser gets wrong:

1. Every model must be found. Averaging over three models while believing it
   was five silently shifts the primary estimator. Every count mismatch raises.
2. Each record must say WHICH AlphaFold model produced it, so model identity
   can be carried into the analysis rather than collapsed into a rank.

Ground truth: ColabFold 1.6.2 source (``colabfold/batch.py``), read directly,
not from memory. Line numbers below refer to that file.

Output filename convention
--------------------------
``file_manager`` (batch.py:377-389) writes every per-prediction file as::

    {jobname}_{kind}_{tag}.{ext}

where ``tag`` is built at batch.py:465::

    tag = f"{model_type}_{model_name}_seed_{seed:03d}"

``model_type`` is one of the ``--model-type`` values after normalisation
(batch.py:1687-1703), e.g. ``alphafold2_multimer_v3`` for complexes, and
``model_name`` is literally ``model_1`` .. ``model_5`` (batch.py:359 compares
``model_name == "model_1"``). ``seed`` iterates
``range(random_seed, random_seed + num_seeds)`` (batch.py:434).

After all predictions finish, the reranking block (batch.py:611-643) renames
every recorded file, prefixing the tag with the rank::

    new_tag = f"rank_{(n+1):03d}_{tag}"
    new_file = result_dir / f"{prefix}_{x}_{new_tag}.{ext}"

So the files this parser reads are::

    {jobname}_scores_rank_001_alphafold2_multimer_v3_model_4_seed_000.json
    {jobname}_unrelaxed_rank_001_alphafold2_multimer_v3_model_4_seed_000.pdb
    {jobname}_relaxed_rank_001_alphafold2_multimer_v3_model_4_seed_000.pdb

Consequences we rely on:

* Model identity IS recoverable from the filename: the ``model_N`` component of
  the tag is the AlphaFold model number, independent of rank. Rank is just the
  ordering ColabFold applied afterwards.
* ``rank`` is assigned by ``mean_scores.argsort()[::-1]`` over
  ``result["ranking_confidence"]`` (batch.py:617), the metric chosen by
  ``--rank`` (``auto`` means ``multimer`` for complexes, i.e. ipTM-weighted, and
  ``plddt`` for monomers; batch.py:1292-1295). ``rank_001`` is therefore a
  selected maximum, which is exactly what we refuse to use as primary.
* Relaxation is applied only to the top ``num_relax`` ranks (batch.py:622-632),
  so a ``_relaxed_`` file may exist for some models and not others. Both are
  distinguishable by the ``relaxed``/``unrelaxed`` kind in the filename.
* Predicted structures are PDB only in 1.6.2. The ``{id}.cif`` files come from
  ``put_mmciffiles_into_resultdir`` (batch.py:1139), which stages *template*
  mmCIFs into ``{result_dir}/templates``; they are not predictions.
* ``--save-recycles`` writes extra files with ext ``r{recycles}.pdb``
  (batch.py:503), i.e. ``..._seed_000.r3.pdb``. They are renamed with the rank
  too, so a glob for ``*.pdb`` would pick up intermediate recycles. This parser
  builds exact filenames instead of globbing for that reason.

Scores JSON schema
------------------
Written at batch.py:579-596, so the keys are exactly:

* ``plddt``: list of float, length = sum of chain lengths (padding stripped by
  ``plddt[:seq_len]``, batch.py:578-580), rounded to 2 decimals. 0-100 scale.
* ``pae``: nested list, ``seq_len x seq_len``, rounded to 2 decimals. In
  Angstroms. Present only when the model returned
  ``predicted_aligned_error`` (batch.py:581; true for ptm and multimer model types).
* ``max_pae``: float, the max of that matrix.
* ``ptm``, ``iptm``: float rounded to 2 decimals. ``iptm`` is dropped for
  single-chain predictions (``if not is_complex: result.pop("iptm")``,
  batch.py:536). Both are written INSIDE the ``if "predicted_aligned_error" in
  result`` branch (batch.py:581-589), so a scores file with no ``pae`` also has
  no ``ptm`` and no ``iptm``: the three stand or fall together.
* extra ``actifptm`` style keys only when ``--calc-extra-ptm`` was passed.

NOTE the 2-decimal rounding: it is in the file, not something we can undo. It
is also why ties in ipTM between models are possible.

``{jobname}_predicted_aligned_error_v1.json`` is NOT a per-model file. It is
written from ``scores[0]`` (batch.py:1639-1643), i.e. the rank_001 model only,
in AlphaFold-DB format (keys ``predicted_aligned_error``,
``max_predicted_aligned_error``). Reading PAE from it would silently give the
best-of-five matrix for every model. This parser never reads it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metrics import interface_metrics

# Filename pieces, from file_manager.get (batch.py:387).
SCORES_KIND = "scores"
UNRELAXED_KIND = "unrelaxed"
RELAXED_KIND = "relaxed"

# tag after reranking: rank_{n:03d}_{model_type}_{model_name}_seed_{seed:03d}
_TAG_RE = re.compile(
    r"^rank_(?P<rank>\d+)_"
    r"(?P<model_type>.+)_"
    r"model_(?P<model_number>\d+)_"
    r"seed_(?P<seed>\d+)$"
)

# AlphaFold ships five parameter sets; --num-models and --model-order both
# index into them (batch.py:1926, "--model-order", default="1,2,3,4,5").
VALID_MODEL_NUMBERS = (1, 2, 3, 4, 5)


class ColabFoldParseError(ValueError):
    """Raised instead of returning a partially parsed result.

    Always carries the consequence of the failure, not just the symptom,
    because the dangerous outcome here is a number that looks fine.
    """


@dataclass(frozen=True)
class ModelPrediction:
    """One AlphaFold model's prediction for one job and one seed.

    ``model_number`` is the AlphaFold model identity (1-5), recovered from the
    filename, NOT the rank. ``rank`` is ColabFold's post-hoc ordering and is
    kept only so we can reproduce the competition's rank_001 number for
    comparison.
    """

    model_number: int
    seed: int
    rank: int
    model_type: str
    tag: str
    plddt: np.ndarray
    pae: np.ndarray | None
    iptm: float | None
    ptm: float | None
    max_pae: float | None
    scores_path: Path
    structure_path: Path
    structure_is_relaxed: bool

    @property
    def model_id(self) -> str:
        """Stable identifier for this prediction, rank excluded on purpose."""
        return f"model_{self.model_number}_seed_{self.seed:03d}"

    @property
    def complex_length(self) -> int:
        return int(self.plddt.shape[0])

    def confidence(self) -> dict[str, float]:
        """ipTM and pTM in the key names ``aggregate_across_models`` reads."""
        out: dict[str, float] = {}
        if self.iptm is not None:
            out["iptm"] = float(self.iptm)
        if self.ptm is not None:
            out["ptm"] = float(self.ptm)
        return out


@dataclass(frozen=True)
class ColabFoldJob:
    """Everything one ``colabfold_batch`` job produced, per model."""

    jobname: str
    result_dir: Path
    predictions: tuple[ModelPrediction, ...]
    model_type: str
    config: dict | None

    @property
    def n_models(self) -> int:
        return len(self.predictions)

    @property
    def complex_length(self) -> int:
        return self.predictions[0].complex_length

    @property
    def model_numbers(self) -> tuple[int, ...]:
        return tuple(p.model_number for p in self.predictions)

    def by_model_number(self, model_number: int) -> ModelPrediction:
        for p in self.predictions:
            if p.model_number == model_number:
                return p
        raise ColabFoldParseError(
            f"No prediction for AlphaFold model {model_number} in job "
            f"{self.jobname!r}; found models {list(self.model_numbers)}."
        )

    def rank001(self) -> ModelPrediction:
        """The prediction ColabFold ranked first.

        Secondary and for comparability only. The primary estimator is the mean
        across all models; see metrics.aggregate_across_models.
        """
        return min(self.predictions, key=lambda p: p.rank)


# ---------------------------------------------------------------------------
# config.json, written by run() at batch.py:1341-1384
# ---------------------------------------------------------------------------


def read_run_config(result_dir: str | Path) -> dict | None:
    """Return the parsed ``config.json`` for the run, or None if absent.

    ColabFold writes it once per output directory, before predicting, and it
    records ``num_models``, ``num_seeds``, ``model_order``, ``model_type``,
    ``rank_by``, ``stop_at_score`` and the colabfold version. We use it to
    cross-check what the caller claims to expect.
    """
    path = Path(result_dir) / "config.json"
    if not path.is_file():
        return None
    try:
        with path.open("r") as handle:
            config = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ColabFoldParseError(
            f"{path} is not valid JSON ({exc}). It is written by ColabFold "
            f"before prediction starts, so a corrupt one means the run was "
            f"interrupted or the directory is not a colabfold_batch output. "
            f"Refusing to guess what was requested."
        ) from exc
    if not isinstance(config, dict):
        raise ColabFoldParseError(
            f"{path} does not contain a JSON object, got {type(config).__name__}."
        )
    return config


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _parse_tag(tag: str, filename: str) -> dict:
    match = _TAG_RE.match(tag)
    if match is None:
        if not tag.startswith("rank_"):
            raise ColabFoldParseError(
                f"{filename!r} has tag {tag!r}, which carries no rank prefix. "
                f"ColabFold renames every output to rank_NNN_... only after all "
                f"models for the job have finished (batch.py:636), so an "
                f"un-ranked scores file means the job died part way through. "
                f"Its models are not a complete set and must not be averaged."
            )
        raise ColabFoldParseError(
            f"{filename!r} has tag {tag!r}, which does not match ColabFold "
            f"1.6.2's rank_NNN_{{model_type}}_model_N_seed_NNN convention. "
            f"Without the model_N component the AlphaFold model identity is "
            f"unknown, so per-model records cannot be built."
        )
    model_number = int(match.group("model_number"))
    if model_number not in VALID_MODEL_NUMBERS:
        raise ColabFoldParseError(
            f"{filename!r} claims AlphaFold model number {model_number}; only "
            f"{list(VALID_MODEL_NUMBERS)} exist. The filename convention has "
            f"changed or the file is not a ColabFold output."
        )
    return {
        "rank": int(match.group("rank")),
        "model_type": match.group("model_type"),
        "model_number": model_number,
        "seed": int(match.group("seed")),
    }


def _load_scores(path: Path) -> dict:
    try:
        with path.open("r") as handle:
            scores = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ColabFoldParseError(
            f"{path.name} is not valid JSON ({exc}). A truncated scores file "
            f"usually means the container was killed mid-write; the model it "
            f"belongs to would be missing from the average."
        ) from exc
    if not isinstance(scores, dict):
        raise ColabFoldParseError(
            f"{path.name} should contain a JSON object of scores, got "
            f"{type(scores).__name__}."
        )
    return scores


def _as_float_array(value, key: str, path: Path) -> np.ndarray:
    """Convert to a float array, refusing anything ragged or non-numeric.

    numpy raises on a ragged nested list, and that raw error says nothing about
    which model it would have dropped, so it is re-raised with the context.
    """
    try:
        return np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ColabFoldParseError(
            f"{path.name} key {key!r} is not a rectangular array of numbers "
            f"({exc}). It cannot be scored, and this prediction must not be "
            f"counted as one of the models."
        ) from exc


def _plddt_array(scores: dict, path: Path) -> np.ndarray:
    if "plddt" not in scores:
        raise ColabFoldParseError(
            f"{path.name} has no 'plddt' key. ColabFold always writes it "
            f"(batch.py:580), so this file was not produced by "
            f"colabfold_batch 1.6.2 or was overwritten."
        )
    plddt = _as_float_array(scores["plddt"], "plddt", path)
    if plddt.ndim != 1:
        raise ColabFoldParseError(
            f"{path.name} 'plddt' has shape {plddt.shape}; it must be a flat "
            f"per-residue array. interface_metrics indexes it by residue, so a "
            f"wrong shape would silently score the wrong residues."
        )
    if plddt.size == 0:
        raise ColabFoldParseError(f"{path.name} 'plddt' is empty.")
    if not np.isfinite(plddt).all():
        raise ColabFoldParseError(
            f"{path.name} 'plddt' contains non-finite values. Their mean would "
            f"be NaN and would poison the across-model average."
        )
    return plddt


def _pae_array(scores: dict, path: Path, n_res: int, require_pae: bool) -> np.ndarray | None:
    if "pae" not in scores:
        if require_pae:
            raise ColabFoldParseError(
                f"{path.name} has no 'pae' key. ColabFold writes PAE only when "
                f"the model returned predicted_aligned_error (batch.py:581), "
                f"which means a non-ptm model type was used. Interface scoring "
                f"is defined on the PAE matrix, so there is nothing to score. "
                f"Re-run with a ptm or multimer model type, or pass "
                f"require_pae=False if PAE is genuinely not wanted."
            )
        return None
    pae = _as_float_array(scores["pae"], "pae", path)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ColabFoldParseError(
            f"{path.name} 'pae' has shape {pae.shape}, which is not a square "
            f"matrix. The interface metric averages the two off-diagonal "
            f"binder/target blocks, which are undefined on a non-square "
            f"matrix."
        )
    if pae.shape[0] != n_res:
        raise ColabFoldParseError(
            f"{path.name} has {n_res} pLDDT residues but a {pae.shape[0]}x"
            f"{pae.shape[1]} PAE matrix. They must describe the same complex; "
            f"mixing them would compute the interface over the wrong residues."
        )
    if not np.isfinite(pae).all():
        raise ColabFoldParseError(
            f"{path.name} 'pae' contains non-finite values, which would make "
            f"every interface mean derived from it NaN."
        )
    return pae


def _scalar(scores: dict, key: str, path: Path) -> float | None:
    if key not in scores:
        return None
    value = scores[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ColabFoldParseError(
            f"{path.name} key {key!r} is {value!r}, not a number."
        )
    value = float(value)
    if not np.isfinite(value):
        raise ColabFoldParseError(
            f"{path.name} key {key!r} is {value!r}, which is not finite."
        )
    return value


def _structure_path(result_dir: Path, jobname: str, tag: str) -> tuple[Path, bool]:
    """Locate this model's PDB, preferring the relaxed one when it exists.

    Exact filenames, never a glob: --save-recycles writes intermediate
    structures with the extension r{recycles}.pdb (batch.py:503) that a glob
    would happily return instead of the final one.
    """
    relaxed = result_dir / f"{jobname}_{RELAXED_KIND}_{tag}.pdb"
    unrelaxed = result_dir / f"{jobname}_{UNRELAXED_KIND}_{tag}.pdb"
    if relaxed.is_file():
        return relaxed, True
    if unrelaxed.is_file():
        return unrelaxed, False
    raise ColabFoldParseError(
        f"Scores exist for tag {tag!r} but neither {relaxed.name} nor "
        f"{unrelaxed.name} is present in {result_dir}. A prediction with no "
        f"structure cannot be checked for chain order or re-scored, so it must "
        f"not be counted as one of the models."
    )


def _scores_files(result_dir: Path, jobname: str) -> list[Path]:
    prefix = f"{jobname}_{SCORES_KIND}_"
    out = []
    for entry in sorted(result_dir.iterdir()):
        name = entry.name
        if entry.is_file() and name.startswith(prefix) and name.endswith(".json"):
            out.append(entry)
    return out


def _check_config(
    config: dict | None,
    expected_models: int,
    expected_seeds: int,
    result_dir: Path,
) -> None:
    if config is None:
        return
    for key, expected in (("num_models", expected_models), ("num_seeds", expected_seeds)):
        found = config.get(key)
        if found is None:
            continue
        if int(found) != int(expected):
            raise ColabFoldParseError(
                f"{result_dir / 'config.json'} records {key}={found} but this "
                f"parse expects {expected}. The run did not do what the "
                f"analysis assumes, so the across-model average would be over "
                f"a different ensemble than the one pre-registered. Fix the "
                f"caller or the fold, do not relax the check."
            )


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def parse_colabfold_output(
    result_dir: str | Path,
    jobname: str,
    *,
    expected_models: int = 5,
    expected_seeds: int = 1,
    require_pae: bool = True,
    require_iptm: bool = True,
) -> ColabFoldJob:
    """Read one job's per-model predictions out of a colabfold_batch directory.

    ``expected_models`` and ``expected_seeds`` are the values passed to
    ``--num-models`` and ``--num-seeds``. Finding any other number of
    predictions raises: the whole point of this project is the mean over a
    known ensemble, and quietly averaging a subset of it would move the
    headline number without anything looking wrong.

    ``require_iptm`` is on by default because ipTM is the complex-level
    confidence the benchmark reports, and ColabFold drops it for single-chain
    predictions (batch.py:536): its absence means a monomer was folded when a
    complex was intended.
    """
    result_dir = Path(result_dir)
    if not result_dir.is_dir():
        raise ColabFoldParseError(
            f"{result_dir} is not a directory, so there is no ColabFold output "
            f"to parse."
        )
    if expected_models < 1 or expected_seeds < 1:
        raise ColabFoldParseError(
            f"expected_models={expected_models} and expected_seeds="
            f"{expected_seeds} must both be at least 1."
        )
    if not jobname:
        raise ColabFoldParseError("jobname must be a non-empty string.")

    config = read_run_config(result_dir)
    _check_config(config, expected_models, expected_seeds, result_dir)

    scores_files = _scores_files(result_dir, jobname)
    if not scores_files:
        present = sorted(p.name for p in result_dir.iterdir())
        raise ColabFoldParseError(
            f"No {jobname}_scores_*.json files in {result_dir}. Either the job "
            f"name is wrong (ColabFold sanitises job names with "
            f"input.safe_filename, replacing every non-alphanumeric character "
            f"other than _ . - with _), or the fold produced no models. "
            f"Directory contains: {present[:20]}"
        )

    predictions: list[ModelPrediction] = []
    prefix_len = len(f"{jobname}_{SCORES_KIND}_")
    for path in scores_files:
        tag = path.name[prefix_len:-len(".json")]
        parsed = _parse_tag(tag, path.name)
        scores = _load_scores(path)

        plddt = _plddt_array(scores, path)
        pae = _pae_array(scores, path, plddt.shape[0], require_pae)
        iptm = _scalar(scores, "iptm", path)
        ptm = _scalar(scores, "ptm", path)
        max_pae = _scalar(scores, "max_pae", path)

        if require_iptm and iptm is None:
            raise ColabFoldParseError(
                f"{path.name} has no 'iptm'. ColabFold removes ipTM for "
                f"single-chain predictions (batch.py:536), so this prediction "
                f"was of a monomer, not a binder/target complex. Scoring an "
                f"interface on it is meaningless."
            )

        structure_path, is_relaxed = _structure_path(result_dir, jobname, tag)

        predictions.append(
            ModelPrediction(
                model_number=parsed["model_number"],
                seed=parsed["seed"],
                rank=parsed["rank"],
                model_type=parsed["model_type"],
                tag=tag,
                plddt=plddt,
                pae=pae,
                iptm=iptm,
                ptm=ptm,
                max_pae=max_pae,
                scores_path=path,
                structure_path=structure_path,
                structure_is_relaxed=is_relaxed,
            )
        )

    _validate_set(predictions, jobname, result_dir, expected_models, expected_seeds, config)

    predictions.sort(key=lambda p: (p.seed, p.model_number))
    model_types = {p.model_type for p in predictions}
    return ColabFoldJob(
        jobname=jobname,
        result_dir=result_dir,
        predictions=tuple(predictions),
        model_type=sorted(model_types)[0],
        config=config,
    )


def _validate_set(
    predictions: list[ModelPrediction],
    jobname: str,
    result_dir: Path,
    expected_models: int,
    expected_seeds: int,
    config: dict | None,
) -> None:
    """The loud part. Every check here guards the across-model mean."""
    expected_total = expected_models * expected_seeds

    keys = [(p.seed, p.model_number) for p in predictions]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise ColabFoldParseError(
            f"Job {jobname!r} in {result_dir} has more than one scores file for "
            f"(seed, model) {duplicates}. A duplicate would be counted twice in "
            f"the mean across models. Stale output from an earlier run is the "
            f"usual cause; clear the directory and refold."
        )

    if len(predictions) != expected_total:
        stop_at_score = None if config is None else config.get("stop_at_score")
        hint = ""
        if stop_at_score is not None and float(stop_at_score) < 100:
            hint = (
                f" config.json has stop_at_score={stop_at_score}, which makes "
                f"ColabFold stop before running every model (batch.py:601), so "
                f"the ensemble is run-dependent and must not be averaged."
            )
        raise ColabFoldParseError(
            f"Job {jobname!r} in {result_dir}: found {len(predictions)} "
            f"prediction(s) for models {sorted(p.model_number for p in predictions)} "
            f"seeds {sorted({p.seed for p in predictions})}, expected "
            f"{expected_total} ({expected_models} models x {expected_seeds} "
            f"seeds). The primary metric is the mean across the full ensemble; "
            f"averaging a subset while believing it was the whole set would "
            f"change the headline number with nothing looking wrong.{hint}"
        )

    seeds = sorted({p.seed for p in predictions})
    if len(seeds) != expected_seeds:
        raise ColabFoldParseError(
            f"Job {jobname!r}: expected {expected_seeds} seed(s) but found "
            f"{len(seeds)} ({seeds}). The per-seed ensembles are not the same "
            f"size, so a mean over all records would weight seeds unequally."
        )
    for seed in seeds:
        found = sorted(p.model_number for p in predictions if p.seed == seed)
        if len(set(found)) != expected_models:
            raise ColabFoldParseError(
                f"Job {jobname!r} seed {seed}: AlphaFold models {found} present, "
                f"expected {expected_models} distinct models. Averaging an "
                f"unbalanced set across seeds would weight some networks more "
                f"than others."
            )

    ranks = sorted(p.rank for p in predictions)
    if ranks != list(range(1, expected_total + 1)):
        raise ColabFoldParseError(
            f"Job {jobname!r}: ranks {ranks} are not the contiguous set 1.."
            f"{expected_total}. ColabFold assigns one rank per prediction "
            f"(batch.py:636), so a gap means output is missing and a repeat "
            f"means two runs' files are mixed in one directory."
        )

    lengths = {p.complex_length for p in predictions}
    if len(lengths) != 1:
        per_model = {p.model_id: p.complex_length for p in predictions}
        raise ColabFoldParseError(
            f"Job {jobname!r}: models disagree on complex length: {per_model}. "
            f"They must all be the same complex; otherwise the binder/target "
            f"split differs between models and the per-model metrics are not "
            f"comparable, let alone averageable."
        )

    model_types = {p.model_type for p in predictions}
    if len(model_types) != 1:
        raise ColabFoldParseError(
            f"Job {jobname!r}: mixed model types {sorted(model_types)} in one "
            f"directory. Averaging across architectures is not the ensemble "
            f"this benchmark pre-registered."
        )
    if config is not None and "model_type" in config:
        if config["model_type"] not in model_types:
            raise ColabFoldParseError(
                f"Job {jobname!r}: filenames say model type "
                f"{sorted(model_types)[0]!r} but config.json says "
                f"{config['model_type']!r}. The directory holds output from a "
                f"different run than its config describes."
            )


# ---------------------------------------------------------------------------
# bridge to metrics.py (no metric maths is reimplemented here)
# ---------------------------------------------------------------------------


def to_metric_records(
    job: ColabFoldJob,
    expected_binder_length: int,
) -> list[dict[str, float]]:
    """Per-model dicts shaped for ``metrics.aggregate_across_models``.

    Thin wiring only: the interface maths lives in ``metrics.interface_metrics``
    and is called, not copied. Each record also carries ``model_number``,
    ``seed`` and ``rank`` so model identity survives into the analysis.
    """
    records: list[dict[str, float]] = []
    for p in job.predictions:
        if p.pae is None:
            raise ColabFoldParseError(
                f"Prediction {p.model_id} of job {job.jobname!r} has no PAE "
                f"matrix, so no interface metric can be computed for it. "
                f"Refusing to return a record that would be averaged as if it "
                f"were complete."
            )
        record: dict[str, float] = dict(
            interface_metrics(p.pae, p.plddt, expected_binder_length)
        )
        record.update(p.confidence())
        record["model_number"] = p.model_number
        record["seed"] = p.seed
        record["rank"] = p.rank
        records.append(record)
    return records
