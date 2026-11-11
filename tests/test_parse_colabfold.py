"""Tests for the ColabFold output parser.

The failure these tests exist to prevent is quiet partial parsing: finding
three models, averaging them, and reporting the result as a mean over five.
That moves the headline number with nothing looking wrong. So most of what
follows checks that the parser RAISES, and the happy-path tests check that
model identity survives rather than being collapsed into a rank.

Synthetic directories use the real ColabFold 1.6.2 filename convention:

    {jobname}_scores_rank_{n:03d}_{model_type}_model_{m}_seed_{s:03d}.json
    {jobname}_unrelaxed_rank_{n:03d}_{model_type}_model_{m}_seed_{s:03d}.pdb
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from binder_bench.metrics import aggregate_across_models
from binder_bench.parse_colabfold import (
    ColabFoldParseError,
    parse_colabfold_output,
    read_run_config,
    to_metric_records,
)
from binder_bench.target import EXPECTED_LENGTH

JOBNAME = "design_0001"
MODEL_TYPE = "alphafold2_multimer_v3"
BINDER_LEN = 40
TOTAL = BINDER_LEN + EXPECTED_LENGTH

# ipTM per AlphaFold model number. Deliberately spread so that the mean and the
# rank_001 value cannot be confused, and so that model 3 is the single
# optimistic one, as an adversarial design would be.
IPTM_BY_MODEL = {1: 0.40, 2: 0.45, 3: 0.92, 4: 0.50, 5: 0.55}


def _pae(binder_len: int = BINDER_LEN, total: int = TOTAL) -> list[list[float]]:
    m = np.ones((total, total), dtype=float)
    m[:binder_len, binder_len:] = 10.0
    m[binder_len:, :binder_len] = 20.0
    return m.tolist()


def _tag(rank: int, model_number: int, seed: int = 0, model_type: str = MODEL_TYPE) -> str:
    return f"rank_{rank:03d}_{model_type}_model_{model_number}_seed_{seed:03d}"


def _write_prediction(
    result_dir,
    rank: int,
    model_number: int,
    *,
    seed: int = 0,
    jobname: str = JOBNAME,
    model_type: str = MODEL_TYPE,
    plddt=None,
    pae=None,
    iptm=None,
    ptm: float | None = 0.70,
    write_structure: bool = True,
    relaxed: bool = False,
    raw_json: str | None = None,
):
    """Write one model's scores JSON plus its PDB, as ColabFold would."""
    tag = _tag(rank, model_number, seed, model_type)
    scores_path = result_dir / f"{jobname}_scores_{tag}.json"

    if raw_json is not None:
        scores_path.write_text(raw_json)
    else:
        if plddt is None:
            plddt = np.full(TOTAL, 80.0).tolist()
        if pae is None:
            pae = _pae()
        scores: dict = {"plddt": plddt}
        if pae is not None:
            arr = np.asarray(pae, dtype=float)
            scores["pae"] = pae
            scores["max_pae"] = float(arr.max())
        if ptm is not None:
            scores["ptm"] = ptm
        value = IPTM_BY_MODEL[model_number] if iptm is None else iptm
        if value is not None:
            scores["iptm"] = value
        scores_path.write_text(json.dumps(scores))

    if write_structure:
        kind = "relaxed" if relaxed else "unrelaxed"
        (result_dir / f"{jobname}_{kind}_{tag}.pdb").write_text("ATOM\nEND\n")
    return scores_path


def _write_config(result_dir, **overrides):
    config = {
        "num_models": 5,
        "num_seeds": 1,
        "model_order": [1, 2, 3, 4, 5],
        "model_type": MODEL_TYPE,
        "rank_by": "multimer",
        "stop_at_score": 100,
        "version": "1.6.2",
    }
    config.update(overrides)
    (result_dir / "config.json").write_text(json.dumps(config))
    return config


@pytest.fixture
def five_models(tmp_path):
    """A complete, well-formed five-model job directory.

    Rank order is deliberately NOT model order: ColabFold ranks multimers by
    ipTM, so model 3 (the optimistic one) is rank_001.
    """
    d = tmp_path / "out"
    d.mkdir()
    _write_config(d)
    ranked = sorted(IPTM_BY_MODEL, key=lambda m: IPTM_BY_MODEL[m], reverse=True)
    for rank, model_number in enumerate(ranked, start=1):
        _write_prediction(d, rank, model_number)
    return d


# ---------------------------------------------------------------------------
# happy path, and model identity in particular
# ---------------------------------------------------------------------------


def test_all_five_models_are_found(five_models):
    job = parse_colabfold_output(five_models, JOBNAME)
    assert job.n_models == 5
    assert sorted(job.model_numbers) == [1, 2, 3, 4, 5]


def test_records_are_sorted_by_model_not_by_rank(five_models):
    """Iteration order must be model identity, so records are reproducible."""
    job = parse_colabfold_output(five_models, JOBNAME)
    assert [p.model_number for p in job.predictions] == [1, 2, 3, 4, 5]


def test_each_record_carries_its_alphafold_model_number(five_models):
    """The point of the parser: which network produced which number."""
    job = parse_colabfold_output(five_models, JOBNAME)
    for p in job.predictions:
        assert p.iptm == pytest.approx(IPTM_BY_MODEL[p.model_number])


def test_model_number_is_independent_of_rank(five_models):
    job = parse_colabfold_output(five_models, JOBNAME)
    best = job.rank001()
    assert best.rank == 1
    assert best.model_number == 3
    # Rank 1 is not model 1: confusing the two is the bug this guards.
    assert job.by_model_number(1).rank != 1


def test_shapes_and_scales(five_models):
    job = parse_colabfold_output(five_models, JOBNAME)
    p = job.by_model_number(2)
    assert p.plddt.shape == (TOTAL,)
    assert p.pae.shape == (TOTAL, TOTAL)
    assert p.ptm == pytest.approx(0.70)
    assert p.max_pae == pytest.approx(20.0)
    assert p.complex_length == TOTAL
    assert job.complex_length == TOTAL
    assert job.model_type == MODEL_TYPE


def test_seed_and_model_id(five_models):
    job = parse_colabfold_output(five_models, JOBNAME)
    assert {p.seed for p in job.predictions} == {0}
    assert job.by_model_number(4).model_id == "model_4_seed_000"


def test_structure_path_points_at_the_unrelaxed_pdb(five_models):
    job = parse_colabfold_output(five_models, JOBNAME)
    p = job.by_model_number(5)
    assert p.structure_path.is_file()
    assert p.structure_path.name.endswith(f"{_tag(p.rank, 5)}.pdb")
    assert "_unrelaxed_" in p.structure_path.name
    assert p.structure_is_relaxed is False


def test_relaxed_structure_is_preferred_and_flagged(tmp_path):
    """--num-relax relaxes only the top ranks, so both kinds can coexist."""
    d = tmp_path / "out"
    d.mkdir()
    _write_config(d)
    ranked = sorted(IPTM_BY_MODEL, key=lambda m: IPTM_BY_MODEL[m], reverse=True)
    for rank, model_number in enumerate(ranked, start=1):
        _write_prediction(d, rank, model_number)
        if rank == 1:
            tag = _tag(rank, model_number)
            (d / f"{JOBNAME}_relaxed_{tag}.pdb").write_text("ATOM\nEND\n")
    job = parse_colabfold_output(d, JOBNAME)
    best = job.rank001()
    assert best.structure_is_relaxed is True
    assert "_relaxed_" in best.structure_path.name
    assert all(not p.structure_is_relaxed for p in job.predictions if p.rank != 1)


def test_save_recycles_files_are_not_mistaken_for_the_structure(five_models):
    """--save-recycles writes ..._seed_000.r3.pdb; a glob would grab those."""
    p = parse_colabfold_output(five_models, JOBNAME).by_model_number(1)
    tag = _tag(p.rank, 1)
    for recycle in range(4):
        (five_models / f"{JOBNAME}_unrelaxed_{tag}.r{recycle}.pdb").write_text("ATOM\n")
    again = parse_colabfold_output(five_models, JOBNAME).by_model_number(1)
    assert again.structure_path.name == f"{JOBNAME}_unrelaxed_{tag}.pdb"


def test_other_colabfold_outputs_are_ignored(five_models):
    """The real directory is full of files that are not per-model scores."""
    (five_models / f"{JOBNAME}_predicted_aligned_error_v1.json").write_text(
        json.dumps({"predicted_aligned_error": [[0.0]], "max_predicted_aligned_error": 0.0})
    )
    (five_models / f"{JOBNAME}.json").write_text("{}")
    (five_models / f"{JOBNAME}.a3m").write_text(">1\nAAA\n")
    (five_models / f"{JOBNAME}_pae.png").write_bytes(b"\x89PNG")
    (five_models / f"{JOBNAME}_plddt.png").write_bytes(b"\x89PNG")
    (five_models / f"{JOBNAME}_coverage.png").write_bytes(b"\x89PNG")
    (five_models / f"{JOBNAME}_template_domain_names.json").write_text("{}")
    (five_models / f"{JOBNAME}.done.txt").write_text("")
    (five_models / "cite.bibtex").write_text("")
    job = parse_colabfold_output(five_models, JOBNAME)
    assert job.n_models == 5


def test_a_second_job_in_the_same_directory_is_not_mixed_in(five_models):
    """colabfold_batch writes every job of a batch into one directory."""
    other = "design_0002"
    _write_config(five_models)
    ranked = sorted(IPTM_BY_MODEL, key=lambda m: IPTM_BY_MODEL[m], reverse=True)
    for rank, model_number in enumerate(ranked, start=1):
        _write_prediction(five_models, rank, model_number, jobname=other)
    job = parse_colabfold_output(five_models, JOBNAME)
    assert job.n_models == 5
    assert all(p.scores_path.name.startswith(f"{JOBNAME}_scores_") for p in job.predictions)
    assert parse_colabfold_output(five_models, other).n_models == 5


def test_multiple_seeds_are_parsed_as_separate_records(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _write_config(d, num_seeds=2)
    rank = 1
    for seed in (0, 1):
        for model_number in (1, 2, 3, 4, 5):
            _write_prediction(d, rank, model_number, seed=seed)
            rank += 1
    job = parse_colabfold_output(d, JOBNAME, expected_models=5, expected_seeds=2)
    assert job.n_models == 10
    assert [p.seed for p in job.predictions] == [0] * 5 + [1] * 5
    assert [p.model_number for p in job.predictions] == [1, 2, 3, 4, 5] * 2


def test_config_is_exposed(five_models):
    job = parse_colabfold_output(five_models, JOBNAME)
    assert job.config["num_models"] == 5
    assert read_run_config(five_models)["version"] == "1.6.2"


def test_missing_config_is_not_fatal(tmp_path):
    """config.json is a cross-check, not the source of truth."""
    d = tmp_path / "out"
    d.mkdir()
    ranked = sorted(IPTM_BY_MODEL, key=lambda m: IPTM_BY_MODEL[m], reverse=True)
    for rank, model_number in enumerate(ranked, start=1):
        _write_prediction(d, rank, model_number)
    job = parse_colabfold_output(d, JOBNAME)
    assert job.config is None
    assert job.n_models == 5
    assert read_run_config(d) is None


# ---------------------------------------------------------------------------
# the loud failures
# ---------------------------------------------------------------------------


def test_missing_model_is_rejected(five_models):
    """Four models found where five were requested must not be averaged."""
    victim = parse_colabfold_output(five_models, JOBNAME).by_model_number(2)
    victim.scores_path.unlink()
    with pytest.raises(ColabFoldParseError, match="averaging a subset"):
        parse_colabfold_output(five_models, JOBNAME)


def test_missing_model_error_names_what_was_found(five_models):
    parse_colabfold_output(five_models, JOBNAME).by_model_number(4).scores_path.unlink()
    with pytest.raises(ColabFoldParseError) as excinfo:
        parse_colabfold_output(five_models, JOBNAME)
    message = str(excinfo.value)
    assert "found 4 prediction" in message
    assert "[1, 2, 3, 5]" in message


def test_empty_directory_is_rejected(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(ColabFoldParseError, match="No design_0001_scores_"):
        parse_colabfold_output(d, JOBNAME)


def test_nonexistent_directory_is_rejected(tmp_path):
    with pytest.raises(ColabFoldParseError, match="not a directory"):
        parse_colabfold_output(tmp_path / "nope", JOBNAME)


def test_wrong_jobname_is_rejected_not_silently_empty(five_models):
    with pytest.raises(ColabFoldParseError, match="safe_filename"):
        parse_colabfold_output(five_models, "design_1")


def test_non_square_pae_is_rejected(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _write_config(d)
    ranked = sorted(IPTM_BY_MODEL, key=lambda m: IPTM_BY_MODEL[m], reverse=True)
    for rank, model_number in enumerate(ranked, start=1):
        pae = None
        if model_number == 3:
            pae = np.ones((TOTAL, TOTAL - 1)).tolist()
        _write_prediction(d, rank, model_number, pae=pae)
    with pytest.raises(ColabFoldParseError, match="not a square"):
        parse_colabfold_output(d, JOBNAME)


def test_ragged_pae_is_rejected(tmp_path):
    """A row-length mismatch must not become an object array."""
    d = tmp_path / "out"
    d.mkdir()
    bad = json.dumps({"plddt": [80.0, 80.0, 80.0], "pae": [[1.0, 2.0], [1.0], [1.0, 2.0]]})
    _write_prediction(d, 1, 1, raw_json=bad)
    with pytest.raises(ColabFoldParseError, match="not a rectangular array"):
        parse_colabfold_output(d, JOBNAME, expected_models=1)


def test_plddt_length_mismatch_is_rejected(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _write_prediction(d, 1, 1, plddt=np.full(TOTAL - 7, 80.0).tolist())
    with pytest.raises(ColabFoldParseError, match="must describe the same complex"):
        parse_colabfold_output(d, JOBNAME, expected_models=1)


def test_models_disagreeing_on_complex_length_are_rejected(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _write_config(d, num_models=2)
    _write_prediction(d, 1, 1)
    _write_prediction(
        d, 2, 2, plddt=np.full(TOTAL - 1, 80.0).tolist(), pae=_pae(BINDER_LEN, TOTAL - 1)
    )
    with pytest.raises(ColabFoldParseError, match="disagree on complex length"):
        parse_colabfold_output(d, JOBNAME, expected_models=2)


def test_missing_pae_is_rejected_by_default(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _write_prediction(d, 1, 1, pae=None, raw_json=json.dumps({"plddt": [80.0] * TOTAL}))
    with pytest.raises(ColabFoldParseError, match="no 'pae' key"):
        parse_colabfold_output(d, JOBNAME, expected_models=1)


def test_missing_pae_can_be_allowed_explicitly(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _write_prediction(
        d, 1, 1, raw_json=json.dumps({"plddt": [80.0] * TOTAL, "iptm": 0.5})
    )
    job = parse_colabfold_output(d, JOBNAME, expected_models=1, require_pae=False)
    assert job.predictions[0].pae is None


def test_missing_plddt_is_rejected(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _write_prediction(d, 1, 1, raw_json=json.dumps({"pae": _pae(), "iptm": 0.5}))
    with pytest.raises(ColabFoldParseError, match="no 'plddt' key"):
        parse_colabfold_output(d, JOBNAME, expected_models=1)


def test_missing_iptm_is_rejected_because_it_means_a_monomer(tmp_path):
    """ColabFold drops ipTM for single-chain predictions."""
    d = tmp_path / "out"
    d.mkdir()
    _write_prediction(d, 1, 1, raw_json=json.dumps({
        "plddt": [80.0] * TOTAL, "pae": _pae(), "max_pae": 20.0, "ptm": 0.7,
    }))
    with pytest.raises(ColabFoldParseError, match="monomer"):
        parse_colabfold_output(d, JOBNAME, expected_models=1)


def test_non_finite_plddt_is_rejected(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    plddt = [80.0] * TOTAL
    raw = json.dumps({"plddt": plddt, "pae": _pae(), "iptm": 0.5}).replace(
        '"plddt": [80.0,', '"plddt": [NaN,'
    )
    _write_prediction(d, 1, 1, raw_json=raw)
    with pytest.raises(ColabFoldParseError, match="non-finite"):
        parse_colabfold_output(d, JOBNAME, expected_models=1)


def test_non_finite_pae_is_rejected(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    pae = np.ones((3, 3)).tolist()
    pae[0][0] = float("inf")
    raw = json.dumps({"plddt": [80.0] * 3, "pae": pae, "iptm": 0.5})
    _write_prediction(d, 1, 1, raw_json=raw)
    with pytest.raises(ColabFoldParseError, match="non-finite"):
        parse_colabfold_output(d, JOBNAME, expected_models=1)


def test_missing_structure_file_is_rejected(five_models):
    job = parse_colabfold_output(five_models, JOBNAME)
    job.by_model_number(5).structure_path.unlink()
    with pytest.raises(ColabFoldParseError, match="neither"):
        parse_colabfold_output(five_models, JOBNAME)


def test_truncated_scores_json_is_rejected(five_models):
    victim = parse_colabfold_output(five_models, JOBNAME).by_model_number(1)
    victim.scores_path.write_text('{"plddt": [80.0, 80.0')
    with pytest.raises(ColabFoldParseError, match="not valid JSON"):
        parse_colabfold_output(five_models, JOBNAME)


def test_corrupt_config_json_is_rejected(five_models):
    (five_models / "config.json").write_text("{not json")
    with pytest.raises(ColabFoldParseError, match="not valid JSON"):
        parse_colabfold_output(five_models, JOBNAME)


def test_config_num_models_disagreeing_with_expectation_is_rejected(five_models):
    _write_config(five_models, num_models=3)
    with pytest.raises(ColabFoldParseError, match="num_models=3"):
        parse_colabfold_output(five_models, JOBNAME)


def test_config_model_type_disagreeing_with_filenames_is_rejected(five_models):
    _write_config(five_models, model_type="alphafold2_multimer_v2")
    with pytest.raises(ColabFoldParseError, match="different run than its config"):
        parse_colabfold_output(five_models, JOBNAME)


def test_early_stop_configuration_is_called_out_in_the_error(five_models):
    """stop_at_score makes the ensemble size run-dependent."""
    _write_config(five_models, stop_at_score=80)
    parse_colabfold_output(five_models, JOBNAME).by_model_number(1).scores_path.unlink()
    with pytest.raises(ColabFoldParseError, match="stop_at_score=80"):
        parse_colabfold_output(five_models, JOBNAME)


def test_duplicate_model_from_a_stale_run_is_rejected(five_models):
    """Two scores files for the same model would be counted twice."""
    _write_prediction(five_models, 7, 3)
    with pytest.raises(ColabFoldParseError, match="more than one scores file"):
        parse_colabfold_output(five_models, JOBNAME)


def test_mixed_model_types_in_one_directory_are_rejected(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _write_prediction(d, 1, 1)
    _write_prediction(d, 2, 2, model_type="alphafold2_ptm")
    with pytest.raises(ColabFoldParseError, match="mixed model types"):
        parse_colabfold_output(d, JOBNAME, expected_models=2)


def test_unranked_scores_file_is_rejected(tmp_path):
    """Files are renamed with their rank only after every model finishes."""
    d = tmp_path / "out"
    d.mkdir()
    tag = f"{MODEL_TYPE}_model_1_seed_000"
    (d / f"{JOBNAME}_scores_{tag}.json").write_text(json.dumps({"plddt": [80.0]}))
    (d / f"{JOBNAME}_unrelaxed_{tag}.pdb").write_text("ATOM\n")
    with pytest.raises(ColabFoldParseError, match="died part way through"):
        parse_colabfold_output(d, JOBNAME, expected_models=1)


def test_tag_without_model_number_is_rejected(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    tag = f"rank_001_{MODEL_TYPE}_seed_000"
    (d / f"{JOBNAME}_scores_{tag}.json").write_text(json.dumps({"plddt": [80.0]}))
    with pytest.raises(ColabFoldParseError, match="model identity is"):
        parse_colabfold_output(d, JOBNAME, expected_models=1)


def test_impossible_model_number_is_rejected(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    tag = f"rank_001_{MODEL_TYPE}_model_9_seed_000"
    (d / f"{JOBNAME}_scores_{tag}.json").write_text(json.dumps({"plddt": [80.0]}))
    with pytest.raises(ColabFoldParseError, match="only \\[1, 2, 3, 4, 5\\] exist"):
        parse_colabfold_output(d, JOBNAME, expected_models=1)


def test_non_contiguous_ranks_are_rejected(tmp_path):
    """A rank gap means output is missing even if the count happens to match."""
    d = tmp_path / "out"
    d.mkdir()
    _write_prediction(d, 1, 1)
    _write_prediction(d, 3, 2)
    with pytest.raises(ColabFoldParseError, match="contiguous set"):
        parse_colabfold_output(d, JOBNAME, expected_models=2)


def test_unbalanced_seeds_are_rejected(tmp_path):
    """Right total, wrong shape: 3 models of one seed and 1 of another."""
    d = tmp_path / "out"
    d.mkdir()
    _write_prediction(d, 1, 1, seed=0)
    _write_prediction(d, 2, 2, seed=0)
    _write_prediction(d, 3, 3, seed=0)
    _write_prediction(d, 4, 1, seed=1)
    with pytest.raises(ColabFoldParseError, match="expected 2 distinct models"):
        parse_colabfold_output(d, JOBNAME, expected_models=2, expected_seeds=2)


def test_zero_expected_models_is_rejected(five_models):
    with pytest.raises(ColabFoldParseError, match="at least 1"):
        parse_colabfold_output(five_models, JOBNAME, expected_models=0)


def test_empty_jobname_is_rejected(five_models):
    with pytest.raises(ColabFoldParseError, match="non-empty"):
        parse_colabfold_output(five_models, "")


def test_asking_for_a_model_that_was_not_folded_is_rejected(five_models):
    job = parse_colabfold_output(five_models, JOBNAME)
    with pytest.raises(ColabFoldParseError, match="No prediction for AlphaFold model"):
        job.by_model_number(7)


# ---------------------------------------------------------------------------
# the handoff to metrics.py
# ---------------------------------------------------------------------------


def test_parsed_arrays_feed_interface_metrics(five_models):
    job = parse_colabfold_output(five_models, JOBNAME)
    records = to_metric_records(job, BINDER_LEN)
    assert len(records) == 5
    for record in records:
        assert record["pae_interface_mean"] == pytest.approx(15.0)
        assert record["binder_length"] == BINDER_LEN
        assert record["complex_length"] == TOTAL
    assert [r["model_number"] for r in records] == [1, 2, 3, 4, 5]


def test_aggregate_over_parsed_records_is_the_mean_not_the_best(five_models):
    """End to end: the parser must not smuggle in a best-of-five estimator."""
    job = parse_colabfold_output(five_models, JOBNAME)
    out = aggregate_across_models(to_metric_records(job, BINDER_LEN))
    assert out["n_models"] == 5
    expected_mean = float(np.mean(list(IPTM_BY_MODEL.values())))
    assert out["iptm__mean"] == pytest.approx(expected_mean)
    # rank_001 is model 3, the one optimistic network out of five.
    assert out["iptm__rank001"] == pytest.approx(IPTM_BY_MODEL[3])
    assert out["iptm__rank001"] > out["iptm__mean"]
    assert out["iptm__sd"] > 0.0


def test_chain_order_flip_still_raises_through_the_parser(five_models):
    """The parser must not defeat the chain-layout guard in metrics.py."""
    job = parse_colabfold_output(five_models, JOBNAME)
    with pytest.raises(ValueError, match="Chain layout mismatch"):
        to_metric_records(job, EXPECTED_LENGTH)


def test_metric_records_refuse_a_prediction_with_no_pae(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _write_prediction(
        d, 1, 1, raw_json=json.dumps({"plddt": [80.0] * TOTAL, "iptm": 0.5})
    )
    job = parse_colabfold_output(d, JOBNAME, expected_models=1, require_pae=False)
    with pytest.raises(ColabFoldParseError, match="no PAE matrix"):
        to_metric_records(job, BINDER_LEN)
