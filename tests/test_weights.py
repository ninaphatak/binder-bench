"""Tests for local weight staging.

This code exists because reading parameters off a network Volume cost over two
GPU-hours across three cards with no output. The tests therefore focus on the
ways a copy can go wrong quietly.
"""

from __future__ import annotations

import pytest

from binder_bench.weights import (
    PARAM_PATTERNS,
    SUCCESS_MARKERS,
    required_files,
    stage_weights_locally,
)

MODEL_TYPE = "alphafold2_multimer_v3"


def _fake_volume(root, num_models=5, size=2048):
    """A stand-in weights Volume with the files ColabFold expects."""
    params = root / "params"
    params.mkdir(parents=True)
    for n in range(1, num_models + 1):
        (params / PARAM_PATTERNS[MODEL_TYPE].format(n=n)).write_bytes(
            bytes([n % 256]) * size
        )
    (params / SUCCESS_MARKERS[MODEL_TYPE]).write_text("done")
    return root


def test_success_marker_is_required_or_colabfold_redownloads():
    """Without it, download_alphafold_params pulls 3.8 GiB per container."""
    names = required_files(MODEL_TYPE, 5)
    assert SUCCESS_MARKERS[MODEL_TYPE] in names


def test_all_five_models_are_required_even_for_a_one_model_run():
    """ColabFold's model_build_order is [3,4,5,1,2] and it loads every one,
    compiling only model 3 and swapping params into that runner. A one-model
    run therefore still reads all five files.

    The earlier version of this listed models 1..num_models and failed on a
    real run with FileNotFoundError for params_model_3_multimer_v3.npz.
    """
    one = required_files(MODEL_TYPE, 1)
    for n in (1, 2, 3, 4, 5):
        assert f"params_model_{n}_multimer_v3.npz" in one
    assert len(one) == 6  # five models plus the marker
    assert required_files(MODEL_TYPE, 5) == one


def test_model_three_is_copied_first():
    """It is the one ColabFold loads first, so a partial copy fails fast."""
    names = required_files(MODEL_TYPE, 1)
    assert names[0] == "params_model_3_multimer_v3.npz"


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match="Unknown model_type"):
        required_files("alphafold3", 5)


@pytest.mark.parametrize("n", [0, 6, -1])
def test_absurd_model_counts_are_rejected(n):
    with pytest.raises(ValueError, match="num_models must be"):
        required_files(MODEL_TYPE, n)


def test_staging_copies_what_is_needed_and_returns_the_data_dir(tmp_path):
    vol = _fake_volume(tmp_path / "vol")
    local = stage_weights_locally(vol, tmp_path / "local", MODEL_TYPE, 2, verbose=False)

    assert local == tmp_path / "local"
    params = local / "params"
    for n in (1, 2, 3, 4, 5):
        assert (params / f"params_model_{n}_multimer_v3.npz").exists()
    assert (params / SUCCESS_MARKERS[MODEL_TYPE]).exists()


def test_contents_are_copied_faithfully(tmp_path):
    vol = _fake_volume(tmp_path / "vol")
    local = stage_weights_locally(vol, tmp_path / "local", MODEL_TYPE, 1, verbose=False)
    src = vol / "params" / "params_model_1_multimer_v3.npz"
    dst = local / "params" / "params_model_1_multimer_v3.npz"
    assert dst.read_bytes() == src.read_bytes()


def test_a_warm_container_does_not_recopy(tmp_path):
    """Chunk two must not pay the copy again."""
    vol = _fake_volume(tmp_path / "vol")
    local_dir = tmp_path / "local"
    stage_weights_locally(vol, local_dir, MODEL_TYPE, 1, verbose=False)

    dst = local_dir / "params" / "params_model_1_multimer_v3.npz"
    marker = dst.stat().st_mtime_ns
    stage_weights_locally(vol, local_dir, MODEL_TYPE, 1, verbose=False)
    assert dst.stat().st_mtime_ns == marker, "file was recopied despite matching size"


def test_a_truncated_local_file_is_replaced_not_trusted(tmp_path):
    """A half-copied parameter file would fail deep inside haiku."""
    vol = _fake_volume(tmp_path / "vol")
    local_dir = tmp_path / "local"
    params = local_dir / "params"
    params.mkdir(parents=True)
    truncated = params / "params_model_1_multimer_v3.npz"
    truncated.write_bytes(b"\x01" * 10)

    stage_weights_locally(vol, local_dir, MODEL_TYPE, 1, verbose=False)
    assert truncated.stat().st_size == 2048


def test_missing_source_raises_rather_than_letting_colabfold_redownload(tmp_path):
    vol = tmp_path / "vol"
    (vol / "params").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="missing from the weights Volume"):
        stage_weights_locally(vol, tmp_path / "local", MODEL_TYPE, 1, verbose=False)


def test_data_dir_layout_matches_what_alphafold_expects(tmp_path):
    """AlphaFold joins "params" onto the data dir internally, so --data must
    be the PARENT of params, not params itself."""
    vol = _fake_volume(tmp_path / "vol")
    local = stage_weights_locally(vol, tmp_path / "local", MODEL_TYPE, 1, verbose=False)
    assert (local / "params").is_dir()
    assert local.name != "params"
