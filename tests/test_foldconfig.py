"""Tests for the frozen ColabFold configuration.

These are mostly regression locks. The configuration is the thing that makes
Step 2 a uniform re-prediction rather than a collection of differently
configured runs, so a silent change to any value here would invalidate the
comparison across designs. Each test states what breaks if the value drifts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from binder_bench.foldconfig import (
    FoldConfig,
    assert_no_ambiguous_flags,
    build_command,
    expected_prediction_count,
)


def test_five_models_and_one_seed():
    """The primary estimator is the mean over five models, so five is load bearing."""
    cfg = FoldConfig()
    assert cfg.num_models == 5
    assert cfg.num_seeds == 1
    assert expected_prediction_count(cfg) == 5


def test_recycles_match_the_competition():
    """Three recycles, so recycling is not a confound against their numbers."""
    assert FoldConfig().num_recycle == 3


def test_multimer_weights_are_explicit():
    """Monomer weights on a two-chain complex would be a different experiment."""
    assert FoldConfig().model_type == "alphafold2_multimer_v3"


def test_templates_are_off():
    """Templates would leak solved EGFR complexes into the predictions."""
    assert FoldConfig().use_templates is False
    cmd = build_command("in.a3m", "out", "/weights")
    assert "--templates" not in cmd


def test_relaxation_is_off():
    assert FoldConfig().num_relax == 0
    cmd = build_command("in.a3m", "out", "/weights")
    assert cmd[cmd.index("--num-relax") + 1] == "0"
    assert "--amber" not in cmd


def test_stop_at_score_stays_at_the_permissive_default():
    """A lower value would silently run fewer than five models."""
    assert FoldConfig().stop_at_score == 100
    cmd = build_command("in.a3m", "out", "/weights")
    assert cmd[cmd.index("--stop-at-score") + 1] == "100"


def test_seed_is_fixed_and_passed():
    cmd = build_command("in.a3m", "out", "/weights")
    assert cmd[cmd.index("--random-seed") + 1] == "0"


def test_weights_directory_is_always_passed():
    """Without --data, every container re-downloads 3.8 GiB of parameters."""
    cmd = build_command("in.a3m", "out", "/weights")
    assert "--data" in cmd
    assert cmd[cmd.index("--data") + 1] == "/weights"


def test_ambiguous_overwrite_flag_is_never_emitted():
    """Its name and its help text contradict each other in 1.6.2."""
    cmd = build_command("in.a3m", "out", "/weights")
    assert "--overwrite-existing-results" not in cmd


def test_command_is_a_list_not_a_string():
    """A list cannot have an argument reinterpreted by a shell."""
    cmd = build_command("in.a3m", "out", "/weights")
    assert isinstance(cmd, list)
    assert all(isinstance(x, str) for x in cmd)


def test_positional_arguments_come_first_and_in_order():
    cmd = build_command("/tmp/chunk", "/tmp/out", "/weights")
    assert cmd[0] == "colabfold_batch"
    assert cmd[1] == "/tmp/chunk"
    assert cmd[2] == "/tmp/out"


def test_paths_are_accepted_as_path_objects():
    cmd = build_command(Path("/tmp/chunk"), Path("/tmp/out"), Path("/weights"))
    assert cmd[1] == "/tmp/chunk"
    assert cmd[2] == "/tmp/out"
    assert cmd[cmd.index("--data") + 1] == "/weights"


@pytest.mark.parametrize(
    "flag",
    [
        "--overwrite-existing-results",
        "--templates",
        "--amber",
        "--use-dropout",
        "--initial-guess",
    ],
)
def test_forbidden_flags_are_rejected_loudly(flag: str):
    cmd = build_command("in.a3m", "out", "/weights") + [flag]
    with pytest.raises(ValueError, match="Refusing to run"):
        assert_no_ambiguous_flags(cmd)


def test_clean_command_passes_the_guard():
    assert_no_ambiguous_flags(build_command("in.a3m", "out", "/weights")) is None


def test_config_record_is_complete_for_provenance():
    """Anything that could change a number must land in the run record."""
    record = FoldConfig().as_record()
    for key in (
        "model_type",
        "num_models",
        "num_recycle",
        "num_seeds",
        "random_seed",
        "rank_by",
        "pair_mode",
        "num_relax",
        "use_templates",
        "stop_at_score",
        "sort_queries_by",
    ):
        assert key in record, f"{key} missing from the provenance record"


def test_config_is_frozen():
    """A config mutated mid-run would make the record a lie."""
    cfg = FoldConfig()
    with pytest.raises(Exception):
        cfg.num_models = 1  # type: ignore[misc]


def test_pair_mode_is_unpaired_for_a_single_sequence_binder():
    """Flagged as a pilot check rather than an assumption."""
    assert FoldConfig().pair_mode == "unpaired"
    cmd = build_command("in.a3m", "out", "/weights")
    assert cmd[cmd.index("--pair-mode") + 1] == "unpaired"


def test_alternate_config_flows_through_to_the_command():
    """So the pilot can fold one design both ways without editing defaults."""
    cfg = FoldConfig(pair_mode="unpaired_paired", num_models=1)
    cmd = build_command("in.a3m", "out", "/weights", config=cfg)
    assert cmd[cmd.index("--pair-mode") + 1] == "unpaired_paired"
    assert cmd[cmd.index("--num-models") + 1] == "1"
    assert expected_prediction_count(cfg) == 1


def test_unified_memory_is_disabled_by_flag():
    """ColabFold asks for 4x the card's memory unless told not to. Three cards
    each stalled over 40 minutes in jax batched_device_put because of it."""
    assert FoldConfig().disable_unified_memory is True
    cmd = build_command("in.a3m", "out", "/weights")
    assert "--disable-unified-memory" in cmd


def test_unified_memory_env_overrides_colabfolds_import_time_defaults():
    """ColabFold only sets these "if k not in os.environ", so pre-setting them
    wins. The values must therefore be present and must not be its defaults."""
    from binder_bench.foldconfig import UNIFIED_MEMORY_OFF_ENV

    assert UNIFIED_MEMORY_OFF_ENV["TF_FORCE_UNIFIED_MEMORY"] == "0"
    frac = float(UNIFIED_MEMORY_OFF_ENV["XLA_PYTHON_CLIENT_MEM_FRACTION"])
    assert 0 < frac <= 1.0, "a fraction above 1.0 oversubscribes the card again"


def test_unified_memory_setting_is_in_the_provenance_record():
    assert "disable_unified_memory" in FoldConfig().as_record()


def test_recycle_early_stop_tolerance_is_explicit_not_inherited():
    """ColabFold's CLI default of None does not mean off: it hands over to
    AF2-multimer's own config default of 0.5, so designs silently get
    different numbers of recycles. The value must be in the record."""
    cfg = FoldConfig()
    assert cfg.recycle_early_stop_tolerance == 0.5
    cmd = build_command("in.a3m", "out", "/weights")
    assert cmd[cmd.index("--recycle-early-stop-tolerance") + 1] == "0.5"
    assert "recycle_early_stop_tolerance" in cfg.as_record()


def test_early_stop_tolerance_can_be_forced_off_for_a_uniformity_check():
    """Zero tolerance runs every design to the full recycle count, which is
    the sensitivity check for whether variable recycling moves the metric."""
    cfg = FoldConfig(recycle_early_stop_tolerance=0.0)
    cmd = build_command("in.a3m", "out", "/weights", config=cfg)
    assert cmd[cmd.index("--recycle-early-stop-tolerance") + 1] == "0.0"
