"""Tests for the cost projection.

The point of these is that the projection must not flatter the budget. Each
test states the error it would catch.
"""

from __future__ import annotations

import pytest

from binder_bench.project_cost import (
    MEDIAN_COMPLEX_LENGTH,
    N_EXPRESSED,
    Projection,
    compare_cards,
    scale_to_length,
)


def _proj(**kw) -> Projection:
    base = dict(
        gpu="L4",
        n_designs=100,
        num_models=5,
        seconds_per_model=60.0,
        compile_seconds_per_chunk=0.0,
        chunk_size=5,
    )
    base.update(kw)
    return Projection(**base)


def test_gpu_seconds_counts_every_model_of_every_design():
    p = _proj()
    assert p.gpu_seconds == pytest.approx(100 * 5 * 60.0)


def test_compilation_is_paid_per_chunk_not_per_design():
    """Charging it per design would overstate cost and could kill the project
    for no reason; omitting it understates. It is per chunk."""
    p = _proj(compile_seconds_per_chunk=300.0)
    assert p.n_chunks == 20
    assert p.gpu_seconds == pytest.approx(100 * 5 * 60.0 + 20 * 300.0)


def test_chunk_count_rounds_up():
    """19 designs in chunks of 5 is four chunks, not three and a bit."""
    assert _proj(n_designs=19).n_chunks == 4


def test_fewer_models_scales_the_bill_linearly():
    """The cheapest lever that does not touch the target construct."""
    five = _proj(num_models=5).total_usd
    three = _proj(num_models=3).total_usd
    assert three == pytest.approx(five * 3 / 5)


def test_parallelism_buys_time_not_money():
    """The error that would let a runaway bill look affordable."""
    p = _proj()
    assert p.wall_clock_hours(1) > p.wall_clock_hours(4)
    # Cost is unchanged by how many containers run at once.
    assert p.total_usd == pytest.approx(_proj().total_usd)


def test_wall_clock_rounds_chunks_up_per_container():
    """21 chunks over 4 containers is 6 rounds, not 5.25."""
    p = _proj(n_designs=105, chunk_size=5)  # 21 chunks
    per_chunk_h = (p.gpu_seconds / p.n_chunks) / 3600
    assert p.wall_clock_hours(4) == pytest.approx(6 * per_chunk_h)


def test_expensive_card_is_not_automatically_more_expensive_per_design():
    """Rate per hour is not cost. This is the whole premise of the pilot.

    An A100-40GB is 2.6x the L4's hourly rate here, so if it folds more than
    2.6x faster it is strictly cheaper per design.
    """
    rows = compare_cards(
        {"L4": 600.0, "A100-40GB": 150.0}, n_designs=N_EXPRESSED, num_models=5
    )
    assert rows[0].gpu == "A100-40GB", "a 4x speedup at 2.6x the rate must win"
    assert rows[0].per_design_usd < rows[1].per_design_usd


def test_cards_are_ranked_cheapest_per_design_first():
    rows = compare_cards({"L4": 600.0, "A100-40GB": 600.0})
    assert [r.gpu for r in rows] == ["L4", "A100-40GB"]


def test_unknown_card_raises_rather_than_projecting_from_a_guess():
    with pytest.raises(ValueError, match="No rate known"):
        _proj(gpu="RTX4090").total_usd


def test_length_scaling_is_mild_over_the_control_to_median_span():
    """674 to 699 residues is the correction we actually apply."""
    scaled = scale_to_length(100.0, 674, MEDIAN_COMPLEX_LENGTH)
    assert 1.05 < scaled / 100.0 < 1.15


def test_length_scaling_is_insensitive_to_the_exponent_over_that_span():
    """So the assumption cannot quietly carry the conclusion."""
    low = scale_to_length(100.0, 674, MEDIAN_COMPLEX_LENGTH, exponent=2.0)
    high = scale_to_length(100.0, 674, MEDIAN_COMPLEX_LENGTH, exponent=3.0)
    assert abs(high - low) < 5.0


def test_length_scaling_refuses_a_large_extrapolation():
    """Projecting a domain-III truncation from a full-ectodomain measurement
    is exactly the extrapolation that must be measured, not assumed."""
    with pytest.raises(ValueError, match="Refusing to scale"):
        scale_to_length(100.0, 699, 200)


def test_length_scaling_rejects_nonsense_lengths():
    with pytest.raises(ValueError, match="positive"):
        scale_to_length(100.0, 0, 699)
