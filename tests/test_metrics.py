"""Tests for the interface metrics and the chain-order guard.

The chain-order guard is the one the spec names explicitly: if chain order
flips, the metric gets computed over the wrong block of the PAE matrix and
returns numbers that look plausible and mean nothing. So these tests check that
it fails rather than that it succeeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from binder_bench.metrics import aggregate_across_models, interface_metrics
from binder_bench.target import EXPECTED_LENGTH, assert_chain_layout, split_index

BINDER_LEN = 40
TOTAL = BINDER_LEN + EXPECTED_LENGTH


def _pae(binder_len: int = BINDER_LEN, total: int = TOTAL) -> np.ndarray:
    """PAE matrix with distinguishable intra and inter chain blocks.

    Intra-chain blocks are 1.0, the binder-to-target block is 10.0 and the
    target-to-binder block is 20.0. So a correct interface mean is 15.0 and
    anything that leaks intra-chain values in will come out lower.
    """
    m = np.ones((total, total), dtype=float)
    m[:binder_len, binder_len:] = 10.0
    m[binder_len:, :binder_len] = 20.0
    return m


def test_interface_mean_matches_adaptyv_definition():
    out = interface_metrics(_pae(), np.full(TOTAL, 80.0), BINDER_LEN)
    assert out["pae_interface_mean"] == pytest.approx(15.0)
    assert out["pae_interface_min"] == pytest.approx(10.0)
    assert out["binder_length"] == BINDER_LEN
    assert out["complex_length"] == TOTAL


def test_plddt_binder_mean_uses_only_the_binder():
    plddt = np.concatenate([np.full(BINDER_LEN, 90.0), np.full(EXPECTED_LENGTH, 10.0)])
    out = interface_metrics(_pae(), plddt, BINDER_LEN)
    assert out["plddt_binder_mean"] == pytest.approx(90.0)


def test_chain_order_flip_is_rejected_not_silently_accepted():
    """A target-first complex must raise, not return a plausible number."""
    with pytest.raises(ValueError, match="Chain layout mismatch"):
        interface_metrics(_pae(), np.full(TOTAL, 80.0), EXPECTED_LENGTH)


def test_binder_length_disagreement_is_rejected():
    with pytest.raises(ValueError, match="Chain layout mismatch"):
        interface_metrics(_pae(), np.full(TOTAL, 80.0), BINDER_LEN + 1)


def test_complex_shorter_than_target_is_rejected():
    with pytest.raises(ValueError, match="Refusing to guess"):
        split_index(EXPECTED_LENGTH - 10)


def test_complex_exactly_target_length_is_rejected():
    """Zero-length binder means the target was folded alone."""
    with pytest.raises(ValueError, match="Refusing to guess"):
        split_index(EXPECTED_LENGTH)


def test_assert_chain_layout_returns_binder_length_when_consistent():
    assert assert_chain_layout(TOTAL, BINDER_LEN) == BINDER_LEN


def test_non_square_pae_is_rejected():
    with pytest.raises(ValueError, match="must be square"):
        interface_metrics(np.ones((10, 11)), np.full(10, 80.0), 5)


def test_plddt_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same complex"):
        interface_metrics(_pae(), np.full(TOTAL - 1, 80.0), BINDER_LEN)


def test_primary_estimator_is_the_mean_not_the_best():
    """The headline number must be the mean across models, not best-of-five."""
    per_model = [
        {"pae_interface_mean": 10.0, "iptm": 0.90},
        {"pae_interface_mean": 20.0, "iptm": 0.50},
        {"pae_interface_mean": 30.0, "iptm": 0.40},
    ]
    out = aggregate_across_models(per_model)
    assert out["pae_interface_mean__mean"] == pytest.approx(20.0)
    # rank_001 is the highest-ipTM model, which here is the optimistic one.
    assert out["pae_interface_mean__rank001"] == pytest.approx(10.0)
    # And the two must not be confused: the optimistic value is strictly better
    # than the mean, which is the whole reason we do not use it as primary.
    assert out["pae_interface_mean__rank001"] < out["pae_interface_mean__mean"]
    assert out["n_models"] == 3


def test_ensemble_spread_is_recorded():
    per_model = [{"pae_interface_mean": v, "iptm": 0.5} for v in (10.0, 12.0, 14.0)]
    out = aggregate_across_models(per_model)
    assert out["pae_interface_mean__sd"] == pytest.approx(2.0)
    assert out["pae_interface_mean__min"] == pytest.approx(10.0)
    assert out["pae_interface_mean__max"] == pytest.approx(14.0)


def test_single_model_spread_is_zero_not_nan():
    out = aggregate_across_models([{"pae_interface_mean": 11.0, "iptm": 0.7}])
    assert out["pae_interface_mean__sd"] == 0.0


def test_empty_aggregate_is_rejected():
    with pytest.raises(ValueError, match="No per-model metrics"):
        aggregate_across_models([])
