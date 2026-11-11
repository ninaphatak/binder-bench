"""Tests for the pre-launch cost ceiling.

The ceiling exists because the first GPU run cost $1.70 and produced nothing,
and a flat assumed hourly rate would have understated an A100 run threefold.
"""

from __future__ import annotations

import pytest

from binder_bench.costs import GPU_RATES_USD_PER_SEC, worst_case_usd


def test_ceiling_is_per_chunk_not_per_container():
    """max_containers bounds wall time, not the bill."""
    one = worst_case_usd("L4", n_containers=1, timeout_sec=7200)
    twelve = worst_case_usd("L4", n_containers=12, timeout_sec=7200)
    assert twelve == pytest.approx(12 * one)


def test_rate_matches_the_recorded_card_price():
    assert worst_case_usd("L4", 1, 3600) == pytest.approx(
        GPU_RATES_USD_PER_SEC["L4"] * 3600
    )


def test_expensive_cards_project_higher_than_cheap_ones():
    """The bug this replaces assumed one flat rate for every card."""
    assert worst_case_usd("A100-40GB", 1, 3600) > worst_case_usd("L4", 1, 3600)
    assert worst_case_usd("H100", 1, 3600) > worst_case_usd("A100-40GB", 1, 3600)


def test_unknown_card_raises_rather_than_guessing():
    """A guessed cheap price for an expensive card is the costly error here."""
    with pytest.raises(ValueError, match="No rate known"):
        worst_case_usd("B200", 1, 3600)
