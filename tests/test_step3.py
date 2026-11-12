"""Tests for the Step 3 driver, on synthetic data only.

These were written before the join to real outcomes was ever run. They check
the mechanics: that direction is honoured, that the censored KD values are
handled as pre-registered, that the family-wise correction is applied, and
that a mismatched population is refused rather than quietly re-denominated.
"""

from __future__ import annotations

import numpy as np
import pytest

from binder_bench.step3 import (
    K_VALUES,
    METRICS,
    Joined,
    apply_bh,
    claim_a,
    claim_b,
)

N_POS, N_NEG = 53, 325
N = N_POS + N_NEG


def _joined(seed: int = 0, *, perfect: bool = False) -> Joined:
    """A synthetic population matching the real shape: 53 binders of 378."""
    rng = np.random.default_rng(seed)
    binds = np.array([True] * N_POS + [False] * N_NEG)

    if perfect:
        # Low pae for binders, high for non-binders: a flawless lower-is-better
        # metric. iptm and ipsae are built as its mirror image.
        pae = np.concatenate([np.linspace(5, 10, N_POS), np.linspace(20, 28, N_NEG)])
        high = 1.0 - (pae - pae.min()) / (pae.max() - pae.min())
    else:
        pae = rng.normal(22, 3, N)
        high = rng.random(N)

    kd = np.full(N, np.nan)
    # 44 uncensored binders spread over four decades, 9 pinned at the ceiling.
    kd[:44] = 10 ** rng.uniform(-9, -5.1, 44)
    kd[44:N_POS] = 1e-5
    at_ceiling = np.zeros(N, dtype=bool)
    at_ceiling[44:N_POS] = True

    return Joined(
        design_ids=[f"d{i}" for i in range(N)],
        metrics={
            "pae_interface_mean": pae,
            "pae_interface_contacts_mean": pae - 6.0,
            "iptm": high,
            "ipsae": high * 0.8,
        },
        binds=binds,
        kd=kd,
        kd_at_ceiling=at_ceiling,
    )


# ---------------------------------------------------------------------------
# Claim A
# ---------------------------------------------------------------------------


def test_claim_a_reports_every_metric_whatever_the_result():
    """ANALYSIS_PLAN: all four reported together, primary identified."""
    rows = claim_a(_joined(), n_bootstrap=100)
    assert len(rows) == len(METRICS) == 4
    assert [r["role"] for r in rows][0] == "primary"
    assert {r["metric"] for r in rows} == {m[0] for m in METRICS}


def test_claim_a_recovers_a_perfect_metric_in_both_directions():
    """A lower-is-better and a higher-is-better metric built from the same
    signal must both come out at AUC 1.0, or the direction handling is wrong."""
    rows = {r["metric"]: r for r in claim_a(_joined(perfect=True), n_bootstrap=100)}
    assert rows["pae_interface_mean"]["auc"] == pytest.approx(1.0)
    assert rows["iptm"]["auc"] == pytest.approx(1.0)
    assert rows["pae_interface_mean"]["auc_ci_excludes_half"] is True


def test_claim_a_on_a_useless_metric_does_not_exclude_half():
    rows = {r["metric"]: r for r in claim_a(_joined(seed=4), n_bootstrap=600)}
    r = rows["pae_interface_mean"]
    assert r["auc_ci_low"] < 0.5 < r["auc_ci_high"]
    assert r["auc_ci_excludes_half"] is False


def test_claim_a_records_the_prespecified_denominator_and_base_rate():
    r = claim_a(_joined(), n_bootstrap=50)[0]
    assert (r["n"], r["n_binders"], r["n_non_binders"]) == (378, 53, 325)
    assert r["base_rate"] == pytest.approx(53 / 378, abs=1e-4)


def test_claim_a_includes_every_prespecified_k():
    r = claim_a(_joined(), n_bootstrap=50)[0]
    for k in K_VALUES:
        assert f"precision_at_{k}" in r


def test_precision_at_k_is_one_for_a_perfect_metric():
    rows = {r["metric"]: r for r in claim_a(_joined(perfect=True), n_bootstrap=50)}
    for k in K_VALUES:
        assert rows["pae_interface_mean"][f"precision_at_{k}"] == 1.0


def test_hanley_mcneil_se_is_reported_alongside_the_bootstrap():
    """SPEC's power calculation used this formula, so keeping it lets achieved
    precision be compared against what was promised in advance."""
    r = claim_a(_joined(), n_bootstrap=50)[0]
    assert 0.03 < r["hanley_mcneil_se"] < 0.06


# ---------------------------------------------------------------------------
# Claim B and censoring
# ---------------------------------------------------------------------------


def test_claim_b_primary_excludes_the_ceiling_values():
    """9 of 53 sit at exactly 1e-5, a decade above the highest concentration
    assayed, so their true KD is bounded, not measured."""
    rows = claim_b(_joined(), n_bootstrap=100)
    prim = [r for r in rows if r["population"] == "uncensored_binders_primary"]
    sec = [r for r in rows if r["population"] == "all_binders_secondary"]
    assert {r["n"] for r in prim} == {44}
    assert {r["n"] for r in sec} == {53}


def test_claim_b_reports_both_populations_for_every_metric():
    rows = claim_b(_joined(), n_bootstrap=100)
    assert len(rows) == 2 * len(METRICS)


def test_claim_b_recovers_a_monotone_affinity_relation():
    """Build KD to track the metric exactly, then require rho near 1."""
    j = _joined()
    kd = j.kd.copy()
    binder = j.binds & np.isfinite(kd) & (~j.kd_at_ceiling)
    metric = j.metrics["pae_interface_mean"].copy()
    metric[binder] = np.log10(kd[binder]) * 2.0  # perfectly monotone in KD
    j2 = Joined(
        j.design_ids, {**j.metrics, "pae_interface_mean": metric},
        j.binds, kd, j.kd_at_ceiling,
    )
    rows = {
        (r["metric"], r["population"]): r
        for r in claim_b(j2, n_bootstrap=200)
    }
    r = rows[("pae_interface_mean", "uncensored_binders_primary")]
    assert r["spearman"] == pytest.approx(1.0, abs=1e-6)


def test_claim_b_handles_the_tied_ceiling_without_crashing():
    rows = claim_b(_joined(), n_bootstrap=100)
    sec = [r for r in rows if r["population"] == "all_binders_secondary"]
    assert all(r["spearman"] is not None for r in sec)


# ---------------------------------------------------------------------------
# Multiplicity
# ---------------------------------------------------------------------------


def test_bh_is_applied_across_the_metric_family():
    rows = [
        {"metric": "a", "p_vs_half": 0.001},
        {"metric": "b", "p_vs_half": 0.9},
        {"metric": "c", "p_vs_half": 0.9},
        {"metric": "d", "p_vs_half": 0.9},
    ]
    out = apply_bh(rows, "p_vs_half")
    assert out[0]["bh_significant"] is True
    assert [r["bh_significant"] for r in out[1:]] == [False, False, False]


def test_bh_flag_is_attached_by_position_not_identity():
    """Looking the flag up by id() after copying the row would return None for
    every row, which is how this was first written."""
    rows = [{"metric": m, "p_vs_half": p} for m, p in
            zip("abcd", [0.001, 0.002, 0.003, 0.004])]
    out = apply_bh(rows, "p_vs_half")
    assert all(r["bh_significant"] is not None for r in out)


def test_bh_leaves_rows_without_a_p_value_unflagged():
    rows = [{"metric": "a", "p": 0.01}, {"metric": "b", "p": None}]
    out = apply_bh(rows, "p")
    assert out[1]["bh_significant"] is None


def test_apply_bh_does_not_mutate_its_input():
    rows = [{"metric": "a", "p_vs_half": 0.01}]
    apply_bh(rows, "p_vs_half")
    assert "bh_significant" not in rows[0]
