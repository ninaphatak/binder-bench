"""Tests for the Step 3 statistics, all on synthetic data with known answers.

Deliberately never touched the real labels. These were written while the ipSAE
pass was still running, before any outcome had been examined, so that Step 3
can be run once as pre-registered rather than improvised afterwards.
"""

from __future__ import annotations

import numpy as np
import pytest

from binder_bench.stats import (
    Estimate,
    auc,
    benjamini_hochberg,
    bootstrap_auc,
    bootstrap_correlation,
    calibration_by_decile,
    hanley_mcneil_se,
    cumulative_precision,
    pearson,
    precision_at_k,
    roc_curve,
    spearman,
)

# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------


def test_perfect_separation_is_auc_one():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    labels = np.array([0, 0, 1, 1])
    assert auc(scores, labels, lower_is_better=False) == pytest.approx(1.0)


def test_direction_is_taken_from_the_argument_not_the_data():
    """The sign of the finding must never be inferred. A metric where low is
    good gives AUC 1.0 one way and 0.0 the other, from identical inputs."""
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    labels = np.array([1, 1, 0, 0])
    assert auc(scores, labels, lower_is_better=True) == pytest.approx(1.0)
    assert auc(scores, labels, lower_is_better=False) == pytest.approx(0.0)


def test_random_scores_give_auc_near_half():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=4000)
    labels = rng.integers(0, 2, size=4000)
    assert abs(auc(scores, labels, lower_is_better=False) - 0.5) < 0.03


def test_all_ties_give_exactly_half():
    """A metric with no information must score 0.5, not 0 or 1 by tie order."""
    scores = np.ones(100)
    labels = np.array([1] * 20 + [0] * 80)
    assert auc(scores, labels, lower_is_better=False) == pytest.approx(0.5)


def test_auc_matches_the_mann_whitney_definition_by_brute_force():
    rng = np.random.default_rng(3)
    scores = rng.normal(size=60)
    labels = rng.integers(0, 2, size=60).astype(bool)
    pos, neg = scores[labels], scores[~labels]
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg
    )
    assert auc(scores, labels, lower_is_better=False) == pytest.approx(
        wins / (len(pos) * len(neg))
    )


def test_single_class_is_rejected():
    with pytest.raises(ValueError, match="needs both classes"):
        auc(np.array([1.0, 2.0]), np.array([1, 1]), lower_is_better=False)


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


def test_hanley_mcneil_matches_the_spec_power_calculation():
    """SPEC promised SE 0.040 to 0.044 at 53 vs 325 over AUC 0.60 to 0.75.
    Reproducing it here is how the achieved precision gets compared against
    what was claimed before the data existed."""
    for a in (0.60, 0.675, 0.75):
        se = hanley_mcneil_se(a, 53, 325)
        assert 0.038 <= se <= 0.046, (a, se)


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    scores = np.concatenate([rng.normal(1, 1, 53), rng.normal(0, 1, 325)])
    labels = np.array([1] * 53 + [0] * 325)
    est = bootstrap_auc(scores, labels, lower_is_better=False, n_bootstrap=400)
    assert est.ci_low <= est.value <= est.ci_high
    assert est.n == 378


def test_bootstrap_is_reproducible_from_its_seed():
    """An interval re-rolled until it looks tidy is not an interval."""
    rng = np.random.default_rng(2)
    scores = np.concatenate([rng.normal(1, 1, 53), rng.normal(0, 1, 325)])
    labels = np.array([1] * 53 + [0] * 325)
    kw = dict(lower_is_better=False, n_bootstrap=200)
    a = bootstrap_auc(scores, labels, **kw)
    b = bootstrap_auc(scores, labels, **kw)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


def test_bootstrap_ci_width_is_comparable_to_the_analytic_se():
    """A percentile CI roughly 2x1.96xSE wide is evidence neither is wrong."""
    rng = np.random.default_rng(5)
    scores = np.concatenate([rng.normal(0.8, 1, 53), rng.normal(0, 1, 325)])
    labels = np.array([1] * 53 + [0] * 325)
    est = bootstrap_auc(scores, labels, lower_is_better=False, n_bootstrap=1500)
    se = hanley_mcneil_se(est.value, 53, 325)
    width = est.ci_high - est.ci_low
    assert 0.5 * (2 * 1.96 * se) < width < 2.0 * (2 * 1.96 * se)


def test_a_useless_metric_gives_a_ci_containing_half():
    rng = np.random.default_rng(7)
    scores = rng.normal(size=378)
    labels = np.array([1] * 53 + [0] * 325)
    rng.shuffle(labels)
    est = bootstrap_auc(scores, labels, lower_is_better=False, n_bootstrap=800)
    assert est.ci_low < 0.5 < est.ci_high


# ---------------------------------------------------------------------------
# Precision at k and calibration
# ---------------------------------------------------------------------------


def test_precision_at_k_on_a_perfect_ranker():
    scores = np.arange(100, dtype=float)
    labels = (scores >= 90).astype(int)
    assert precision_at_k(scores, labels, 10, lower_is_better=False) == 1.0


def test_precision_at_k_respects_direction():
    scores = np.arange(100, dtype=float)
    labels = (scores < 10).astype(int)
    assert precision_at_k(scores, labels, 10, lower_is_better=True) == 1.0


def test_precision_at_k_of_a_useless_metric_sits_near_the_base_rate():
    rng = np.random.default_rng(11)
    scores = rng.normal(size=3780)
    labels = (rng.random(3780) < 0.14).astype(int)
    p = precision_at_k(scores, labels, 500, lower_is_better=False)
    assert 0.09 < p < 0.19


def test_precision_at_k_rejects_impossible_k():
    with pytest.raises(ValueError, match="k must be"):
        precision_at_k(np.arange(5.0), np.zeros(5), 6, lower_is_better=False)


def test_calibration_is_monotone_for_a_real_ranker():
    """ANALYSIS_PLAN's ranker-versus-threshold test."""
    rng = np.random.default_rng(13)
    scores = rng.normal(size=4000)
    labels = (rng.random(4000) < 1 / (1 + np.exp(-2 * scores))).astype(int)
    curve = calibration_by_decile(scores, labels, lower_is_better=False)
    rates = [c["bind_rate"] for c in curve]
    assert rates[0] > rates[-1]
    # Broadly decreasing: most adjacent pairs should not increase.
    drops = sum(1 for x, y in zip(rates, rates[1:]) if y <= x)
    assert drops >= 7


def test_calibration_is_flat_for_a_useless_metric():
    rng = np.random.default_rng(17)
    scores = rng.normal(size=4000)
    labels = (rng.random(4000) < 0.14).astype(int)
    curve = calibration_by_decile(scores, labels, lower_is_better=False)
    rates = np.array([c["bind_rate"] for c in curve])
    assert rates.std() < 0.05


def test_calibration_covers_every_design_exactly_once():
    scores = np.arange(378, dtype=float)
    labels = np.zeros(378)
    curve = calibration_by_decile(scores, labels, lower_is_better=False)
    assert sum(c["n"] for c in curve) == 378
    assert len(curve) == 10


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def test_spearman_is_one_for_any_monotone_relation():
    x = np.arange(1, 21, dtype=float)
    assert spearman(x, np.exp(x / 5)) == pytest.approx(1.0)


def test_spearman_handles_ties_without_crashing():
    """9 of the 53 binders sit at exactly the KD ceiling, so ties are real."""
    x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1.0, 2, 3, 4, 5, 1e-5, 1e-5, 1e-5, 1e-5])
    r = spearman(x, y)
    assert -1.0 <= r <= 1.0


def test_spearman_rejects_a_constant_variable():
    with pytest.raises(ValueError, match="constant"):
        spearman(np.arange(10.0), np.ones(10))


def test_spearman_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least 3"):
        spearman(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_correlation_bootstrap_brackets_its_estimate():
    rng = np.random.default_rng(19)
    x = rng.normal(size=44)
    y = 0.6 * x + rng.normal(size=44)
    est = bootstrap_correlation(x, y, n_bootstrap=500)
    assert est.ci_low <= est.value <= est.ci_high


def test_correlation_ci_at_n44_is_about_as_wide_as_the_plan_predicted():
    """ANALYSIS_PLAN expects a half-width near 0.27 at this n via Fisher z.
    This checks the bootstrap lands in the same neighbourhood."""
    rng = np.random.default_rng(23)
    x = rng.normal(size=44)
    y = 0.5 * x + rng.normal(size=44)
    est = bootstrap_correlation(x, y, n_bootstrap=1500)
    half = (est.ci_high - est.ci_low) / 2
    assert 0.12 < half < 0.45


# ---------------------------------------------------------------------------
# Multiplicity
# ---------------------------------------------------------------------------


def test_bh_rejects_nothing_when_all_p_are_large():
    assert benjamini_hochberg([0.4, 0.5, 0.6, 0.9]) == [False] * 4


def test_bh_rejects_everything_when_all_p_are_tiny():
    assert benjamini_hochberg([1e-6] * 4) == [True] * 4


def test_bh_is_less_strict_than_bonferroni():
    """The reason to use it: four metrics should not cost 4x the threshold."""
    # Bonferroni would need every p below 0.0125 and rejects none of these.
    # BH's step-up rule rejects the whole family, because the largest p clears
    # its own threshold of alpha*4/4 = 0.05.
    p = [0.02, 0.03, 0.04, 0.045]
    assert not any(x <= 0.05 / 4 for x in p), "Bonferroni should reject none"
    assert all(benjamini_hochberg(p)), "BH should reject the family"


def test_bh_uses_the_step_up_rule_not_naive_thresholding():
    """A small p must stay rejected even when a larger one in the family
    fails its own threshold."""
    out = benjamini_hochberg([0.001, 0.9, 0.9, 0.9])
    assert out[0] is True
    assert out[1:] == [False, False, False]


def test_bh_rejects_invalid_p_values():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        benjamini_hochberg([0.5, 1.4])


def test_estimate_record_is_rounded_for_reporting():
    e = Estimate(0.123456, 0.1, 0.2, 378, 10000)
    assert e.as_record()["value"] == 0.1235
    assert e.as_record()["n"] == 378


# ---------------------------------------------------------------------------
# Curves, for the figure only
# ---------------------------------------------------------------------------


def test_roc_curve_starts_at_origin_and_ends_at_one_one():
    rng = np.random.default_rng(29)
    scores = rng.normal(size=200)
    labels = rng.integers(0, 2, size=200).astype(bool)
    fpr, tpr = roc_curve(scores, labels, lower_is_better=False)
    assert (fpr[0], tpr[0]) == (0.0, 0.0)
    assert fpr[-1] == pytest.approx(1.0)
    assert tpr[-1] == pytest.approx(1.0)


def test_roc_curve_is_monotone_in_both_coordinates():
    rng = np.random.default_rng(31)
    scores = rng.normal(size=300)
    labels = (rng.random(300) < 0.14).astype(bool)
    fpr, tpr = roc_curve(scores, labels, lower_is_better=False)
    assert np.all(np.diff(fpr) >= 0)
    assert np.all(np.diff(tpr) >= 0)


def test_roc_curve_area_agrees_with_the_reported_auc():
    """The figure and the headline must not disagree. auc() is computed from
    ranks, the curve from thresholds; trapezoid area over the curve has to land
    on the same number or one of them is wrong."""
    rng = np.random.default_rng(37)
    scores = rng.normal(size=400)
    labels = (rng.random(400) < 0.2).astype(bool)
    for direction in (True, False):
        fpr, tpr = roc_curve(scores, labels, lower_is_better=direction)
        assert np.trapezoid(tpr, fpr) == pytest.approx(
            auc(scores, labels, lower_is_better=direction), abs=1e-9
        )


def test_roc_curve_respects_direction():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    labels = np.array([1, 1, 0, 0]).astype(bool)
    assert np.trapezoid(*reversed(roc_curve(
        scores, labels, lower_is_better=True))) == pytest.approx(1.0)


def test_roc_curve_collapses_ties_into_one_point():
    """ipTM is rounded to two decimals upstream, so ties are dense. A curve
    that steps once per sample would imply resolution the metric lacks."""
    scores = np.array([1.0, 1.0, 1.0, 1.0, 2.0])
    labels = np.array([1, 0, 1, 0, 1]).astype(bool)
    fpr, _ = roc_curve(scores, labels, lower_is_better=False)
    assert len(fpr) == 3  # origin, the 2.0 group, the tied 1.0 group


def test_cumulative_precision_matches_precision_at_k():
    rng = np.random.default_rng(41)
    scores = rng.normal(size=120)
    labels = (rng.random(120) < 0.3).astype(bool)
    curve = cumulative_precision(scores, labels, lower_is_better=False)
    for k in (1, 7, 25, 120):
        assert curve[k - 1] == pytest.approx(
            precision_at_k(scores, labels, k, lower_is_better=False)
        )


def test_cumulative_precision_ends_at_the_base_rate():
    labels = np.array([True] * 53 + [False] * 325)
    scores = np.arange(378, dtype=float)
    curve = cumulative_precision(scores, labels, lower_is_better=False)
    assert curve[-1] == pytest.approx(53 / 378)
