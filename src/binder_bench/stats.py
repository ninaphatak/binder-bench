"""Step 3 statistics, implemented before any outcome was examined.

Written while the ipSAE pass was still running, deliberately. The point of a
pre-registered plan is that the analysis is fixed in advance; writing the code
after seeing which metric separates binders would let a hundred small choices
drift toward the answer. Every function here is tested against synthetic data
with a known answer, never against the real labels.

What ANALYSIS_PLAN.md fixes, and this module implements:

- **Claim A**, binary discrimination: AUC over the 378 expressed designs, a
  10,000 resample bootstrap 95% CI, whether that CI excludes 0.5, precision at
  top k for k in {10, 20, 50} against the 14.0% base rate, and a calibration
  curve by metric decile.
- **Claim B**, affinity ranking: Spearman between metric and KD among binders
  only, bootstrap CI, Pearson on log10 KD secondary.
- **Censoring**: 9 of 53 binders sit exactly at the 1e-5 M assay ceiling, a
  decade above the highest concentration assayed. Primary is the uncensored
  binders; secondary is all of them with the ceiling values tied.
- **Multiplicity**: Benjamini-Hochberg across the metric family.

Direction convention. Some metrics are better when LOW (pae_interface_mean,
pae_interface_contacts_mean) and some when HIGH (iptm, ipsae). Every function
here takes ``lower_is_better`` explicitly and never infers it from the data,
because inferring it would silently flip a result whose sign is the finding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Resamples for every bootstrap interval in this module. Fixed by
#: ANALYSIS_PLAN.md, not tuned.
N_BOOTSTRAP = 10_000

#: Seed for every bootstrap, so intervals are reproducible rather than
#: re-rolled until they look tidy.
BOOTSTRAP_SEED = 20261111


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a bootstrap interval."""

    value: float
    ci_low: float
    ci_high: float
    n: int
    n_bootstrap: int

    def as_record(self) -> dict:
        return {
            "value": round(self.value, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "n": self.n,
            "n_bootstrap": self.n_bootstrap,
        }


def _oriented(scores: np.ndarray, lower_is_better: bool) -> np.ndarray:
    """Flip so that LARGER always means more likely to bind.

    Done once, explicitly, so the rest of the module has a single convention
    and no function has to remember which way a given metric points.
    """
    s = np.asarray(scores, dtype=float)
    return -s if lower_is_better else s


def auc(
    scores: np.ndarray, labels: np.ndarray, *, lower_is_better: bool
) -> float:
    """Area under the ROC curve, by the Mann-Whitney relationship.

    Equals the probability that a randomly chosen positive outranks a randomly
    chosen negative, with ties counted as half. Computed from ranks rather than
    by integrating a curve, because the rank form handles ties exactly and
    needs no threshold grid.
    """
    s = _oriented(scores, lower_is_better)
    y = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"AUC needs both classes, got {n_pos} positives and {n_neg} "
            f"negatives."
        )
    # Average ranks so tied scores contribute 0.5 apiece.
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # Replace tied groups by their mean rank.
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def hanley_mcneil_se(auc_value: float, n_pos: int, n_neg: int) -> float:
    """Analytic AUC standard error (Hanley & McNeil, Radiology 143, 1982).

    Reported ALONGSIDE the bootstrap interval, not instead of it. SPEC's power
    calculation used this formula, so keeping it lets the achieved precision be
    compared directly against what was promised before the data existed.
    """
    if not 0.0 <= auc_value <= 1.0:
        raise ValueError(f"AUC must be in [0, 1], got {auc_value}")
    if n_pos < 1 or n_neg < 1:
        raise ValueError("need at least one of each class")
    a = auc_value
    q1 = a / (2 - a)
    q2 = 2 * a**2 / (1 + a)
    var = (
        a * (1 - a) + (n_pos - 1) * (q1 - a**2) + (n_neg - 1) * (q2 - a**2)
    ) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def bootstrap_auc(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    lower_is_better: bool,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> Estimate:
    """Percentile bootstrap CI for AUC, resampling designs.

    STRATIFIED by label, so every resample keeps 53 positives and 325
    negatives. Unstratified resampling of a 14% base rate occasionally draws
    very few positives, which widens the interval for a reason that has nothing
    to do with the metric.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels).astype(bool)
    point = auc(s, y, lower_is_better=lower_is_better)

    pos_idx = np.flatnonzero(y)
    neg_idx = np.flatnonzero(~y)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        pi = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        ni = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([pi, ni])
        vals[i] = auc(s[idx], y[idx], lower_is_better=lower_is_better)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return Estimate(point, float(lo), float(hi), len(s), n_bootstrap)


def precision_at_k(
    scores: np.ndarray, labels: np.ndarray, k: int, *, lower_is_better: bool
) -> float:
    """Share of the top k by metric that actually bind.

    The practitioner's question: order the k best-looking designs, how many
    work. Ties at the boundary are resolved by the metric's own ordering rather
    than randomly, which is the optimistic reading, so a tie-heavy metric is
    not penalised for something it cannot control.
    """
    s = _oriented(scores, lower_is_better)
    y = np.asarray(labels).astype(bool)
    if k < 1 or k > len(s):
        raise ValueError(f"k must be in 1..{len(s)}, got {k}")
    top = np.argsort(-s, kind="mergesort")[:k]
    return float(y[top].mean())


def calibration_by_decile(
    scores: np.ndarray, labels: np.ndarray, *, lower_is_better: bool, n_bins: int = 10
) -> list[dict]:
    """Observed bind rate per metric decile, best decile first.

    ANALYSIS_PLAN's test of ranker versus threshold: a genuine ranker gives a
    monotone curve, a threshold gives a flat one with a step.
    """
    s = _oriented(scores, lower_is_better)
    y = np.asarray(labels).astype(bool)
    order = np.argsort(-s, kind="mergesort")
    out = []
    for b, chunk in enumerate(np.array_split(order, n_bins)):
        if len(chunk) == 0:
            continue
        out.append(
            {
                "decile": b + 1,
                "n": int(len(chunk)),
                "n_binders": int(y[chunk].sum()),
                "bind_rate": float(y[chunk].mean()),
                "metric_min": float(np.min(np.asarray(scores)[chunk])),
                "metric_max": float(np.max(np.asarray(scores)[chunk])),
            }
        )
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, ties averaged."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if len(a) < 3:
        raise ValueError("need at least 3 points for a rank correlation")

    def rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=float)
        r[order] = np.arange(1, len(x) + 1, dtype=float)
        uniq, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, r)
        return (sums / counts)[inv]

    ra, rb = rank(a) - rank(a).mean(), rank(b) - rank(b).mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    if denom == 0:
        raise ValueError(
            "a rank correlation is undefined when one variable is constant"
        )
    return float((ra * rb).sum() / denom)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float) - np.mean(a)
    b = np.asarray(b, dtype=float) - np.mean(b)
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom == 0:
        raise ValueError("correlation undefined when one variable is constant")
    return float((a * b).sum() / denom)


def bootstrap_correlation(
    a: np.ndarray,
    b: np.ndarray,
    *,
    method: str = "spearman",
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> Estimate:
    """Percentile bootstrap CI for a correlation, resampling pairs."""
    fn = {"spearman": spearman, "pearson": pearson}[method]
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    point = fn(a, b)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(a), size=len(a))
        try:
            vals.append(fn(a[idx], b[idx]))
        except ValueError:
            # A resample can be constant; it carries no information about the
            # correlation and is dropped rather than counted as zero.
            continue
    if len(vals) < n_bootstrap // 2:
        raise ValueError(
            f"only {len(vals)} of {n_bootstrap} resamples were usable, so the "
            f"interval would not be trustworthy"
        )
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return Estimate(point, float(lo), float(hi), len(a), len(vals))


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Which hypotheses survive BH correction at ``alpha``.

    Applied across the metric family, as ANALYSIS_PLAN specifies, so testing
    four metrics does not inflate the chance that one looks significant.
    """
    p = np.asarray(pvalues, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    thresholds = alpha * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresholds
    # Everything up to the largest index that passes is rejected.
    cutoff = np.flatnonzero(passed)
    out = np.zeros(m, dtype=bool)
    if len(cutoff):
        out[order[: cutoff[-1] + 1]] = True
    return out.tolist()
