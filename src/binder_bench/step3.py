"""Step 3: run the pre-registered analysis once.

This module JOINS the predictions to the assay outcomes. That join is the
moment the pre-registration stops protecting anything, so everything it does
was fixed in advance:

- the population: the 378 expressed designs, controls excluded
- the four metrics and which direction each points
- the two co-primary claims and the statistic for each
- how the 9 ceiling-censored KD values are handled
- Benjamini-Hochberg across the metric family

Nothing here chooses anything on the basis of a result. There are no branches
on the value of a statistic. The outputs are written for all four metrics
whether they are flattering or not, with the primary identified as primary.

Run it with ``scripts/run_step3.py``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import KD_CEILING
from .stats import (
    N_BOOTSTRAP,
    auc,
    benjamini_hochberg,
    bootstrap_auc,
    bootstrap_correlation,
    calibration_by_decile,
    hanley_mcneil_se,
    spearman,
)

#: The four metrics, their direction, and their role. Fixed by ANALYSIS_PLAN.md
#: sections 1 and 1a. Order is the reporting order; the primary is first.
METRICS: tuple[tuple[str, bool, str], ...] = (
    ("pae_interface_mean", True, "primary"),
    ("pae_interface_contacts_mean", True, "secondary_S1"),
    ("iptm", False, "secondary_S2"),
    ("ipsae", False, "secondary_S3"),
)

#: Precision-at-k values fixed in advance.
K_VALUES = (10, 20, 50)


@dataclass(frozen=True)
class Joined:
    """Predictions joined to outcomes, for the pre-registered population."""

    design_ids: list[str]
    metrics: dict[str, np.ndarray]
    binds: np.ndarray
    kd: np.ndarray            # NaN where absent
    kd_at_ceiling: np.ndarray

    @property
    def n(self) -> int:
        return len(self.design_ids)


def _metric_value(record: dict, name: str) -> float:
    """Pull one metric from a fold record, by its pre-registered name."""
    m = record.get("metrics", {})
    s = record.get("secondary_metrics", {})
    if name == "pae_interface_mean":
        return float(m["pae_interface_mean__mean"])
    if name == "iptm":
        return float(m["iptm__mean"])
    if name == "pae_interface_contacts_mean":
        return float(s["pae_interface_contacts_mean__mean"])
    if name == "ipsae":
        return float(s["ipsae"]["ipsae__mean"])
    raise KeyError(f"unknown metric {name!r}")


def load_joined(folds_dir: Path | str, designs_parquet: Path | str) -> Joined:
    """Join fold records to assay outcomes over the pre-registered population.

    Refuses rather than guessing on any mismatch: a design with no fold, a fold
    with no design, an incomplete model set, or a missing metric. Silently
    dropping any of those would change the denominator, which CLAUDE.md names
    as a stop-and-ask.
    """
    import pandas as pd

    df = pd.read_parquet(designs_parquet)
    pool = df[(~df["is_control"]) & df["expressed"]].copy()

    records: dict[str, dict] = {}
    for path in sorted(Path(folds_dir).glob("*.json")):
        rec = json.loads(path.read_text())
        records[rec["design_id"]] = rec

    wanted = list(pool["design_id"])
    missing = [d for d in wanted if d not in records]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(wanted)} expressed designs have no fold "
            f"record, e.g. {missing[:3]}. The denominator is pre-registered at "
            f"378, so this must be fixed rather than worked around."
        )

    rows = []
    for d in wanted:
        rec = records[d]
        if rec["status"] != "folded":
            raise ValueError(f"{d}: status is {rec['status']!r}, not 'folded'")
        if rec["metrics"]["n_models"] != 5:
            raise ValueError(
                f"{d}: {rec['metrics']['n_models']} models, expected 5. The "
                f"primary estimator is the mean across five."
            )
        n_ipsae = rec["secondary_metrics"]["ipsae"]["ipsae__n_models"]
        if n_ipsae != 5:
            raise ValueError(
                f"{d}: ipSAE averaged {n_ipsae} models, expected 5."
            )
        rows.append({name: _metric_value(rec, name) for name, _, _ in METRICS})

    metrics = {
        name: np.array([r[name] for r in rows], dtype=float)
        for name, _, _ in METRICS
    }
    for name, arr in metrics.items():
        if np.isnan(arr).any():
            raise ValueError(
                f"{name} has {int(np.isnan(arr).sum())} NaN values. A NaN here "
                f"would be silently excluded by the statistics."
            )

    return Joined(
        design_ids=wanted,
        metrics=metrics,
        binds=pool["binds"].to_numpy().astype(bool),
        kd=pool["kd"].to_numpy(dtype=float),
        kd_at_ceiling=pool["kd_at_ceiling"].to_numpy().astype(bool),
    )


def _normal_two_sided_p(z: float) -> float:
    """Two-sided p from a z statistic, via the error function."""
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def claim_a(j: Joined, *, n_bootstrap: int = N_BOOTSTRAP) -> list[dict]:
    """Claim A, binary discrimination, for every metric.

    AUC with a stratified bootstrap CI, the Hanley-McNeil analytic SE kept
    alongside so achieved precision can be checked against SPEC's advance
    power calculation, precision at top k, and the decile calibration curve.
    """
    n_pos = int(j.binds.sum())
    n_neg = int((~j.binds).sum())
    base_rate = n_pos / j.n

    out = []
    for name, lower_is_better, role in METRICS:
        scores = j.metrics[name]
        est = bootstrap_auc(
            scores, j.binds, lower_is_better=lower_is_better,
            n_bootstrap=n_bootstrap,
        )
        se = hanley_mcneil_se(est.value, n_pos, n_neg)
        p = _normal_two_sided_p((est.value - 0.5) / se) if se > 0 else 1.0
        out.append(
            {
                "metric": name,
                "role": role,
                "lower_is_better": lower_is_better,
                "n": j.n,
                "n_binders": n_pos,
                "n_non_binders": n_neg,
                "base_rate": round(base_rate, 4),
                "auc": round(est.value, 4),
                "auc_ci_low": round(est.ci_low, 4),
                "auc_ci_high": round(est.ci_high, 4),
                "auc_ci_excludes_half": bool(
                    est.ci_low > 0.5 or est.ci_high < 0.5
                ),
                "hanley_mcneil_se": round(se, 4),
                "p_vs_half": p,
                **{
                    f"precision_at_{k}": round(
                        float(
                            np.mean(
                                j.binds[
                                    np.argsort(
                                        -(scores if not lower_is_better else -scores),
                                        kind="mergesort",
                                    )[:k]
                                ]
                            )
                        ),
                        4,
                    )
                    for k in K_VALUES
                },
                "n_bootstrap": est.n_bootstrap,
            }
        )
    return out


def claim_b(j: Joined, *, n_bootstrap: int = N_BOOTSTRAP) -> list[dict]:
    """Claim B, affinity ranking among binders only.

    Primary population is the UNCENSORED binders. 9 of the 53 sit exactly at
    the 1e-5 M ceiling, a decade above the highest concentration assayed, so
    their true KD is unknown and only bounded. Secondary includes all 53 with
    the ceiling values tied, as pre-registered. Both are reported whichever is
    more favourable.
    """
    binder = j.binds & np.isfinite(j.kd)
    uncensored = binder & (~j.kd_at_ceiling)

    out = []
    for name, lower_is_better, role in METRICS:
        scores = j.metrics[name]
        for population, mask in (
            ("uncensored_binders_primary", uncensored),
            ("all_binders_secondary", binder),
        ):
            x = scores[mask]
            y = j.kd[mask]
            n = int(mask.sum())
            if n < 4:
                out.append(
                    {
                        "metric": name, "role": role, "population": population,
                        "n": n, "spearman": None,
                        "note": "too few points for a rank correlation",
                    }
                )
                continue
            est = bootstrap_correlation(
                x, y, method="spearman", n_bootstrap=n_bootstrap
            )
            # Fisher z for the p-value, as ANALYSIS_PLAN specifies.
            r = est.value
            z = math.atanh(max(min(r, 0.999999), -0.999999))
            se_z = 1.0 / math.sqrt(max(n - 3, 1))
            out.append(
                {
                    "metric": name,
                    "role": role,
                    "population": population,
                    "n": n,
                    # Sign convention: a metric that correctly ranks affinity
                    # should correlate POSITIVELY with KD when low scores mean
                    # confident, because low KD is tight binding.
                    "spearman": round(r, 4),
                    "ci_low": round(est.ci_low, 4),
                    "ci_high": round(est.ci_high, 4),
                    "ci_excludes_zero": bool(
                        est.ci_low > 0 or est.ci_high < 0
                    ),
                    "fisher_z_p": _normal_two_sided_p(z / se_z),
                    "n_bootstrap": est.n_bootstrap,
                }
            )
    return out


def apply_bh(rows: list[dict], p_key: str, alpha: float = 0.05) -> list[dict]:
    """Benjamini-Hochberg across the metric family, in place on a copy."""
    usable_idx = [i for i, r in enumerate(rows) if r.get(p_key) is not None]
    flags = benjamini_hochberg(
        [rows[i][p_key] for i in usable_idx], alpha=alpha
    )
    flag_by_index = dict(zip(usable_idx, flags))

    out = []
    for i, original in enumerate(rows):
        # Index, not id(): rebinding to a copy before looking the flag up
        # would miss every row, since the copy has a different identity.
        r = dict(original)
        r["bh_significant"] = flag_by_index.get(i)
        if r.get(p_key) is not None:
            r[p_key] = round(r[p_key], 6)
        out.append(r)
    return out
