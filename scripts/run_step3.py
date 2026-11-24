#!/usr/bin/env python
"""Run the pre-registered Step 3 analysis ONCE and write its tables.

Usage:
    uv run python scripts/run_step3.py --metrics results/tables/per_design_metrics.csv
    uv run python scripts/run_step3.py --folds <dir>

--metrics reads the per-design table committed to this repository, so the
published result can be recomputed without a GPU and without the fold records.
--folds reads those records straight off the Modal Volume. Both paths apply the
same population, the same refusals and the same estimator, and both produce the
same numbers.

Everything this does is fixed by ANALYSIS_PLAN.md. It takes no decisions based
on a result and has no branches on the value of a statistic. All four metrics
are written whether they flatter the hypothesis or not.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from binder_bench.step3 import (
    METRICS,
    apply_bh,
    claim_a,
    claim_b,
    load_joined,
    load_joined_from_metrics,
)
from binder_bench.stats import calibration_by_decile


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[step3] wrote {path} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--metrics",
        help="per-design metrics CSV, e.g. results/tables/per_design_metrics.csv",
    )
    src.add_argument("--folds", help="directory of fold records from the Volume")
    ap.add_argument("--designs", default="data/designs.parquet")
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--n-bootstrap", type=int, default=None)
    args = ap.parse_args()

    if args.metrics:
        j = load_joined_from_metrics(args.metrics, args.designs)
        print(f"[step3] metrics from {args.metrics}")
    else:
        j = load_joined(args.folds, args.designs)
        print(f"[step3] metrics from fold records under {args.folds}")
    print(
        f"[step3] joined {j.n} expressed designs | "
        f"{int(j.binds.sum())} binders | "
        f"{int((j.binds & j.kd_at_ceiling).sum())} at the KD ceiling"
    )

    kw = {} if args.n_bootstrap is None else {"n_bootstrap": args.n_bootstrap}
    a = apply_bh(claim_a(j, **kw), "p_vs_half")
    b = apply_bh(claim_b(j, **kw), "fisher_z_p")

    out = Path(args.out)
    _write_csv(out / "claim_a_discrimination.csv", a)
    _write_csv(out / "claim_b_affinity_ranking.csv", b)

    calib: list[dict] = []
    for name, lower_is_better, role in METRICS:
        for row in calibration_by_decile(
            j.metrics[name], j.binds, lower_is_better=lower_is_better
        ):
            calib.append({"metric": name, "role": role, **row})
    _write_csv(out / "calibration_by_decile.csv", calib)

    (out / "step3_summary.json").write_text(
        json.dumps({"claim_a": a, "claim_b": b}, indent=2, default=str)
    )
    print(f"[step3] wrote {out / 'step3_summary.json'}")


if __name__ == "__main__":
    main()
