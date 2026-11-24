#!/usr/bin/env python
"""Export one row per folded design, so the analysis is reproducible offline.

Why this exists. The fold records live on a Modal Volume and the structures and
raw PAE matrices are gigabytes, none of which belongs in git. Without this table
a reader can see the OUTPUT tables in ``results/tables/`` but has no way to
recompute them, which makes the headline an assertion rather than a check.

This writes the pre-registered estimator for each metric (the mean across the
five models) plus the spread and the per-model count, so a reader can run
``scripts/run_step3.py --metrics`` and reproduce every number in the README
without renting a GPU or trusting this repository's author.

It reads fold records and nothing else. It never touches an outcome, so running
it cannot influence a statistic.

Usage:
    uv run python scripts/export_per_design_metrics.py \
        --folds <dir> --out results/tables/per_design_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

#: Columns in output order. Each is (column name, how to pull it from a record).
#: Written out explicitly rather than by dumping the record, so adding a field
#: to the fold record never silently changes the published table.
COLUMNS: tuple[tuple[str, str], ...] = (
    ("design_id", "design_id"),
    ("status", "status"),
    ("binder_length", "binder_length"),
    ("target_length", "target_length"),
    ("complex_length", "complex_length"),
    ("n_models", "metrics.n_models"),
    # The four pre-registered metrics, primary estimator first.
    ("pae_interface_mean", "metrics.pae_interface_mean__mean"),
    ("pae_interface_mean__sd", "metrics.pae_interface_mean__sd"),
    ("pae_interface_mean__min", "metrics.pae_interface_mean__min"),
    ("pae_interface_mean__max", "metrics.pae_interface_mean__max"),
    ("pae_interface_contacts_mean",
     "secondary_metrics.pae_interface_contacts_mean__mean"),
    ("n_models_with_contacts", "secondary_metrics.n_models_with_contacts"),
    ("iptm", "metrics.iptm__mean"),
    ("iptm__sd", "metrics.iptm__sd"),
    ("ipsae", "secondary_metrics.ipsae.ipsae__mean"),
    ("ipsae__min", "secondary_metrics.ipsae.ipsae__min"),
    ("ipsae__max", "secondary_metrics.ipsae.ipsae__max"),
    ("ipsae__n_models", "secondary_metrics.ipsae.ipsae__n_models"),
    # Context, not used by either claim.
    ("ptm", "metrics.ptm__mean"),
    ("plddt_binder_mean", "metrics.plddt_binder_mean__mean"),
    # Kept only for comparability with the competition's published numbers,
    # which were read off rank_001 rather than averaged over five models.
    ("pae_interface_mean__rank001", "metrics.pae_interface_mean__rank001"),
    ("iptm__rank001", "metrics.iptm__rank001"),
    # Provenance.
    ("gpu_type", "gpu_type"),
    ("colabfold_version", "colabfold_version"),
)


def _dig(record: dict, dotted: str):
    """Follow a dotted path, returning None at the first missing key."""
    cur = record
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def export(folds_dir: Path, out_path: Path) -> int:
    paths = sorted(folds_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"no fold records under {folds_dir}")

    rows = []
    for p in paths:
        rec = json.loads(p.read_text())
        rows.append({name: _dig(rec, path) for name, path in COLUMNS})

    # Sort by design_id so the file is stable across runs and a diff is
    # meaningful. Glob order is filesystem order and is not guaranteed.
    rows.sort(key=lambda r: r["design_id"] or "")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[c[0] for c in COLUMNS])
        w.writeheader()
        w.writerows(rows)

    missing = [
        r["design_id"] for r in rows
        if r["pae_interface_mean"] is None or r["ipsae"] is None
    ]
    if missing:
        print(f"[export] WARNING {len(missing)} rows missing a metric: {missing[:5]}")
    print(f"[export] wrote {out_path} ({len(rows)} rows)")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", required=True)
    ap.add_argument("--out", default="results/tables/per_design_metrics.csv")
    args = ap.parse_args()
    export(Path(args.folds), Path(args.out))


if __name__ == "__main__":
    main()
