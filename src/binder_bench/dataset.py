"""Step 1: build data/designs.parquet and data/submissions.parquet.

Pure, no Modal imports, no network. Reads the cloned competition data and
writes two small tables. Every exclusion is counted and printed, because
SPEC.md forbids dropping designs without logging the count and reason.

Run with:
    uv run python -m binder_bench.dataset --competition-dir ~/molecules/external/egfr_competition_2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Label definition. Written down here, with its source, because SPEC.md
# requires the threshold and its provenance recorded before anything is
# computed, and forbids changing it after seeing an outcome.
# ---------------------------------------------------------------------------

BINDS_DEFINITION = """\
binds := (result_summary.csv `binding` == "true")

Source: Adaptyv's own call from the Round 2 binding assay, not a leaderboard
rank. Their method, from the competition README: Bio-Layer Interferometry,
multi-cycle kinetics against EGFR, antigen flowed at 1000 nM down to 10 nM, in
duplicate, HBS-T with 0.5% BSA at 25 C. Signals baseline-corrected and fitted
with a 1:1 binding model per replicate; equilibrium analysis used where the
maximum signal fell below the quantifiable threshold or kinetics were too fast
for the instrument.

We take their binary call rather than imposing a KD cutoff of our own, because
a cutoff we invented would be a threshold tuned by us on an outcome we can see,
which the hard rules forbid.
"""

# Nine of the 53 binders carry KD exactly 1.000e-05 M. The assay's highest
# antigen concentration was 1000 nM = 1e-6 M, so 1e-5 sits a full decade above
# anything actually flowed over the probe. Nine identical values at that point
# is a ceiling, not a measurement. Flagged per design so Claim B can treat
# them as right-censored rather than silently correlating against a cap.
KD_CEILING = 1e-5


@dataclass
class BuildReport:
    """Counts for DECISIONS.md. Every number here must be logged."""

    rows_in_csv: int = 0
    controls: int = 0
    designs: int = 0
    excluded_not_expressed: int = 0
    expressed: int = 0
    binders: int = 0
    kd_present: int = 0
    kd_at_ceiling: int = 0
    submissions_total: int = 0
    submissions_tested: int = 0
    submissions_untested: int = 0
    unmatched_tested: list[str] = field(default_factory=list)
    neutralisation_designs: int = 0

    def render(self) -> str:
        return json.dumps(
            {k: v for k, v in self.__dict__.items() if k != "unmatched_tested"},
            indent=2,
        )


def _read_result_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))


def _design_id(row: dict[str, str]) -> str:
    """Stable per-design key. Everything downstream is keyed by this.

    Uses username.name where available, which is how the competition's own
    files and the neutralisation package name things. Falls back to a sequence
    hash for the controls, which have no username.
    """
    user = row["username"].strip()
    name = row["name"].strip()
    if user:
        return f"{user}.{name}"
    digest = hashlib.sha256(row["sequence"].strip().encode()).hexdigest()[:8]
    return f"control.{name or digest}"


def _nullable_float(raw: str) -> float | None:
    raw = raw.strip()
    return float(raw) if raw else None


def build_designs(competition_dir: Path, report: BuildReport) -> pd.DataFrame:
    rows = _read_result_summary(competition_dir / "results" / "result_summary.csv")
    report.rows_in_csv = len(rows)

    records = []
    for row in rows:
        is_control = not row["username"].strip()
        expressed = row["expression"].strip() != "none"
        kd = _nullable_float(row["kd"])
        records.append(
            {
                "design_id": _design_id(row),
                "sequence": row["sequence"].strip(),
                # Self-reported by the competitor, messy, kept verbatim per SPEC.
                "design_method": row["design_models"].strip(),
                "expressed": expressed,
                "expression_level": row["expression"].strip(),
                "binds": row["binding"].strip() == "true",
                "binding_call_raw": row["binding"].strip(),
                "binding_strength": row["binding_strength"].strip(),
                "kd": kd,
                "kd_at_ceiling": kd is not None and abs(kd - KD_CEILING) < 1e-12,
                # Not available: the neutralisation package ships only raw
                # sensorgram traces, with no published coefficient. Left null.
                "neutralisation_coefficient": None,
                "is_control": is_control,
                # Adaptyv's published metrics. COMPARISON POINT ONLY. Never our
                # prediction: different model, version and config (ColabFold
                # with templates, 5 models, 3 recycles, 3 seeds, best of 15).
                # Suffixed so they cannot be mistaken for ours downstream.
                "pae_interaction_adaptyv": _nullable_float(row["pae_interaction"]),
                "iptm_adaptyv": _nullable_float(row["iptm"]),
                "plddt_adaptyv": _nullable_float(row["plddt"]),
                "username": row["username"].strip(),
                "competition_name": row["name"].strip(),
            }
        )

    df = pd.DataFrame.from_records(records)

    report.controls = int(df["is_control"].sum())
    designs = df[~df["is_control"]].copy()
    report.designs = len(designs)

    # DENOMINATOR DECISION, per SPEC.md: restrict to designs that expressed. A
    # design that failed to express says nothing about whether the structure
    # prediction was right; that is a different failure mode (solubility,
    # aggregation, codon usage). The analysis is therefore conditional on
    # expression, which is a stated threat to validity.
    report.excluded_not_expressed = int((~designs["expressed"]).sum())
    report.expressed = int(designs["expressed"].sum())
    report.binders = int(designs.loc[designs["expressed"], "binds"].sum())
    report.kd_present = int(designs["kd"].notna().sum())
    report.kd_at_ceiling = int(designs["kd_at_ceiling"].sum())

    # Keep controls in the table, flagged. They are needed for the Step 2
    # positive control and must never enter the Step 3 statistics.
    return df


def build_submissions(
    embeddings_zip: Path, designs: pd.DataFrame, report: BuildReport
) -> pd.DataFrame:
    """All 1131 submitted sequences, flagged by whether they were tested.

    This is what makes the truncation picture a measurement rather than an
    assumption: the untested designs were never assayed and never will have
    binding labels, so the only honest comparison is on features computable
    from sequence alone.
    """
    with zipfile.ZipFile(embeddings_zip) as zf:
        with zf.open("esm2_650m.csv") as fh:
            reader = csv.reader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"))
            next(reader)  # header: sequence, then 1280 embedding dims
            submitted = [r[0].strip() for r in reader]

    tested = set(designs.loc[~designs["is_control"], "sequence"])
    report.submissions_total = len(submitted)

    df = pd.DataFrame({"sequence": submitted})
    df["tested"] = df["sequence"].isin(tested)
    report.submissions_tested = int(df["tested"].sum())
    report.submissions_untested = int((~df["tested"]).sum())
    report.unmatched_tested = sorted(tested - set(submitted))

    if report.unmatched_tested:
        raise ValueError(
            f"{len(report.unmatched_tested)} tested sequences are absent from the "
            f"1131 submissions. The truncation comparison assumes tested is a "
            f"subset of submitted. Investigate before continuing."
        )

    return df


def count_neutralisation(neutralisation_zip: Path | None, report: BuildReport) -> None:
    if neutralisation_zip is None or not neutralisation_zip.exists():
        return
    with zipfile.ZipFile(neutralisation_zip) as zf:
        report.neutralisation_designs = sum(
            1 for n in zf.namelist() if "/raw_data/" in n and n.endswith(".csv")
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--competition-dir", type=Path, required=True)
    ap.add_argument("--embeddings-zip", type=Path, required=True)
    ap.add_argument("--neutralisation-zip", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    args = ap.parse_args()

    report = BuildReport()
    designs = build_designs(args.competition_dir.expanduser(), report)
    submissions = build_submissions(
        args.embeddings_zip.expanduser(), designs, report
    )
    count_neutralisation(
        args.neutralisation_zip.expanduser() if args.neutralisation_zip else None,
        report,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    designs.to_parquet(args.out_dir / "designs.parquet", index=False)
    submissions.to_parquet(args.out_dir / "submissions.parquet", index=False)
    (args.out_dir / "binds_definition.txt").write_text(BINDS_DEFINITION)

    print(report.render())
    print("\nwrote:")
    print(f"  {args.out_dir / 'designs.parquet'}  ({len(designs)} rows)")
    print(f"  {args.out_dir / 'submissions.parquet'}  ({len(submissions)} rows)")
    print(f"  {args.out_dir / 'binds_definition.txt'}")


if __name__ == "__main__":
    main()
