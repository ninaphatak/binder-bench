"""Secondary metric S3: ipSAE, via the authors' own implementation.

### Why the reference implementation and not ours

ipSAE exists because ipTM depends on the size of the chain pair, so the same
interface scores differently in different constructs. Dunbrack, *Rēs ipSAE
loquuntur: What's wrong with AlphaFold's ipTM score and how to fix it*,
bioRxiv 2025.02.10.637595. That is the same defect our own secondary metric S1
works around crudely, by restricting the PAE average to contacting pairs; ipSAE
does it properly, by restricting to residue pairs whose interchain PAE passes a
cutoff AND recomputing the ``d0`` normalisation from the number of residues
that pass rather than from total chain length.

ANALYSIS_PLAN.md section 1a commits to running the authors' code rather than a
reimplementation, so that what we report is the published metric and not our
reading of it. ``vendor/ipsae.py`` is their version 4 (3 January 2026), MIT
licensed, copied verbatim with its header intact, and pinned by SHA-256 here.
If the file changes, this module refuses to run rather than silently reporting
numbers from different code.

### Parameters, fixed before any outcome was examined

``ipsae.py`` takes ``pae_cutoff`` and ``dist_cutoff`` as REQUIRED arguments; it
has no defaults of its own. Every invocation example in the authors' README
uses ``10 10``, for AF2, AF3 and Boltz alike, so that is what we use. The
choice is recorded here and in DECISIONS.md, dated before the first
computation, because a cutoff selected after comparing outcomes would be a
forking path.

The reported scalar is the row with ``Type == "max"``, which the README defines
as the maximum of the two asymmetric values. Both asymmetric values are kept
too, since discarding them would hide which direction drives the score.

### Input format

No format invention is required. ColabFold's per-model scores JSON already
carries exactly the keys the script reads: ``pae``, ``plddt``, ``iptm``,
``ptm``. When only a PAE matrix is available, a minimal JSON is written with
``pae`` alone; the script then computes pDockQ from a zero pLDDT vector, so
**pDockQ from such a run is meaningless and is discarded**. ipSAE itself needs
only the PAE and the coordinates, so it is unaffected.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Vendored reference implementation, pinned.
VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor"
IPSAE_SCRIPT = VENDOR_DIR / "ipsae.py"
IPSAE_SHA256 = "10cf9b08c68c91e06cb28526cf2026f47a3980c9048fd3226d13e3304eaf1c27"
IPSAE_VERSION = "4 (2026-01-03)"
IPSAE_CITATION = "Dunbrack, bioRxiv 2025.02.10.637595"

#: Required arguments with no upstream defaults. Every README example uses 10 10.
IPSAE_PAE_CUTOFF = 10.0
IPSAE_DIST_CUTOFF = 10.0


@dataclass(frozen=True)
class IpsaeResult:
    """ipSAE for one predicted complex."""

    ipsae: float          # Type == "max", the headline value
    ipsae_asym: tuple[float, ...]
    ipsae_d0chn: float
    ipsae_d0dom: float
    iptm_d0chn: float
    n_chain_pairs: int
    pae_cutoff: float
    dist_cutoff: float

    def as_record(self) -> dict:
        return {
            "ipsae": self.ipsae,
            "ipsae_asym": list(self.ipsae_asym),
            "ipsae_d0chn": self.ipsae_d0chn,
            "ipsae_d0dom": self.ipsae_d0dom,
            "iptm_d0chn": self.iptm_d0chn,
            "ipsae_pae_cutoff": self.pae_cutoff,
            "ipsae_dist_cutoff": self.dist_cutoff,
            "ipsae_version": IPSAE_VERSION,
            "ipsae_citation": IPSAE_CITATION,
        }


def assert_reference_script_unmodified(script: Path = IPSAE_SCRIPT) -> None:
    """Refuse to run if the vendored script is not the pinned version.

    The point of vendoring is that the published metric is reproducible. A
    silently edited copy would produce numbers attributed to someone else's
    method, which is worse than no number at all.
    """
    if not script.exists():
        raise FileNotFoundError(
            f"{script} is missing. It is the vendored ipSAE reference "
            f"implementation and ANALYSIS_PLAN.md commits to using it rather "
            f"than a reimplementation."
        )
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    if digest != IPSAE_SHA256:
        raise ValueError(
            f"{script} has SHA-256 {digest}, expected {IPSAE_SHA256}. The "
            f"vendored reference implementation has changed. Re-pin it "
            f"deliberately and record the change, do not run numbers from "
            f"unknown code."
        )


def _parse_output(text: str) -> IpsaeResult:
    """Pull the scalars out of ipsae.py's whitespace-aligned table.

    Columns, from the authors' README:
        Chn1 Chn2 PAE Dist Type ipSAE ipSAE_d0chn ipSAE_d0dom ipTM_af
        ipTM_d0chn pDockQ pDockQ2 LIS n0res n0chn n0dom ...
    """
    header_idx = None
    rows: list[list[str]] = []
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "Chn1":
            header_idx = fields
            continue
        if header_idx is not None and len(fields) >= 10:
            rows.append(fields)

    if header_idx is None:
        raise ValueError(
            "ipsae.py produced no result table. Output was:\n" + text[:2000]
        )
    if not rows:
        raise ValueError(
            "ipsae.py produced a header but no rows, so no chain pair was "
            "scored. Output was:\n" + text[:2000]
        )

    def col(fields: list[str], name: str) -> float:
        return float(fields[header_idx.index(name)])

    max_rows = [r for r in rows if r[header_idx.index("Type")] == "max"]
    asym_rows = [r for r in rows if r[header_idx.index("Type")] == "asym"]
    if not max_rows:
        raise ValueError(
            "ipsae.py returned no row with Type 'max', which is the "
            "symmetric summary this project reports. Rows found: "
            + ", ".join(r[header_idx.index("Type")] for r in rows)
        )

    best = max_rows[0]
    return IpsaeResult(
        ipsae=col(best, "ipSAE"),
        ipsae_asym=tuple(col(r, "ipSAE") for r in asym_rows),
        ipsae_d0chn=col(best, "ipSAE_d0chn"),
        ipsae_d0dom=col(best, "ipSAE_d0dom"),
        iptm_d0chn=col(best, "ipTM_d0chn"),
        n_chain_pairs=len(max_rows),
        pae_cutoff=col(best, "PAE"),
        dist_cutoff=col(best, "Dist"),
    )


def compute_ipsae(
    structure_path: Path | str,
    pae: np.ndarray,
    *,
    plddt: np.ndarray | None = None,
    iptm: float | None = None,
    ptm: float | None = None,
    pae_cutoff: float = IPSAE_PAE_CUTOFF,
    dist_cutoff: float = IPSAE_DIST_CUTOFF,
    python: str | None = None,
) -> IpsaeResult:
    """Run the vendored ipSAE on one predicted structure and its PAE matrix.

    Work happens in a temporary directory because ipsae.py writes its outputs
    beside the structure file, and the results Volume must not accumulate stray
    files next to the structures Step 4 reads.
    """
    assert_reference_script_unmodified()

    # Default to the interpreter we are already running under, not to whatever
    # "python3" means on PATH. ipsae.py needs numpy, and outside an activated
    # virtualenv "python3" is the system interpreter, which does not have it.
    # That made three tests fail for anyone who ran pytest as
    # ".venv/bin/python -m pytest" instead of "uv run pytest", with an error
    # pointing at the vendored script rather than at the interpreter choice.
    if python is None:
        python = sys.executable

    structure_path = Path(structure_path)
    if not structure_path.exists():
        raise FileNotFoundError(structure_path)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError(f"PAE must be square, got {pae.shape}")

    payload: dict = {"pae": np.asarray(pae, dtype=float).tolist()}
    if plddt is not None:
        payload["plddt"] = np.asarray(plddt, dtype=float).tolist()
    if iptm is not None:
        payload["iptm"] = float(iptm)
    if ptm is not None:
        payload["ptm"] = float(ptm)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        # The script derives its output names from the structure path, so the
        # structure is copied in rather than read in place.
        local_pdb = tmpdir / "model.pdb"
        local_pdb.write_bytes(structure_path.read_bytes())
        pae_json = tmpdir / "scores.json"
        pae_json.write_text(json.dumps(payload))

        proc = subprocess.run(
            [
                python,
                str(IPSAE_SCRIPT),
                str(pae_json),
                str(local_pdb),
                str(int(pae_cutoff)),
                str(int(dist_cutoff)),
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )

        out_file = tmpdir / f"model_{int(pae_cutoff):02d}_{int(dist_cutoff):02d}.txt"
        if not out_file.exists():
            raise RuntimeError(
                f"ipsae.py rc={proc.returncode} produced no output file at "
                f"{out_file.name}.\n--- stdout ---\n{proc.stdout[-3000:]}\n"
                f"--- stderr ---\n{proc.stderr[-3000:]}"
            )

        text = out_file.read_text()
        if not text.strip():
            # The script creates its output files before computing, so an
            # empty file means it died part way and the traceback is the only
            # useful information. It writes nothing to the file on failure.
            raise RuntimeError(
                f"ipsae.py rc={proc.returncode} created an EMPTY output file, "
                f"so it failed after opening it. One known cause: the "
                f"structure has no CB atoms (it collects CB, or CA only for "
                f"glycine), which leaves its coordinate array empty.\n"
                f"--- stderr ---\n{proc.stderr[-3000:]}"
            )
        return _parse_output(text)
