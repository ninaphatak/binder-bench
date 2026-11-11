"""Tests for the ipSAE runner.

The metric is computed by the authors' own code, so these tests cover the parts
that are OURS: the pin on the vendored script, the parsing of its table, and
the refusal paths. They deliberately do not re-test ipSAE's arithmetic, because
ANALYSIS_PLAN.md commits to reporting the published metric rather than our
reading of it, and a test asserting our expectation of their number would
defeat that.
"""

from __future__ import annotations

import numpy as np
import pytest

from binder_bench.ipsae_runner import (
    IPSAE_DIST_CUTOFF,
    IPSAE_PAE_CUTOFF,
    IPSAE_SCRIPT,
    IPSAE_SHA256,
    _parse_output,
    assert_reference_script_unmodified,
    compute_ipsae,
)

# A realistic two-row table in the authors' column order.
TABLE = """
Chn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom  ipTM_af  ipTM_d0chn     pDockQ     pDockQ2    LIS       n0res  n0chn  n0dom   d0res   d0chn   d0dom  nres1   nres2   dist1   dist2  Model
A    B      10   10  asym   0.677300   0.691900   0.685000   0.440000   0.691900   0.120000   0.330000  0.210000     40    674    120   3.100   8.900   5.200     32     47     30     45  model
B    A      10   10  asym   0.134900   0.201000   0.190000   0.440000   0.691900   0.120000   0.330000  0.210000     40    674    120   3.100   8.900   5.200     47     32     45     30  model
A    B      10   10  max    0.677300   0.691900   0.685000   0.440000   0.691900   0.120000   0.330000  0.210000     40    674    120   3.100   8.900   5.200     32     47     30     45  model
"""


def _small_complex(path, binder_len=12, target_len=40):
    """A tiny two-chain PDB with a real contact, enough for ipsae.py to score.

    GLYCINE deliberately: ipsae.py builds its coordinate array from CB atoms,
    accepting CA only when the residue is GLY (vendor/ipsae.py line 384). A
    CA-only alanine chain leaves that array empty and the script dies. Real
    AF2 output always carries full side chains, so this only bites fixtures.
    """
    lines = []
    serial = 1
    for i in range(binder_len):
        lines.append(
            f"ATOM  {serial:5d}  CA  GLY A{i + 1:4d}    "
            f"{i * 3.8:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 80.00           C"
        )
        serial += 1
    lines.append("TER")
    for j in range(target_len):
        lines.append(
            f"ATOM  {serial:5d}  CA  GLY B{j + 1:4d}    "
            f"{j * 3.8:8.3f}{5.0:8.3f}{0.0:8.3f}  1.00 80.00           C"
        )
        serial += 1
    lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return binder_len + target_len


# ---------------------------------------------------------------------------
# The pin
# ---------------------------------------------------------------------------


def test_vendored_script_matches_its_pinned_checksum():
    """The whole value of vendoring is that the published metric is
    reproducible. A silently edited copy would attribute our numbers to
    someone else's method."""
    assert_reference_script_unmodified()


def test_modified_script_is_refused(tmp_path):
    fake = tmp_path / "ipsae.py"
    fake.write_text("print('not the real thing')\n")
    with pytest.raises(ValueError, match="has SHA-256"):
        assert_reference_script_unmodified(fake)


def test_missing_script_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="vendored ipSAE"):
        assert_reference_script_unmodified(tmp_path / "absent.py")


def test_checksum_constant_matches_the_file_on_disk():
    """Guards against updating the file and forgetting the constant."""
    import hashlib

    assert hashlib.sha256(IPSAE_SCRIPT.read_bytes()).hexdigest() == IPSAE_SHA256


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_the_max_row_is_what_gets_reported():
    r = _parse_output(TABLE)
    assert r.ipsae == pytest.approx(0.6773)


def test_both_asymmetric_values_are_kept():
    """Discarding them would hide which direction drives the score, and for a
    53-residue binder against a 621-residue target the two differ fivefold."""
    r = _parse_output(TABLE)
    assert sorted(r.ipsae_asym) == [pytest.approx(0.1349), pytest.approx(0.6773)]


def test_cutoffs_are_read_back_from_the_output_not_assumed():
    r = _parse_output(TABLE)
    assert r.pae_cutoff == pytest.approx(10.0)
    assert r.dist_cutoff == pytest.approx(10.0)


def test_a_table_with_no_max_row_is_rejected():
    only_asym = "\n".join(
        line for line in TABLE.splitlines() if "max" not in line
    )
    with pytest.raises(ValueError, match="no row with Type 'max'"):
        _parse_output(only_asym)


def test_output_with_no_table_is_rejected():
    with pytest.raises(ValueError, match="no result table"):
        _parse_output("ipsae.py: something went wrong\n")


def test_header_without_rows_is_rejected():
    header = [line for line in TABLE.splitlines() if line.startswith("Chn1")][0]
    with pytest.raises(ValueError, match="no rows"):
        _parse_output(header + "\n")


# ---------------------------------------------------------------------------
# End to end against the real script
# ---------------------------------------------------------------------------


def test_runs_end_to_end_on_a_small_complex(tmp_path):
    pdb = tmp_path / "c.pdb"
    n = _small_complex(pdb)
    pae = np.full((n, n), 4.0)  # confident everywhere
    r = compute_ipsae(pdb, pae, plddt=np.full(n, 85.0))
    assert 0.0 <= r.ipsae <= 1.0
    assert r.pae_cutoff == IPSAE_PAE_CUTOFF
    assert r.dist_cutoff == IPSAE_DIST_CUTOFF


def test_a_confident_interface_scores_above_a_hopeless_one(tmp_path):
    """Directional sanity only, not a check of their arithmetic."""
    pdb = tmp_path / "c.pdb"
    n = _small_complex(pdb)
    good = compute_ipsae(pdb, np.full((n, n), 2.0), plddt=np.full(n, 90.0))
    bad = compute_ipsae(pdb, np.full((n, n), 29.0), plddt=np.full(n, 40.0))
    assert good.ipsae > bad.ipsae


def test_non_square_pae_is_rejected(tmp_path):
    pdb = tmp_path / "c.pdb"
    _small_complex(pdb)
    with pytest.raises(ValueError, match="must be square"):
        compute_ipsae(pdb, np.ones((10, 11)))


def test_missing_structure_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        compute_ipsae(tmp_path / "nope.pdb", np.ones((5, 5)))


def test_record_carries_version_and_citation(tmp_path):
    """Provenance: a number from someone else's method must say whose."""
    pdb = tmp_path / "c.pdb"
    n = _small_complex(pdb)
    rec = compute_ipsae(pdb, np.full((n, n), 5.0)).as_record()
    assert "ipsae_version" in rec
    assert "Dunbrack" in rec["ipsae_citation"]
    assert rec["ipsae_pae_cutoff"] == 10.0
