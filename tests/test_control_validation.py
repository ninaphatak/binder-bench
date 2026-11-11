"""Tests for the control gate.

The gate replaces "the controls must score well", which the real controls fail
while being structurally correct. Each test names the failure it would catch.
"""

from __future__ import annotations

import numpy as np
import pytest

from binder_bench.control_validation import (
    CETUXIMAB_EPITOPE_1YY9,
    CONTROL_EPITOPES,
    EGF_EPITOPE_1IVO,
    MIN_EPITOPE_RECALL,
    EpitopeRecovery,
    assert_control_reproduces_binding_mode,
    contacted_target_residues,
    epitope_recovery,
)

# ---------------------------------------------------------------------------
# Synthetic structures. Two chains of CA-only residues on a line, with the
# binder placed next to a chosen stretch of the target, so the contacted
# residues are known exactly.
# ---------------------------------------------------------------------------


def _pdb(binder_len: int, target_len: int, contact_start: int, gap: float = 3.0) -> str:
    """Binder residues sit 3 Angstrom from target residues contact_start onward."""
    lines = []
    serial = 1
    for i in range(binder_len):
        lines.append(
            f"ATOM  {serial:5d}  CA  ALA A{i + 1:4d}    "
            f"{i * 4.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
        )
        serial += 1
    lines.append("TER")
    for j in range(target_len):
        # Target runs parallel, offset in y by `gap` only for the contact window.
        y = gap if contact_start <= j + 1 < contact_start + binder_len else 60.0
        x = (j + 1 - contact_start) * 4.0
        lines.append(
            f"ATOM  {serial:5d}  CA  ALA B{j + 1:4d}    "
            f"{x:8.3f}{y:8.3f}{0.0:8.3f}  1.00  0.00           C"
        )
        serial += 1
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


@pytest.fixture
def structure(tmp_path):
    def _make(binder_len=10, target_len=100, contact_start=20):
        p = tmp_path / f"s_{binder_len}_{target_len}_{contact_start}.pdb"
        p.write_text(_pdb(binder_len, target_len, contact_start))
        return p

    return _make


def test_contacts_are_found_where_the_binder_was_placed(structure):
    path = structure(binder_len=10, target_len=100, contact_start=20)
    res = contacted_target_residues(path, 10)
    assert res, "no contacts found at all"
    assert min(res) >= 20 and max(res) < 30


def test_a_flipped_chain_order_raises(structure):
    """The failure the metrics also guard: binder and target swapped."""
    path = structure(binder_len=10, target_len=100, contact_start=20)
    with pytest.raises(ValueError, match="Chain order may be flipped"):
        contacted_target_residues(path, 100)


def test_a_single_chain_structure_raises(tmp_path):
    """A malformed alignment cardinality header folds the complex as ONE chain
    and still yields plausible scores. It must not yield plausible contacts."""
    p = tmp_path / "one.pdb"
    p.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A   2       4.000   0.000   0.000  1.00  0.00           C\n"
        "TER\nEND\n"
    )
    with pytest.raises(ValueError, match="needs"):
        contacted_target_residues(p, 1)


# ---------------------------------------------------------------------------
# Recovery arithmetic
# ---------------------------------------------------------------------------


def _rec(recall, precision=0.5, chance=0.08, design_id="control.Human_EGF"):
    return EpitopeRecovery(
        design_id=design_id,
        structure="x.pdb",
        n_predicted=50,
        n_reference=43,
        n_recovered=int(round(recall * 43)),
        recall=recall,
        precision=precision,
        f1=0.0,
        chance_recall=chance,
    )


def test_enrichment_over_chance_is_the_headline():
    """A correct pipeline lands near 10, a chain-swapped one near 1."""
    assert _rec(0.84).enrichment_over_chance == pytest.approx(10.5, abs=0.1)
    assert _rec(0.08).enrichment_over_chance == pytest.approx(1.0, abs=0.01)


def test_recovery_against_a_reference_counts_the_intersection(structure):
    path = structure(binder_len=10, target_len=100, contact_start=20)
    # Reference deliberately half overlapping the 20-29 contact window.
    ref = frozenset({20, 21, 22, 23, 24, 90, 91, 92, 93, 94})
    r = epitope_recovery(
        path, "synthetic", 10, reference=ref, n_target_residues=100
    )
    assert r.n_reference == 10
    assert r.n_recovered == 5
    assert r.recall == pytest.approx(0.5)
    assert 0 < r.precision <= 1
    assert r.chance_recall == pytest.approx(r.n_predicted / 100)


def test_unknown_design_without_a_reference_raises(structure):
    path = structure()
    with pytest.raises(ValueError, match="No reference epitope"):
        epitope_recovery(path, "design_42", 10)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_gate_passes_at_the_measured_recalls():
    """The five observed Human_EGF recalls must not trip the gate."""
    observed = [0.84, 0.81, 0.81, 0.77, 0.70]
    assert_control_reproduces_binding_mode([_rec(r) for r in observed])


def test_gate_passes_for_cetuximab_at_its_measured_recall():
    """0.72 recall with poor precision still means the epitope was found."""
    assert_control_reproduces_binding_mode(
        [_rec(0.72, precision=0.26, design_id="control.Cetuximab_scFv")]
    )


def test_gate_fails_a_chance_level_control():
    """What a flipped chain order or a one-chain fold would look like."""
    with pytest.raises(AssertionError, match="did not reproduce its known"):
        assert_control_reproduces_binding_mode([_rec(0.08)])


def test_gate_reports_which_control_failed():
    with pytest.raises(AssertionError, match="control.Cetuximab_scFv"):
        assert_control_reproduces_binding_mode(
            [
                _rec(0.84, design_id="control.Human_EGF"),
                _rec(0.05, design_id="control.Cetuximab_scFv"),
            ]
        )


def test_gate_averages_across_models_rather_than_requiring_every_one():
    """Models legitimately vary; the pipeline is what is under test."""
    # Mean of 0.84 and 0.10 is 0.47, above the 0.40 floor.
    assert_control_reproduces_binding_mode([_rec(0.84), _rec(0.10)])


def test_gate_threshold_sits_between_chance_and_the_observed_values():
    """A floor above the observed values would fail a working pipeline; one at
    chance level would pass a broken one."""
    assert 0.08 < MIN_EPITOPE_RECALL < 0.70


def test_empty_input_is_rejected_rather_than_passing_vacuously():
    with pytest.raises(ValueError, match="No epitope recoveries"):
        assert_control_reproduces_binding_mode([])


# ---------------------------------------------------------------------------
# The reference data itself
# ---------------------------------------------------------------------------


def test_reference_epitopes_are_the_measured_sizes():
    assert len(EGF_EPITOPE_1IVO) == 43
    assert len(CETUXIMAB_EPITOPE_1YY9) == 25


def test_cetuximab_epitope_lies_entirely_in_the_domain_III_region():
    """Cetuximab binds domain III only, so every reference residue should sit
    in the C-terminal half of the ectodomain. A stray low residue number would
    mean a numbering offset crept in."""
    assert min(CETUXIMAB_EPITOPE_1YY9) > 300


def test_egf_epitope_spans_two_separated_regions():
    """EGF bridges domains I and III, so its epitope must have a wide gap, and
    that is what makes it a strong test of placement."""
    sorted_res = sorted(EGF_EPITOPE_1IVO)
    gaps = [b - a for a, b in zip(sorted_res, sorted_res[1:])]
    assert max(gaps) > 150, "expected a large gap between the two binding regions"


def test_both_controls_have_a_reference():
    assert set(CONTROL_EPITOPES) == {"control.Human_EGF", "control.Cetuximab_scFv"}


def test_epitope_residues_are_within_the_construct():
    """Our construct is 621 residues in mature numbering."""
    for name, ep in CONTROL_EPITOPES.items():
        assert min(ep) >= 1, name
        assert max(ep) <= 621, name
