"""Tests for the geometric interface metrics.

Two things are being tested, in the style of test_metrics.py: that the guards
FAIL when the chain layout is wrong, and that the numbers are right against
geometry that can be worked out by hand or in closed form.

Every structure here is synthetic PDB or mmCIF text written into tmp_path with
atoms at chosen coordinates, so the expected contact counts are exact integers
and the expected SASAs are analytic. Nothing is compared against a reference
implementation, because there is no reference implementation in this project's
dependency set and adding one would mean trusting it.

SASA tolerances. The Shrake-Rupley SASA is a quadrature on a finite point set,
so it is only exact in the limit. The two cases used here:

- An atom with no neighbours has every test point accessible, so its SASA is
  exactly 4*pi*(r+probe)^2 for any point count. Tested to float equality.
- Two overlapping spheres have a closed-form answer: with equal expanded radii
  R and centre separation d < 2R, each sphere loses a spherical cap of area
  2*pi*R*(R - d/2), so the total accessible area is 4*pi*R^2 + 2*pi*R*d. At the
  default 960 points the implementation lands within 0.2% of that across
  d = 2 to 5 Angstroms, so the tests accept 0.5%, which is loose enough not to
  be flaky under a different numpy and tight enough that a real error in the
  burial test (which would move the answer by whole percent) fails it. At 92
  points, the original paper's density, the error grows to about 2%, and the
  tests assert that coarser sampling is worse, which is the direction a
  quadrature error must go.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from binder_bench.geometry import (
    CONTACT_CUTOFF,
    PROBE_RADIUS,
    VDW_RADII,
    geometry_metrics,
    golden_spiral_points,
    interchain_contacts,
    parse_structure,
    shrake_rupley_sasa,
)
from binder_bench.target import EXPECTED_LENGTH

# --------------------------------------------------------------------------
# Synthetic structure builders
# --------------------------------------------------------------------------


def _atom_line(
    serial: int,
    name: str,
    resname: str,
    chain: str,
    resseq: int,
    xyz: tuple[float, float, float],
    element: str,
    record: str = "ATOM  ",
    altloc: str = " ",
    icode: str = " ",
    with_element: bool = True,
) -> str:
    """One PDB ATOM record, columns per the PDB v3.3 specification.

    ``name`` is centred in columns 13-16, which is right for the one-letter
    elements that make up a protein. Pass a 4-character string to place it
    verbatim instead, which is how a two-letter element such as ``"SE  "`` has
    to be written: PDB right-justifies the symbol in columns 13-14.

    ``with_element=False`` truncates the record at column 66, as a file written
    before the element columns existed would be.
    """
    x, y, z = xyz
    name_field = name if len(name) == 4 else f"{name:^4s}"
    line = (
        f"{record:<6s}{serial:5d} {name_field}{altloc:1s}{resname:>3s} "
        f"{chain:1s}{resseq:4d}{icode:1s}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}"
    )
    assert len(line) == 66, f"bad PDB record width {len(line)}"
    if not with_element:
        return line
    line += f"          {element:>2s}"
    assert len(line) == 78, f"bad PDB record width {len(line)}"
    return line


def _write_pdb(path, atoms: list[dict]) -> str:
    """``atoms`` is a list of kwargs for _atom_line, serial filled in."""
    lines = ["REMARK   1 SYNTHETIC TEST STRUCTURE"]
    for i, a in enumerate(atoms, start=1):
        lines.append(_atom_line(serial=i, **a))
    lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _ca(chain: str, resseq: int, x: float, y: float = 0.0, z: float = 0.0) -> dict:
    """A single carbon pseudo-atom standing in for a whole residue."""
    return {
        "name": "CA",
        "resname": "ALA",
        "chain": chain,
        "resseq": resseq,
        "xyz": (x, y, z),
        "element": "C",
    }


def _two_chain_pdb(tmp_path, binder_x: list[float], target_x: list[float], name="c.pdb"):
    """One atom per residue, binder chain A on the x axis then target chain B."""
    atoms = [_ca("A", i + 1, x) for i, x in enumerate(binder_x)]
    atoms += [_ca("B", i + 1, x) for i, x in enumerate(target_x)]
    return _write_pdb(tmp_path / name, atoms)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_parses_chains_in_file_order_binder_first(tmp_path):
    st = parse_structure(_two_chain_pdb(tmp_path, [0.0, 1.0, 2.0], [20.0, 21.0]))
    assert st.chain_order == ("A", "B")
    assert st.n_residues("A") == 3
    assert st.n_residues("B") == 2
    assert st.n_atoms == 5


def test_hydrogens_are_dropped_at_parse_time(tmp_path):
    atoms = [_ca("A", 1, 0.0), _ca("B", 1, 3.0)]
    atoms.append(
        {
            "name": "HA",
            "resname": "ALA",
            "chain": "B",
            "resseq": 1,
            "xyz": (3.5, 0.0, 0.0),
            "element": "H",
        }
    )
    st = parse_structure(_write_pdb(tmp_path / "h.pdb", atoms))
    assert st.n_atoms == 2
    assert "H" not in st.elements


def test_element_is_inferred_when_the_columns_are_blank(tmp_path):
    """A PDB without element columns still parses, including the H names.

    The three cases that matter: a protein ``" CA "`` is carbon and must not be
    read as calcium, a hydrogen named ``"1HB "`` is hydrogen despite the
    leading digit, and selenomethionine's ``"SE  "`` is selenium because PDB
    right-justifies a two-letter symbol in columns 13-14.
    """
    body = [
        _atom_line(1, "CA", "ALA", "A", 1, (0.0, 0.0, 0.0), "", with_element=False),
        _atom_line(2, "1HB ", "ALA", "A", 1, (1.0, 0.0, 0.0), "", with_element=False),
        _atom_line(3, "SE  ", "MSE", "B", 1, (8.0, 0.0, 0.0), "", with_element=False),
    ]
    p = tmp_path / "noelem.pdb"
    p.write_text("\n".join(body) + "\nEND\n")
    st = parse_structure(p)
    assert st.elements == ("C", "SE")  # the 1HB hydrogen was dropped


def test_only_the_first_model_is_read(tmp_path):
    body = [
        "MODEL        1",
        _atom_line(1, "CA", "ALA", "A", 1, (0.0, 0.0, 0.0), "C"),
        _atom_line(2, "CA", "ALA", "B", 1, (3.0, 0.0, 0.0), "C"),
        "ENDMDL",
        "MODEL        2",
        _atom_line(3, "CA", "ALA", "A", 1, (99.0, 0.0, 0.0), "C"),
        "ENDMDL",
        "END",
    ]
    p = tmp_path / "models.pdb"
    p.write_text("\n".join(body) + "\n")
    st = parse_structure(p)
    assert st.n_atoms == 2


def test_alternate_locations_past_the_first_are_dropped(tmp_path):
    body = [
        _atom_line(1, "CA", "ALA", "A", 1, (0.0, 0.0, 0.0), "C", altloc="A"),
        _atom_line(2, "CA", "ALA", "A", 1, (0.5, 0.0, 0.0), "C", altloc="B"),
        _atom_line(3, "CA", "ALA", "B", 1, (3.0, 0.0, 0.0), "C"),
        "END",
    ]
    p = tmp_path / "altloc.pdb"
    p.write_text("\n".join(body) + "\n")
    assert parse_structure(p).n_atoms == 2


def test_hetatm_excluded_by_default_and_included_on_request(tmp_path):
    body = [
        _atom_line(1, "CA", "ALA", "A", 1, (0.0, 0.0, 0.0), "C"),
        _atom_line(2, "CA", "ALA", "B", 1, (3.0, 0.0, 0.0), "C"),
        _atom_line(3, "O", "HOH", "W", 1, (1.5, 0.0, 0.0), "O", record="HETATM"),
        "END",
    ]
    p = tmp_path / "het.pdb"
    p.write_text("\n".join(body) + "\n")
    assert parse_structure(p).n_atoms == 2
    assert parse_structure(p, include_hetatm=True).n_atoms == 3


def test_mmcif_gives_the_same_metrics_as_the_equivalent_pdb(tmp_path):
    binder_x = [0.0, 1.5, 3.0]
    target_x = [7.0, 8.5]
    pdb = _two_chain_pdb(tmp_path, binder_x, target_x)

    rows = []
    serial = 0
    for cid, xs in (("A", binder_x), ("B", target_x)):
        for i, x in enumerate(xs, start=1):
            serial += 1
            rows.append(
                f"ATOM {serial} C CA . ALA {cid} {i} ? "
                f"{x:.3f} 0.000 0.000 1.00 0.00 1"
            )
    cif = tmp_path / "c.cif"
    cif.write_text(
        "data_test\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.id\n"
        "_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_alt_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.auth_seq_id\n"
        "_atom_site.pdbx_PDB_ins_code\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "_atom_site.occupancy\n"
        "_atom_site.B_iso_or_equiv\n"
        "_atom_site.pdbx_PDB_model_num\n" + "\n".join(rows) + "\n#\n"
    )

    kw = dict(expected_target_length=2, n_points=240)
    from_pdb = geometry_metrics(pdb, 3, **kw)
    from_cif = geometry_metrics(cif, 3, **kw)
    assert from_cif["n_contacts_5A"] == from_pdb["n_contacts_5A"]
    assert from_cif["bsa"] == pytest.approx(from_pdb["bsa"])


def test_mmcif_second_model_is_ignored(tmp_path):
    cif = tmp_path / "m.cif"
    cif.write_text(
        "data_test\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.auth_seq_id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "_atom_site.pdbx_PDB_model_num\n"
        "ATOM C CA A 1 0.0 0.0 0.0 1\n"
        "ATOM C CA B 1 3.0 0.0 0.0 1\n"
        "ATOM C CA A 1 99.0 0.0 0.0 2\n"
        "#\n"
    )
    assert parse_structure(cif).n_atoms == 2


# --------------------------------------------------------------------------
# Contacts, with hand-calculable answers
# --------------------------------------------------------------------------


def test_contact_count_is_exact_for_hand_placed_atoms(tmp_path):
    """Binder at x = 0, 1; target at x = 4.5, 10.

    Interchain distances: 4.5, 10.0, 3.5, 9.0. Two are within 5 Angstroms, so
    the answer is exactly 2. The intrachain pair at 1.0 Angstrom must not be
    counted, and if it were the answer would be 3.
    """
    out = geometry_metrics(
        _two_chain_pdb(tmp_path, [0.0, 1.0], [4.5, 10.0]),
        expected_binder_length=2,
        expected_target_length=2,
        n_points=92,
    )
    assert out["n_contacts_5A"] == 2
    assert out["n_contact_residue_pairs_5A"] == 2
    assert out["n_interface_residues_binder"] == 2
    assert out["n_interface_residues_target"] == 1


def test_a_pair_at_exactly_the_cutoff_counts_as_a_contact(tmp_path):
    out = geometry_metrics(
        _two_chain_pdb(tmp_path, [0.0], [CONTACT_CUTOFF]),
        expected_binder_length=1,
        expected_target_length=1,
        n_points=92,
    )
    assert out["n_contacts_5A"] == 1


def test_a_pair_just_past_the_cutoff_does_not_count(tmp_path):
    out = geometry_metrics(
        _two_chain_pdb(tmp_path, [0.0], [CONTACT_CUTOFF + 0.001]),
        expected_binder_length=1,
        expected_target_length=1,
        n_points=92,
    )
    assert out["n_contacts_5A"] == 0


def test_intrachain_proximity_never_contributes(tmp_path):
    """Five binder atoms packed 1 Angstrom apart, target 40 Angstroms away."""
    out = geometry_metrics(
        _two_chain_pdb(tmp_path, [0.0, 1.0, 2.0, 3.0, 4.0], [40.0, 41.0]),
        expected_binder_length=5,
        expected_target_length=2,
        n_points=92,
    )
    assert out["n_contacts_5A"] == 0


def test_hydrogens_do_not_contribute_contacts(tmp_path):
    """Adding a target hydrogen 2 Angstroms from a binder atom changes nothing."""
    base = [_ca("A", 1, 0.0), _ca("B", 1, 8.0)]
    with_h = base + [
        {
            "name": "HA",
            "resname": "ALA",
            "chain": "B",
            "resseq": 1,
            "xyz": (2.0, 0.0, 0.0),
            "element": "H",
        }
    ]
    kw = dict(expected_binder_length=1, expected_target_length=1, n_points=92)
    assert geometry_metrics(_write_pdb(tmp_path / "a.pdb", base), **kw)[
        "n_contacts_5A"
    ] == 0
    assert geometry_metrics(_write_pdb(tmp_path / "b.pdb", with_h), **kw)[
        "n_contacts_5A"
    ] == 0


def test_all_pairs_contact_when_every_pair_is_close(tmp_path):
    """3 binder atoms and 4 target atoms all mutually within 5 Angstroms."""
    out = geometry_metrics(
        _two_chain_pdb(tmp_path, [0.0, 0.5, 1.0], [1.5, 2.0, 2.5, 3.0]),
        expected_binder_length=3,
        expected_target_length=4,
        n_points=92,
    )
    assert out["n_contacts_5A"] == 12


def test_interchain_contacts_rejects_an_empty_chain():
    with pytest.raises(ValueError, match="empty chain"):
        interchain_contacts(np.zeros((0, 3)), np.zeros((3, 3)))


def test_interchain_contacts_rejects_a_nonpositive_cutoff():
    with pytest.raises(ValueError, match="must be positive"):
        interchain_contacts(np.zeros((1, 3)), np.zeros((1, 3)), cutoff=0.0)


def test_contact_chunking_matches_an_unchunked_count():
    """More atoms than the internal row chunk must give the same answer."""
    rng = np.random.default_rng(20261110)
    a = rng.uniform(0.0, 12.0, size=(1300, 3))
    b = rng.uniform(0.0, 12.0, size=(40, 3))
    got = interchain_contacts(a, b)["n_atom_pairs"]
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    assert got == int((d2 <= CONTACT_CUTOFF**2).sum())


# --------------------------------------------------------------------------
# Shrake-Rupley, validated against closed-form areas
# --------------------------------------------------------------------------


def test_sphere_points_are_unit_vectors():
    pts = golden_spiral_points(960)
    assert pts.shape == (960, 3)
    assert np.allclose(np.linalg.norm(pts, axis=1), 1.0)
    # Quasi-uniform, so the centroid is near the origin.
    assert np.linalg.norm(pts.mean(axis=0)) < 1e-2


def test_sphere_points_rejects_zero():
    with pytest.raises(ValueError, match="at least one sphere point"):
        golden_spiral_points(0)


def test_isolated_atom_sasa_is_the_exact_full_sphere():
    """No neighbours means every test point is accessible, so this is exact."""
    r = VDW_RADII["C"]
    expected = 4.0 * math.pi * (r + PROBE_RADIUS) ** 2
    for n in (92, 240, 960):
        got = shrake_rupley_sasa(np.zeros((1, 3)), np.array([r]), n_points=n)
        assert got[0] == pytest.approx(expected, rel=1e-12)


def test_two_overlapping_spheres_match_the_analytic_cap_area():
    """Total accessible area of two equal overlapping spheres.

    Each loses a cap of area 2*pi*R*(R - d/2), so the total is
    4*pi*R^2 + 2*pi*R*d. Tolerance 0.5% relative; see the module docstring for
    why that number.
    """
    r = VDW_RADII["C"]
    R = r + PROBE_RADIUS
    for d in (2.0, 3.0, 4.0, 5.0):
        coords = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
        exact = 4.0 * math.pi * R * R + 2.0 * math.pi * R * d
        got = shrake_rupley_sasa(coords, np.array([r, r]), n_points=960).sum()
        assert got == pytest.approx(exact, rel=5e-3), f"d={d}"


def test_coarser_sphere_sampling_is_less_accurate():
    """The quadrature error must shrink with more points, or it is not one."""
    r = VDW_RADII["C"]
    R = r + PROBE_RADIUS
    d = 3.0
    coords = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
    exact = 4.0 * math.pi * R * R + 2.0 * math.pi * R * d
    err = {
        n: abs(shrake_rupley_sasa(coords, np.array([r, r]), n_points=n).sum() - exact)
        / exact
        for n in (92, 960, 20000)
    }
    assert err[92] < 0.025  # the 1973 paper's density, about 2% here
    assert err[960] < 0.005
    assert err[20000] < err[92]


def test_fully_engulfed_atom_has_zero_sasa():
    """A tiny atom at the centre of a huge one is completely buried."""
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    sasa = shrake_rupley_sasa(coords, np.array([1.2, 20.0]), n_points=240)
    assert sasa[0] == 0.0
    assert sasa[1] > 0.0


def test_distant_atoms_are_additive():
    r = VDW_RADII["C"]
    one = 4.0 * math.pi * (r + PROBE_RADIUS) ** 2
    coords = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
    got = shrake_rupley_sasa(coords, np.full(3, r), n_points=240).sum()
    assert got == pytest.approx(3.0 * one, rel=1e-12)


def test_zero_probe_radius_is_allowed_and_shrinks_the_area():
    r = VDW_RADII["C"]
    coords = np.zeros((1, 3))
    got = shrake_rupley_sasa(coords, np.array([r]), probe_radius=0.0, n_points=92)
    assert got[0] == pytest.approx(4.0 * math.pi * r * r, rel=1e-12)


def test_sasa_rejects_a_negative_probe():
    with pytest.raises(ValueError, match="must not be negative"):
        shrake_rupley_sasa(np.zeros((1, 3)), np.array([1.7]), probe_radius=-1.0)


def test_sasa_rejects_mismatched_radii():
    with pytest.raises(ValueError, match="same atoms"):
        shrake_rupley_sasa(np.zeros((3, 3)), np.array([1.7, 1.7]))


def test_sasa_rejects_wrongly_shaped_coordinates():
    with pytest.raises(ValueError, match=r"must be \(n_atoms, 3\)"):
        shrake_rupley_sasa(np.zeros((3, 2)), np.array([1.7, 1.7, 1.7]))


def test_sasa_neighbour_chunking_matches_a_single_chunk():
    """More atoms than the internal row chunk must give the same per-atom SASA."""
    rng = np.random.default_rng(7)
    coords = rng.uniform(0.0, 22.0, size=(700, 3))
    radii = np.full(700, VDW_RADII["C"])
    a = shrake_rupley_sasa(coords, radii, n_points=92)
    # Permuting the atoms exercises a different chunk split for the same system.
    perm = rng.permutation(700)
    b = shrake_rupley_sasa(coords[perm], radii, n_points=92)
    assert np.allclose(a[perm], b)


# --------------------------------------------------------------------------
# BSA bookkeeping
# --------------------------------------------------------------------------


def test_bsa_equals_the_sum_of_the_two_per_side_deltas(tmp_path):
    """The stated identity must hold exactly, not approximately.

    bsa is defined as SASA(A alone) + SASA(B alone) - SASA(complex), and the
    per-side deltas partition the same quantity, so they must agree.
    """
    out = geometry_metrics(
        _two_chain_pdb(tmp_path, [0.0, 2.0, 4.0], [6.0, 8.0, 10.0]),
        expected_binder_length=3,
        expected_target_length=3,
        n_points=960,
    )
    assert out["bsa"] == pytest.approx(
        out["delta_sasa_binder"] + out["delta_sasa_target"], rel=1e-9, abs=1e-6
    )


def test_bsa_per_side_is_exactly_half_and_is_not_the_headline(tmp_path):
    out = geometry_metrics(
        _two_chain_pdb(tmp_path, [0.0, 2.0], [4.0, 6.0]),
        expected_binder_length=2,
        expected_target_length=2,
        n_points=240,
    )
    assert out["bsa_per_side"] == pytest.approx(out["bsa"] / 2.0)
    assert out["bsa"] > out["bsa_per_side"] > 0.0


def test_non_touching_chains_bury_nothing(tmp_path):
    out = geometry_metrics(
        _two_chain_pdb(tmp_path, [0.0, 2.0], [60.0, 62.0]),
        expected_binder_length=2,
        expected_target_length=2,
        n_points=240,
    )
    assert out["n_contacts_5A"] == 0
    assert out["bsa"] == pytest.approx(0.0, abs=1e-9)


def test_bsa_of_two_touching_atoms_matches_the_analytic_cap(tmp_path):
    """Smallest interface there is: one atom per chain, overlapping probes.

    bsa = 2*full sphere - (4*pi*R^2 + 2*pi*R*d) = 4*pi*R^2 - 2*pi*R*d, which
    for C-C at d = 3.0 is 2*pi*R*(2R - d).
    """
    r = VDW_RADII["C"]
    R = r + PROBE_RADIUS
    d = 3.0
    out = geometry_metrics(
        _two_chain_pdb(tmp_path, [0.0], [d]),
        expected_binder_length=1,
        expected_target_length=1,
        n_points=20000,
    )
    expected = 2.0 * math.pi * R * (2.0 * R - d)
    assert out["bsa"] == pytest.approx(expected, rel=5e-3)


def test_closer_chains_bury_more_area(tmp_path):
    kw = dict(expected_binder_length=2, expected_target_length=2, n_points=960)
    near = geometry_metrics(
        _two_chain_pdb(tmp_path, [0.0, 2.0], [4.0, 6.0], name="near.pdb"), **kw
    )
    far = geometry_metrics(
        _two_chain_pdb(tmp_path, [0.0, 2.0], [9.0, 11.0], name="far.pdb"), **kw
    )
    assert near["bsa"] > far["bsa"]
    assert near["n_contacts_5A"] > far["n_contacts_5A"]


def test_reported_parameters_are_echoed_back(tmp_path):
    """The BSA convention is only reproducible if its parameters travel with it."""
    out = geometry_metrics(
        _two_chain_pdb(tmp_path, [0.0], [3.0]),
        expected_binder_length=1,
        expected_target_length=1,
        n_points=92,
        probe_radius=1.2,
        contact_cutoff=4.0,
    )
    assert out["probe_radius_angstrom"] == 1.2
    assert out["n_sphere_points"] == 92
    assert out["contact_cutoff_angstrom"] == 4.0
    assert out["binder_length"] == 1
    assert out["complex_length"] == 2
    assert out["n_heavy_atoms_binder"] == 1
    assert out["n_heavy_atoms_target"] == 1


# --------------------------------------------------------------------------
# The loud failures. These are the point of the module.
# --------------------------------------------------------------------------


def test_single_chain_is_rejected(tmp_path):
    p = _write_pdb(tmp_path / "one.pdb", [_ca("A", 1, 0.0), _ca("A", 2, 2.0)])
    with pytest.raises(ValueError, match="expected exactly 2"):
        geometry_metrics(p, expected_binder_length=2, expected_target_length=2)


def test_three_chains_are_rejected(tmp_path):
    atoms = [_ca("A", 1, 0.0), _ca("B", 1, 3.0), _ca("C", 1, 6.0)]
    p = _write_pdb(tmp_path / "three.pdb", atoms)
    with pytest.raises(ValueError, match="has 3 chain"):
        geometry_metrics(p, expected_binder_length=1, expected_target_length=1)


def test_binder_length_mismatch_is_rejected(tmp_path):
    p = _two_chain_pdb(tmp_path, [0.0, 2.0, 4.0], [6.0, 8.0])
    with pytest.raises(ValueError, match="Chain layout mismatch"):
        geometry_metrics(p, expected_binder_length=4, expected_target_length=2)


def test_target_length_mismatch_is_rejected(tmp_path):
    p = _two_chain_pdb(tmp_path, [0.0, 2.0, 4.0], [6.0, 8.0])
    with pytest.raises(ValueError, match="pinned target construct"):
        geometry_metrics(p, expected_binder_length=3, expected_target_length=7)


def test_chain_order_flip_is_named_as_such_not_just_a_mismatch(tmp_path):
    """Target-first must say so, because that is the mistake that recurs."""
    p = _two_chain_pdb(tmp_path, [0.0, 2.0, 4.0, 6.0, 8.0], [20.0, 22.0])
    with pytest.raises(ValueError, match="Chain order looks flipped"):
        geometry_metrics(p, expected_binder_length=2, expected_target_length=5)


def test_default_target_length_is_the_pinned_egfr_construct(tmp_path):
    """With no override the target chain must be 621 residues, or it raises."""
    p = _two_chain_pdb(tmp_path, [0.0, 2.0, 4.0], [20.0, 22.0])
    with pytest.raises(ValueError, match=str(EXPECTED_LENGTH)):
        geometry_metrics(p, expected_binder_length=3)


def test_the_pinned_construct_length_passes_without_an_override(tmp_path):
    """A 5 + 621 complex goes through the default path and the shared guard."""
    binder_x = [float(i) * 3.0 for i in range(5)]
    target_x = [40.0 + float(i) * 3.8 for i in range(EXPECTED_LENGTH)]
    p = _two_chain_pdb(tmp_path, binder_x, target_x, name="egfr.pdb")
    out = geometry_metrics(p, expected_binder_length=5, n_points=92)
    assert out["binder_length"] == 5
    assert out["complex_length"] == 5 + EXPECTED_LENGTH
    assert out["n_contacts_5A"] == 0


def test_file_with_no_atom_records_is_rejected(tmp_path):
    p = tmp_path / "empty.pdb"
    p.write_text("HEADER    NOTHING HERE\nREMARK   1 NO COORDINATES\nEND\n")
    with pytest.raises(ValueError, match="no usable atom records"):
        geometry_metrics(p, expected_binder_length=1, expected_target_length=1)


def test_completely_empty_file_is_rejected(tmp_path):
    p = tmp_path / "blank.pdb"
    p.write_text("")
    with pytest.raises(ValueError, match="no usable atom records"):
        parse_structure(p)


def test_hydrogen_only_file_is_rejected(tmp_path):
    atoms = [
        {
            "name": "H",
            "resname": "ALA",
            "chain": "A",
            "resseq": 1,
            "xyz": (0.0, 0.0, 0.0),
            "element": "H",
        }
    ]
    p = _write_pdb(tmp_path / "honly.pdb", atoms)
    with pytest.raises(ValueError, match="none of them are heavy atoms"):
        parse_structure(p)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_structure(tmp_path / "nope.pdb")


def test_unknown_element_raises_rather_than_defaulting_to_carbon(tmp_path):
    atoms = [_ca("A", 1, 0.0), _ca("B", 1, 3.0)]
    atoms.append(
        {
            "name": "XX",
            "resname": "UNK",
            "chain": "B",
            "resseq": 1,
            "xyz": (4.0, 0.0, 0.0),
            "element": "XX",
        }
    )
    p = _write_pdb(tmp_path / "unk.pdb", atoms)
    with pytest.raises(ValueError, match="Unknown element"):
        geometry_metrics(p, expected_binder_length=1, expected_target_length=1)


def test_truncated_atom_record_is_rejected(tmp_path):
    p = tmp_path / "short.pdb"
    p.write_text("ATOM      1  CA  ALA A   1       0.000\nEND\n")
    with pytest.raises(ValueError, match="Truncated ATOM record"):
        parse_structure(p)


def test_nonpositive_expected_lengths_are_rejected(tmp_path):
    p = _two_chain_pdb(tmp_path, [0.0], [3.0])
    with pytest.raises(ValueError, match="expected_binder_length must be positive"):
        geometry_metrics(p, expected_binder_length=0, expected_target_length=1)
    with pytest.raises(ValueError, match="expected_target_length must be positive"):
        geometry_metrics(p, expected_binder_length=1, expected_target_length=0)


def test_mmcif_loop_without_coordinates_is_rejected(tmp_path):
    cif = tmp_path / "bad.cif"
    cif.write_text(
        "data_test\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.auth_seq_id\n"
        "ATOM C CA A 1\n"
        "#\n"
    )
    with pytest.raises(ValueError, match="Cartn_x"):
        parse_structure(cif)


def test_mmcif_row_with_the_wrong_number_of_values_is_rejected(tmp_path):
    cif = tmp_path / "ragged.cif"
    cif.write_text(
        "data_test\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.auth_seq_id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "ATOM C CA A 1 0.0 0.0\n"
        "#\n"
    )
    with pytest.raises(ValueError, match="declares 8 columns"):
        parse_structure(cif)
