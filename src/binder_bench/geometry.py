"""Structural interface metrics from a predicted complex. Pure, no Modal imports.

Two of the per-design metrics SPEC.md Step 2 requires, ``n_contacts_5A`` and
``bsa``, plus the pieces they are built from. They are also two of the Step 3
baselines: if counting atoms within 5 Angstroms predicts binding as well as a
deep-learning confidence score does, that is the finding.

Chain order is BINDER FIRST, TARGET SECOND, which is Adaptyv's convention and
the one ``metrics.py`` and ``target.py`` already assume. Here it is asserted
against the parsed structure rather than trusted, for the reason SPEC.md gives
under "Named check: chain ordering": a metric computed on the wrong chain
assignment returns a plausible looking number that means nothing.

No new dependencies. The PDB and mmCIF readers below are deliberately minimal
and self-contained so that nothing has to be added to the Modal fold image.


CONTACT DEFINITION
------------------
``n_contacts_5A`` counts **ordered-pair-free atom pairs**: one heavy atom from
the binder chain and one heavy atom from the target chain whose centre-to-centre
distance is **<= 5.0 Angstroms**. Each such pair is counted once. Heavy atom
means element is not H and not D; hydrogens are dropped at parse time, so an
AF2 prediction (which has none) and a protonated structure give the same count.
Intrachain pairs are never counted. The comparison is on squared distances
against 25.0, so a pair at exactly 5.000 Angstroms is a contact.

``n_contact_residue_pairs_5A`` is the same relation collapsed to residues, and
``n_interface_residues_*`` counts residues with at least one such contact.


BSA CONVENTION, STATED IN FULL
------------------------------
Buried surface area needs a solvent-accessible surface area (SASA), and there
are two conventions in circulation for what to call "the" BSA. Both are
returned, named so they cannot be confused:

    bsa          = SASA(binder alone) + SASA(target alone) - SASA(complex)
    bsa_per_side = bsa / 2

``bsa`` is the TOTAL area buried, summing both sides of the interface, and it
is the key that feeds the ``bsa`` column SPEC.md asks for. ``bsa_per_side`` is
the halved convention, which is what is usually meant by "interface area" or
"buried surface area per partner" (and what PISA reports as interface area).
Nothing here is silently halved: pick the key you mean.

The two monomer SASAs are computed from the complex coordinates with the other
chain deleted. No repacking, no minimisation, no rigid-body separation: the
chains keep the conformation the predictor gave them. So this is the standard
"delta SASA on the predicted complex" quantity, not a binding free energy
decomposition, and it does not include any conformational-change term.

Parameters, all free to change but recorded here because an undocumented BSA
number is not reproducible:

- Algorithm: **Shrake-Rupley** numerical SASA (Shrake & Rupley, J Mol Biol 79
  (1973) 351-371). Each atom's expanded sphere of radius ``r_i + r_probe`` is
  sampled with test points; a point is buried if it falls inside any other
  atom's expanded sphere; the atom's SASA is its sphere area times the
  accessible fraction.
- Probe radius: **1.4 Angstroms**, the conventional water radius, as used in
  the original paper and by FreeSASA, MDTraj and biotite by default.
- Sphere sampling: **960 points per atom** by default (``N_SPHERE_POINTS``),
  placed on a **golden-spiral (Fibonacci) lattice**. The 1973 paper used an
  icosahedral tessellation; the golden spiral is the quasi-uniform construction
  MDTraj uses, is quasi-random rather than random, and gives the same answer to
  well under a percent at this density. 960 is MDTraj's default. Measured
  against the closed-form area of two overlapping spheres (see
  tests/test_geometry.py) the error at 960 points is below 0.2% for centre
  separations of 2 to 5 Angstroms, and about 2% at the 92 points the original
  paper used. The error is a quadrature error with no guaranteed sign, so it
  does not cancel between the complex and the monomers; it is simply small
  relative to a BSA of order 1000 square Angstroms.
- Atomic radii: **Bondi van der Waals radii** (Bondi, J Phys Chem 68 (1964)
  441-451), in ``VDW_RADII`` below. Bondi is the set the 1973 paper itself
  used. Hydrogens are excluded from the calculation entirely rather than being
  given Bondi's 1.20, so these are united-atom-style SASAs computed on heavy
  atoms only, which is what AF2 output supports (it has no hydrogens).
- An element not in the table raises. It is not defaulted to carbon.

Because the radii are heavy-atom-only and the probe is 1.4, absolute SASA
values here run a little lower than a protonated all-atom calculation would.
Only differences between SASAs of identical atom sets are used for ``bsa``, so
that offset cancels in the metric that matters.


PERFORMANCE
-----------
Sized for SPEC.md's 378 complexes of roughly 660 to 870 residues, about 5000 to
6500 heavy atoms. No Python loop runs over atom pairs.

- Contacts: the binder-by-target distance block is at most about 2000 x 4800,
  which would be a 77 MB float64 array. It is computed in row chunks of
  ``_PAIR_CHUNK`` instead, so the peak is about 20 MB and nothing depends on
  the whole block fitting.
- SASA: neighbour lists are built from the same chunked all-pairs distances, so
  the full 6500 x 6500 matrix (340 MB) is never materialised; peak is one
  ``_PAIR_CHUNK`` x n_atoms row block, about 27 MB. The accessibility test then
  loops once per atom in Python (a few thousand iterations) with the
  point-by-neighbour test vectorised inside, which is the one place a Python
  loop remains.

Measured on this machine, a 150 + 621 residue complex of 5400 heavy atoms at
protein packing density takes about 2.8 seconds end to end at the default 960
points, and 1.2 seconds at 240. Nearly all of it is SASA. Over 378 designs that
is under 20 minutes on one core, which is nothing next to the folds that
produced the structures, so the default stays at the accurate setting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .target import EXPECTED_LENGTH, assert_chain_layout

# --------------------------------------------------------------------------
# Parameters. See the module docstring for the source of every number.
# --------------------------------------------------------------------------

PROBE_RADIUS = 1.4
N_SPHERE_POINTS = 960
CONTACT_CUTOFF = 5.0

# Bondi, J Phys Chem 68 (1964) 441-451. H and D are listed for completeness but
# are filtered out before any calculation; see the module docstring.
VDW_RADII: dict[str, float] = {
    "H": 1.20,
    "D": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "SE": 1.90,
}

HYDROGEN_ELEMENTS = frozenset({"H", "D"})

# Row-chunk height for every all-pairs distance computation here.
_PAIR_CHUNK = 512

# Elements whose two-letter symbol must be recognised when a PDB file omits the
# element columns. Deliberately short: a protein structure with a 2-letter
# element other than SE is not something this pipeline produces.
_TWO_LETTER_ELEMENTS = frozenset({"SE", "CL", "BR", "FE", "ZN", "MG", "MN", "CA", "NA"})


# --------------------------------------------------------------------------
# Parsed structure
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Structure:
    """Heavy atoms of one model of one structure, as parallel arrays.

    ``chain_order`` is the order in which chains first appear in the file,
    which for ColabFold output is the order the sequences were given in, hence
    binder first. Nothing here reorders chains or guesses which is which.
    """

    coords: np.ndarray  # (n_atoms, 3) float64
    elements: tuple[str, ...]
    atom_names: tuple[str, ...]
    chain_ids: tuple[str, ...]
    residue_keys: tuple[tuple[str, int, str], ...]  # (chain, resseq, icode)
    chain_order: tuple[str, ...]
    source: str

    @property
    def n_atoms(self) -> int:
        return int(self.coords.shape[0])

    def atom_mask(self, chain_id: str) -> np.ndarray:
        arr = np.asarray(self.chain_ids)
        return arr == chain_id

    def residue_ids(self, chain_id: str) -> tuple[tuple[str, int, str], ...]:
        """Distinct residues of a chain, in order of first appearance."""
        seen: dict[tuple[str, int, str], None] = {}
        for key, cid in zip(self.residue_keys, self.chain_ids):
            if cid == chain_id:
                seen.setdefault(key, None)
        return tuple(seen)

    def n_residues(self, chain_id: str) -> int:
        return len(self.residue_ids(chain_id))

    def radii(self) -> np.ndarray:
        """Bondi radii per atom. Raises on an element not in ``VDW_RADII``."""
        unknown = sorted({e for e in self.elements if e not in VDW_RADII})
        if unknown:
            raise ValueError(
                f"Unknown element(s) {unknown} in {self.source}. There is no "
                f"van der Waals radius for them in VDW_RADII (Bondi 1964) and "
                f"defaulting to carbon would put an undocumented fudge inside "
                f"the BSA number. Add the radius with its source instead."
            )
        return np.array([VDW_RADII[e] for e in self.elements], dtype=float)


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------


def _element_from_atom_name(raw_name: str) -> str:
    """Infer an element when a PDB file leaves columns 77-78 blank.

    ColabFold writes the element columns, so this is a fallback for
    hand-written or third-party files. PDB puts the symbol of a two-letter
    element in columns 13-14, so a name whose first two characters form a known
    two-letter symbol is read that way; otherwise the first alphabetic
    character wins, which maps names like ``1HB`` onto H.
    """
    stripped = raw_name.strip().upper()
    if not stripped:
        return ""
    if raw_name[:2].strip().upper() in _TWO_LETTER_ELEMENTS and not raw_name[0].isspace():
        return raw_name[:2].strip().upper()
    for ch in stripped:
        if ch.isalpha():
            return ch
    return ""


def _parse_pdb(text: str, include_hetatm: bool) -> list[dict]:
    records: list[dict] = []
    for line in text.splitlines():
        tag = line[:6]
        if tag == "ENDMDL":
            break  # first model only
        if tag == "ATOM  ":
            pass
        elif tag == "HETATM":
            if not include_hetatm:
                continue
        else:
            continue
        if len(line) < 54:
            raise ValueError(
                f"Truncated ATOM record, {len(line)} characters, needs at least "
                f"54 to hold coordinates: {line!r}"
            )
        altloc = line[16].strip()
        if altloc not in ("", "A"):
            continue
        raw_name = line[12:16]
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        if not element:
            element = _element_from_atom_name(raw_name)
        records.append(
            {
                "name": raw_name.strip(),
                "element": element,
                "chain": line[21].strip() or " ",
                "resseq": int(line[22:26]),
                "icode": line[26].strip(),
                "xyz": (float(line[30:38]), float(line[38:46]), float(line[46:54])),
            }
        )
    return records


def _cif_tokens(line: str) -> list[str]:
    """Split one mmCIF data row, respecting single and double quotes."""
    out: list[str] = []
    i, n = 0, len(line)
    while i < n:
        if line[i].isspace():
            i += 1
            continue
        if line[i] in "'\"":
            quote = line[i]
            i += 1
            start = i
            while i < n and not (line[i] == quote and (i + 1 >= n or line[i + 1].isspace())):
                i += 1
            out.append(line[start:i])
            i += 1
        else:
            start = i
            while i < n and not line[i].isspace():
                i += 1
            out.append(line[start:i])
    return out


def _parse_cif(text: str, include_hetatm: bool) -> list[dict]:
    lines = text.splitlines()
    tags: list[str] = []
    rows: list[list[str]] = []

    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j = i + 1
            local: list[str] = []
            while j < len(lines) and lines[j].strip().startswith("_"):
                local.append(lines[j].strip())
                j += 1
            if local and local[0].startswith("_atom_site."):
                tags = [t.split(".", 1)[1] for t in local]
                while j < len(lines):
                    s = lines[j].strip()
                    if not s or s.startswith("#") or s.startswith("_") or s == "loop_":
                        break
                    rows.append(_cif_tokens(s))
                    j += 1
                break
            i = j
        else:
            i += 1

    if not tags:
        return []

    idx = {t: k for k, t in enumerate(tags)}

    def need(*names: str) -> int:
        for nm in names:
            if nm in idx:
                return idx[nm]
        raise ValueError(
            f"mmCIF _atom_site loop has none of {names}. Columns present: "
            f"{sorted(idx)}. Cannot read coordinates or chain assignment "
            f"without it."
        )

    c_group = idx.get("group_PDB")
    c_name = need("label_atom_id", "auth_atom_id")
    c_sym = idx.get("type_symbol")
    c_alt = idx.get("label_alt_id")
    c_chain = need("auth_asym_id", "label_asym_id")
    c_seq = need("auth_seq_id", "label_seq_id")
    c_ins = idx.get("pdbx_PDB_ins_code")
    c_x, c_y, c_z = need("Cartn_x"), need("Cartn_y"), need("Cartn_z")
    c_model = idx.get("pdbx_PDB_model_num")

    first_model: str | None = None
    records: list[dict] = []
    for row in rows:
        if len(row) != len(tags):
            raise ValueError(
                f"mmCIF _atom_site row has {len(row)} values but the loop "
                f"declares {len(tags)} columns: {row!r}"
            )
        if c_model is not None:
            if first_model is None:
                first_model = row[c_model]
            elif row[c_model] != first_model:
                continue
        if c_group is not None:
            group = row[c_group].upper()
            if group == "HETATM" and not include_hetatm:
                continue
            if group not in ("ATOM", "HETATM"):
                continue
        if c_alt is not None and row[c_alt] not in (".", "?", "", "A"):
            continue
        name = row[c_name].strip('"')
        element = row[c_sym].strip().upper() if c_sym is not None else ""
        if element in (".", "?"):
            element = ""
        if not element:
            element = _element_from_atom_name(name)
        ins = row[c_ins].strip() if c_ins is not None else ""
        if ins in (".", "?"):
            ins = ""
        seq_raw = row[c_seq]
        if seq_raw in (".", "?"):
            raise ValueError(
                f"mmCIF _atom_site row has no residue number: {row!r}. Residue "
                f"counts drive the chain-layout check, so this cannot be skipped."
            )
        records.append(
            {
                "name": name,
                "element": element,
                "chain": row[c_chain] or " ",
                "resseq": int(seq_raw),
                "icode": ins,
                "xyz": (float(row[c_x]), float(row[c_y]), float(row[c_z])),
            }
        )
    return records


def parse_structure(
    path: str | Path,
    *,
    include_hetatm: bool = False,
) -> Structure:
    """Read a PDB or mmCIF file and return its heavy atoms, first model only.

    Format is chosen by suffix (``.cif``/``.mmcif`` versus anything else) and
    falls back to the other reader if the chosen one finds nothing, so a
    mislabelled file still parses rather than reporting "no ATOM records".

    Hydrogens and deuteriums are dropped. Alternate locations other than the
    first (altloc blank or ``A``) are dropped. HETATM records are dropped
    unless ``include_hetatm``, because a ColabFold prediction has none and an
    unexpected ligand or water silently entering a SASA calculation is exactly
    the kind of quiet wrongness this project is about.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No structure at {p}")
    text = p.read_text()

    cif_first = p.suffix.lower() in (".cif", ".mmcif")
    readers = (_parse_cif, _parse_pdb) if cif_first else (_parse_pdb, _parse_cif)
    records: list[dict] = []
    for reader in readers:
        records = reader(text, include_hetatm)
        if records:
            break

    if not records:
        raise ValueError(
            f"{p} contains no usable atom records. Looked for PDB ATOM lines "
            f"and an mmCIF _atom_site loop and found neither (HETATM records "
            f"are excluded unless include_hetatm=True). Refusing to return "
            f"metrics for an empty structure."
        )

    heavy = [r for r in records if r["element"] not in HYDROGEN_ELEMENTS]
    if not heavy:
        raise ValueError(
            f"{p} has {len(records)} atoms but none of them are heavy atoms. "
            f"Every element parsed as hydrogen, which usually means the "
            f"element columns are wrong."
        )

    chain_order: list[str] = []
    for r in heavy:
        if r["chain"] not in chain_order:
            chain_order.append(r["chain"])

    return Structure(
        coords=np.array([r["xyz"] for r in heavy], dtype=float),
        elements=tuple(r["element"] for r in heavy),
        atom_names=tuple(r["name"] for r in heavy),
        chain_ids=tuple(r["chain"] for r in heavy),
        residue_keys=tuple((r["chain"], r["resseq"], r["icode"]) for r in heavy),
        chain_order=tuple(chain_order),
        source=str(p),
    )


# --------------------------------------------------------------------------
# Contacts
# --------------------------------------------------------------------------


def interchain_contacts(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    residue_index_a: np.ndarray | None = None,
    residue_index_b: np.ndarray | None = None,
    cutoff: float = CONTACT_CUTOFF,
) -> dict[str, int]:
    """Count heavy-atom contacts between two atom sets within ``cutoff``.

    Both inputs must already be heavy atoms of a single chain each. Returns the
    atom-pair count, and if residue indices are supplied, the residue-pair
    count and the number of residues on each side that make any contact.
    """
    if coords_a.size == 0 or coords_b.size == 0:
        raise ValueError(
            "Cannot count interchain contacts with an empty chain. One of the "
            "two chains has no heavy atoms."
        )
    if cutoff <= 0:
        raise ValueError(f"Contact cutoff must be positive, got {cutoff}")

    cut2 = float(cutoff) ** 2
    n_pairs = 0
    res_pairs: set[tuple[int, int]] = set()
    hit_a: set[int] = set()
    hit_b: set[int] = set()
    want_residues = residue_index_a is not None and residue_index_b is not None

    for start in range(0, coords_a.shape[0], _PAIR_CHUNK):
        stop = min(start + _PAIR_CHUNK, coords_a.shape[0])
        d2 = np.zeros((stop - start, coords_b.shape[0]), dtype=float)
        for dim in range(3):
            diff = coords_a[start:stop, dim, None] - coords_b[None, :, dim]
            d2 += diff * diff
        close = d2 <= cut2
        n_pairs += int(close.sum())
        if want_residues:
            ia, ib = np.nonzero(close)
            if ia.size:
                ra = np.asarray(residue_index_a)[start + ia]
                rb = np.asarray(residue_index_b)[ib]
                res_pairs.update(zip(ra.tolist(), rb.tolist()))
                hit_a.update(ra.tolist())
                hit_b.update(rb.tolist())

    out = {"n_atom_pairs": n_pairs}
    if want_residues:
        out["n_residue_pairs"] = len(res_pairs)
        out["n_residues_a"] = len(hit_a)
        out["n_residues_b"] = len(hit_b)
    return out


# --------------------------------------------------------------------------
# Shrake-Rupley SASA
# --------------------------------------------------------------------------


def golden_spiral_points(n_points: int) -> np.ndarray:
    """``n_points`` quasi-uniform points on the unit sphere, golden spiral."""
    if n_points < 1:
        raise ValueError(f"Need at least one sphere point, got {n_points}")
    k = np.arange(n_points, dtype=float) + 0.5
    cos_phi = 1.0 - 2.0 * k / n_points
    sin_phi = np.sqrt(np.clip(1.0 - cos_phi * cos_phi, 0.0, None))
    theta = math.pi * (1.0 + 5.0**0.5) * k
    return np.column_stack(
        [np.cos(theta) * sin_phi, np.sin(theta) * sin_phi, cos_phi]
    )


def _neighbour_lists(coords: np.ndarray, expanded: np.ndarray) -> list[np.ndarray]:
    """Per atom, indices of atoms whose expanded spheres could overlap it.

    Chunked over rows so the full n-by-n distance matrix is never allocated.
    """
    n = coords.shape[0]
    out: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * n
    for start in range(0, n, _PAIR_CHUNK):
        stop = min(start + _PAIR_CHUNK, n)
        d2 = np.zeros((stop - start, n), dtype=float)
        for dim in range(3):
            diff = coords[start:stop, dim, None] - coords[None, :, dim]
            d2 += diff * diff
        thresh = expanded[start:stop, None] + expanded[None, :]
        close = d2 < thresh * thresh
        close[np.arange(stop - start), np.arange(start, stop)] = False
        for row in range(stop - start):
            out[start + row] = np.flatnonzero(close[row])
    return out


def shrake_rupley_sasa(
    coords: np.ndarray,
    radii: np.ndarray,
    probe_radius: float = PROBE_RADIUS,
    n_points: int = N_SPHERE_POINTS,
) -> np.ndarray:
    """Per-atom solvent-accessible surface area, in square Angstroms.

    Shrake & Rupley (1973). Each atom carries a sphere of radius
    ``radius + probe_radius`` sampled at ``n_points`` golden-spiral points; a
    point is buried if it lies inside any other atom's expanded sphere. See the
    module docstring for the parameter defaults and their sources.
    """
    coords = np.asarray(coords, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Coordinates must be (n_atoms, 3), got {coords.shape}")
    if radii.shape != (coords.shape[0],):
        raise ValueError(
            f"Got {radii.shape[0] if radii.ndim else 0} radii for "
            f"{coords.shape[0]} atoms. These must describe the same atoms."
        )
    if probe_radius < 0:
        raise ValueError(f"Probe radius must not be negative, got {probe_radius}")
    if coords.shape[0] == 0:
        return np.empty(0, dtype=float)

    expanded = radii + probe_radius
    unit = golden_spiral_points(n_points)
    neighbours = _neighbour_lists(coords, expanded)

    sasa = np.empty(coords.shape[0], dtype=float)
    sq_expanded = expanded * expanded
    for i in range(coords.shape[0]):
        nb = neighbours[i]
        if nb.size == 0:
            accessible = n_points
        else:
            points = coords[i] + expanded[i] * unit  # (n_points, 3)
            nb_xyz = coords[nb]
            d2 = np.zeros((n_points, nb.size), dtype=float)
            for dim in range(3):
                diff = points[:, dim, None] - nb_xyz[None, :, dim]
                d2 += diff * diff
            buried = (d2 < sq_expanded[nb][None, :]).any(axis=1)
            accessible = n_points - int(buried.sum())
        sasa[i] = 4.0 * math.pi * sq_expanded[i] * accessible / n_points
    return sasa


# --------------------------------------------------------------------------
# The metric entry point
# --------------------------------------------------------------------------


def geometry_metrics(
    structure_path: str | Path,
    expected_binder_length: int,
    *,
    expected_target_length: int = EXPECTED_LENGTH,
    probe_radius: float = PROBE_RADIUS,
    n_points: int = N_SPHERE_POINTS,
    contact_cutoff: float = CONTACT_CUTOFF,
    include_hetatm: bool = False,
) -> dict[str, float]:
    """Compute ``n_contacts_5A`` and ``bsa`` for one predicted complex.

    ``expected_binder_length`` is the designed binder's sequence length and is
    checked against the first chain's residue count. ``expected_target_length``
    defaults to the pinned EGFR ectodomain's 621 residues and is checked
    against the second chain; override it only for a synthetic structure or a
    deliberately different target, never to make a mismatch go away.

    Raises rather than returning anything if the structure is not two chains
    laid out binder-first with those two lengths. See the module docstring for
    the contact definition and the exact BSA convention.
    """
    if expected_binder_length <= 0:
        raise ValueError(
            f"expected_binder_length must be positive, got "
            f"{expected_binder_length}"
        )
    if expected_target_length <= 0:
        raise ValueError(
            f"expected_target_length must be positive, got "
            f"{expected_target_length}"
        )

    st = parse_structure(structure_path, include_hetatm=include_hetatm)

    if len(st.chain_order) != 2:
        raise ValueError(
            f"{st.source} has {len(st.chain_order)} chain(s) "
            f"{list(st.chain_order)}, expected exactly 2 (binder then target). "
            f"Interface metrics are only defined between two chains; with any "
            f"other number there is no way to say which atoms are the "
            f"interface, and a guess would produce a plausible looking number "
            f"that means nothing."
        )

    binder_id, target_id = st.chain_order
    n_binder_res = st.n_residues(binder_id)
    n_target_res = st.n_residues(target_id)

    # Named first, because target-first is the recurring mistake and a generic
    # "mismatch" message buries it. Undetectable, and skipped, when the two
    # expected lengths happen to be equal.
    if (
        expected_binder_length != expected_target_length
        and n_binder_res == expected_target_length
        and n_target_res == expected_binder_length
    ):
        raise ValueError(
            f"Chain order looks flipped in {st.source}. First chain "
            f"{binder_id!r} has {n_binder_res} residues and second chain "
            f"{target_id!r} has {n_target_res}, which is the target-first "
            f"layout. This pipeline requires binder first, target second, to "
            f"match Adaptyv's convention. Refusing to guess."
        )
    if n_binder_res != expected_binder_length:
        raise ValueError(
            f"Chain layout mismatch in {st.source}. First chain {binder_id!r} "
            f"has {n_binder_res} residues but the design's binder sequence is "
            f"{expected_binder_length} residues. Contacts and BSA would be "
            f"computed with the wrong chain assignment and would look "
            f"plausible. Refusing to continue."
        )
    if n_target_res != expected_target_length:
        raise ValueError(
            f"Chain layout mismatch in {st.source}. Second chain "
            f"{target_id!r} has {n_target_res} residues but the pinned target "
            f"construct is {expected_target_length} residues. Either the wrong "
            f"construct was folded or residues are missing from the output. "
            f"Refusing to continue."
        )

    # Same guard metrics.py uses on the PAE matrix, so the two code paths
    # cannot drift apart on what "binder first" means. Only meaningful for the
    # real EGFR construct.
    if expected_target_length == EXPECTED_LENGTH:
        assert_chain_layout(n_binder_res + n_target_res, expected_binder_length)

    mask_b = st.atom_mask(binder_id)
    mask_t = st.atom_mask(target_id)
    coords_b = st.coords[mask_b]
    coords_t = st.coords[mask_t]

    res_of_atom = {key: k for k, key in enumerate(dict.fromkeys(st.residue_keys))}
    res_idx = np.array([res_of_atom[key] for key in st.residue_keys], dtype=np.int64)

    contacts = interchain_contacts(
        coords_b,
        coords_t,
        residue_index_a=res_idx[mask_b],
        residue_index_b=res_idx[mask_t],
        cutoff=contact_cutoff,
    )

    radii = st.radii()
    sasa_complex_atoms = shrake_rupley_sasa(
        st.coords, radii, probe_radius=probe_radius, n_points=n_points
    )
    sasa_binder_atoms = shrake_rupley_sasa(
        coords_b, radii[mask_b], probe_radius=probe_radius, n_points=n_points
    )
    sasa_target_atoms = shrake_rupley_sasa(
        coords_t, radii[mask_t], probe_radius=probe_radius, n_points=n_points
    )

    sasa_complex = float(sasa_complex_atoms.sum())
    sasa_binder_alone = float(sasa_binder_atoms.sum())
    sasa_target_alone = float(sasa_target_atoms.sum())

    # BSA = SASA(binder alone) + SASA(target alone) - SASA(complex). TOTAL,
    # both sides of the interface, NOT halved. bsa_per_side is the halved one.
    bsa = sasa_binder_alone + sasa_target_alone - sasa_complex

    delta_binder = sasa_binder_alone - float(sasa_complex_atoms[mask_b].sum())
    delta_target = sasa_target_alone - float(sasa_complex_atoms[mask_t].sum())

    return {
        "n_contacts_5A": int(contacts["n_atom_pairs"]),
        "bsa": bsa,
        "bsa_per_side": bsa / 2.0,
        "sasa_complex": sasa_complex,
        "sasa_binder_alone": sasa_binder_alone,
        "sasa_target_alone": sasa_target_alone,
        "delta_sasa_binder": delta_binder,
        "delta_sasa_target": delta_target,
        "n_contact_residue_pairs_5A": int(contacts["n_residue_pairs"]),
        "n_interface_residues_binder": int(contacts["n_residues_a"]),
        "n_interface_residues_target": int(contacts["n_residues_b"]),
        "binder_length": int(n_binder_res),
        "complex_length": int(n_binder_res + n_target_res),
        "n_heavy_atoms_binder": int(coords_b.shape[0]),
        "n_heavy_atoms_target": int(coords_t.shape[0]),
        "contact_cutoff_angstrom": float(contact_cutoff),
        "probe_radius_angstrom": float(probe_radius),
        "n_sphere_points": int(n_points),
    }
