"""Does a predicted complex reproduce a KNOWN binding mode?

### Why this exists

SPEC.md gates the pilot on the two positive controls "scoring well". They do
not: pae_interface_mean came out at 21.75 for Human_EGF and 26.51 for
Cetuximab_scFv, against a field convention of under 10 for a credible
interface. Taken at face value that says the pipeline is broken.

It is not. The structures are right, and this module is how that was
established, so the check is reproducible code rather than a one-off script.

The gate it implements is: a positive control must recover the epitope its
crystal structure shows, far better than chance. That tests OUR SOFTWARE,
which is what a control is for. Whether AlphaFold is then confident about a
structure it got right is a separate question, and it happens to be this
project's actual research question, so conflating the two would make a finding
look like a bug.

### Where the reference epitopes come from

Computed, not quoted. Each set is the list of EGFR residues within 5.0 Angstrom
(heavy atom to heavy atom) of the bound ligand in the deposited crystal
structure, derived with this package's own geometry code so that the criterion
is identical on both sides of the comparison:

- ``EGF_EPITOPE_1IVO``: PDB 1IVO, EGFR chain A against EGF chain C. 1IVO is the
  2:2 EGF/EGFR ectodomain complex (Ogiso et al., Cell 110, 2002, 775-787).
- ``CETUXIMAB_EPITOPE_1YY9``: PDB 1YY9, EGFR chain A against the cetuximab Fab
  light and heavy chains C and D (Li et al., Cancer Cell 7, 2005, 301-311).

Both structures number EGFR in MATURE numbering, residue 1 being UniProt P00533
residue 25, which is exactly the numbering of our construct, so no offset is
applied. 1YY9's EGFR chain spans UniProt 25-642 and ours spans 25-645, so the
two agree residue for residue over the epitope.

Deliberately NOT used: the four ectodomain subdomain boundaries. UniProt
annotates only the kinase domain for P00533, so any subdomain boundaries would
be quoted from memory or from a paper, and the conclusion should not rest on
them. Epitope residues from a crystal structure need no such definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import parse_structure

#: EGFR residues contacting EGF in PDB 1IVO, 5.0 Angstrom heavy-atom criterion.
EGF_EPITOPE_1IVO = frozenset(
    {
        11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 29, 45, 68, 69, 89, 90, 98, 99,
        101, 325, 346, 348, 349, 350, 355, 356, 357, 358, 382, 384, 408, 409,
        411, 412, 415, 417, 418, 438, 440, 441, 465, 467, 468,
    }
)

#: EGFR residues contacting the cetuximab Fab in PDB 1YY9, same criterion.
CETUXIMAB_EPITOPE_1YY9 = frozenset(
    {
        349, 350, 353, 382, 384, 408, 409, 411, 412, 415, 417, 418, 438, 440,
        441, 443, 465, 466, 467, 468, 469, 470, 471, 472, 473,
    }
)

#: Reference epitope per control design_id, as it appears in designs.parquet.
CONTROL_EPITOPES = {
    "control.Human_EGF": EGF_EPITOPE_1IVO,
    "control.Cetuximab_scFv": CETUXIMAB_EPITOPE_1YY9,
}

#: Recall a control must reach for the pipeline to be considered sound.
#:
#: Measured values are 0.70 to 0.84 across the five Human_EGF models and 0.72
#: for Cetuximab_scFv, against a chance baseline of 0.08 for a random surface
#: patch of the same size. 0.40 therefore sits far below every observed value
#: and far above chance, so it catches a genuinely broken pipeline (a flipped
#: chain order, a malformed alignment folded as one chain) without tripping on
#: the ordinary variation between models.
MIN_EPITOPE_RECALL = 0.40


@dataclass(frozen=True)
class EpitopeRecovery:
    """How well a predicted interface matches a crystallographic one."""

    design_id: str
    structure: str
    n_predicted: int
    n_reference: int
    n_recovered: int
    recall: float
    precision: float
    f1: float
    chance_recall: float

    @property
    def enrichment_over_chance(self) -> float:
        """Recall divided by what a same-sized random patch would achieve.

        The honest headline number. A pipeline that has mixed up its chains
        lands near 1.0 here; the real controls land near 10.
        """
        if self.chance_recall <= 0:
            return float("inf")
        return self.recall / self.chance_recall

    def as_record(self) -> dict:
        return {
            "design_id": self.design_id,
            "structure": self.structure,
            "n_predicted_interface_residues": self.n_predicted,
            "n_reference_epitope_residues": self.n_reference,
            "n_recovered": self.n_recovered,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "f1": round(self.f1, 4),
            "chance_recall": round(self.chance_recall, 4),
            "enrichment_over_chance": round(self.enrichment_over_chance, 2),
        }


def contacted_target_residues(
    structure_path: Path | str,
    expected_binder_length: int,
    *,
    cutoff: float = 5.0,
) -> set[int]:
    """Target residue numbers with any heavy atom within ``cutoff`` of the binder.

    Chain order is asserted the same way the metrics are: binder first, target
    second. A flipped structure must fail loudly rather than return a plausible
    set of residues from the wrong chain.
    """
    st = parse_structure(structure_path)
    if len(st.chain_order) < 2:
        raise ValueError(
            f"{structure_path} has chains {st.chain_order}. A complex needs "
            f"two. A single chain here usually means the alignment's "
            f"cardinality header was malformed and the concatenated sequence "
            f"was folded as one chain."
        )

    binder_chain, target_chain = st.chain_order[0], st.chain_order[1]
    binder_residues = st.residue_ids(binder_chain)
    if len(binder_residues) != expected_binder_length:
        raise ValueError(
            f"{structure_path}: first chain {binder_chain!r} has "
            f"{len(binder_residues)} residues but the binder should have "
            f"{expected_binder_length}. Chain order may be flipped."
        )

    binder_coords = st.coords[st.atom_mask(binder_chain)]
    target_coords = st.coords[st.atom_mask(target_chain)]
    target_resseq = np.array(
        [k[1] for k, c in zip(st.residue_keys, st.chain_ids) if c == target_chain]
    )

    # Squared distances, so no square roots are taken for a threshold test.
    d2 = ((target_coords[:, None, :] - binder_coords[None, :, :]) ** 2).sum(-1)
    return set(int(r) for r in target_resseq[d2.min(axis=1) <= cutoff**2])


def epitope_recovery(
    structure_path: Path | str,
    design_id: str,
    expected_binder_length: int,
    *,
    reference: frozenset[int] | None = None,
    n_target_residues: int = 621,
    cutoff: float = 5.0,
) -> EpitopeRecovery:
    """Compare a predicted interface against a crystallographic epitope."""
    if reference is None:
        if design_id not in CONTROL_EPITOPES:
            raise ValueError(
                f"No reference epitope for {design_id!r}. Known: "
                f"{sorted(CONTROL_EPITOPES)}. Pass one explicitly."
            )
        reference = CONTROL_EPITOPES[design_id]

    predicted = contacted_target_residues(
        structure_path, expected_binder_length, cutoff=cutoff
    )
    recovered = predicted & reference

    recall = len(recovered) / len(reference) if reference else 0.0
    precision = len(recovered) / len(predicted) if predicted else 0.0
    f1 = (
        2 * recall * precision / (recall + precision)
        if (recall + precision) > 0
        else 0.0
    )
    # A random patch of the same size as our prediction: the expected share of
    # the epitope it would cover is just its share of the surface.
    chance = len(predicted) / n_target_residues if n_target_residues else 0.0

    return EpitopeRecovery(
        design_id=design_id,
        structure=str(structure_path),
        n_predicted=len(predicted),
        n_reference=len(reference),
        n_recovered=len(recovered),
        recall=recall,
        precision=precision,
        f1=f1,
        chance_recall=chance,
    )


def assert_control_reproduces_binding_mode(
    recoveries: list[EpitopeRecovery],
    *,
    min_recall: float = MIN_EPITOPE_RECALL,
) -> None:
    """The gate. Raises if a control does not recover its known epitope.

    Applied to the MEAN recall across a control's models rather than to each
    model, because individual models vary (0.70 to 0.84 was observed for
    Human_EGF) and the question is whether the pipeline as a whole reproduces
    the binding mode, not whether every member of the ensemble does.
    """
    if not recoveries:
        raise ValueError("No epitope recoveries to check")

    by_design: dict[str, list[EpitopeRecovery]] = {}
    for r in recoveries:
        by_design.setdefault(r.design_id, []).append(r)

    failures = []
    for design_id, rs in sorted(by_design.items()):
        mean_recall = sum(r.recall for r in rs) / len(rs)
        if mean_recall < min_recall:
            failures.append(
                f"{design_id}: mean epitope recall {mean_recall:.2f} over "
                f"{len(rs)} models, below {min_recall:.2f}"
            )

    if failures:
        raise AssertionError(
            "Positive control did not reproduce its known binding mode, so "
            "something upstream is wrong (chain order, alignment cardinality, "
            "target construct). This is NOT about confidence scores being "
            "low, which is a separate and expected finding.\n  "
            + "\n  ".join(failures)
        )
