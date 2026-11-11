"""The EGFR target construct, pinned.

Why this module exists at all: a wrong target construct makes every prediction
meaningless and is the single cheapest way to waste the whole GPU budget. So the
construct is pinned by checksum here rather than fetched at fold time, and the
checksum is asserted before any prediction runs.

Provenance, established in Step 0 (see DECISIONS.md 2026-11-10):

- Adaptyv's own metrics code (adaptyvbio/competition_metrics,
  compute_af2_metrics.py) hardcodes ``target_length=621`` with the comment
  "the EGFR sequence has 621 residues". That number is load-bearing in their
  code: they locate the binder/target boundary in the PAE matrix by subtracting
  it from the total length, so every pae_interaction value they published
  depends on it being right.
- UniProt P00533 annotates the topological domain "Extracellular" as residues
  25 to 645, which is 621 residues exactly.

So the target is the FULL ectodomain, domains I through IV. SPEC.md originally
assumed competitors targeted a single domain; that was wrong.

Truncation was considered and rejected. A design binding domain I, II or IV
would be scored against a target lacking its epitope, producing a confident
looking number for an interaction that cannot happen. Worse, the Cetuximab
control would NOT catch it, because cetuximab binds domain III and would still
score well on a domain-III-only construct. Human EGF is the control that catches
it, because EGF needs domains I and III together.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# UniProt P00533, residues 25 to 645 (topological domain "Extracellular").
UNIPROT_ACCESSION = "P00533"
RESIDUE_START = 25
RESIDUE_END = 645
EXPECTED_LENGTH = 621
EXPECTED_SHA256 = "be4a09cf6ffda325749e1dbc3969f690c9e5ff242407d8891ddbf8df0726dd37"

# Approximate subdomain boundaries in UniProt numbering, for interpreting where
# a binder sits. Recorded as approximate on purpose: UniProt itself only
# annotates two "Approximate" leucine-rich repeats at 75-300 and 390-600.
SUBDOMAINS_APPROX = {
    "I_L1": (25, 190),
    "II_CR1": (190, 335),
    "III_L2": (335, 505),
    "IV_CR2": (505, 645),
}

# The sequence itself, embedded. NOT read from a file.
#
# It lived in data/*.fasta until 2026-11-10, when the first container run
# failed with FileNotFoundError on /data/egfr_ectodomain_P00533_25-645.fasta.
# Two things were wrong with that. The repo file is never uploaded to the
# container, and the path was computed as parents[2] / "data", which resolves
# to the repo root locally but to the root filesystem in the container, so it
# silently pointed at the mounted data Volume. A pinned construct belongs in
# code that ships with the package, not in a file that may or may not be
# mounted. 621 characters is nothing.
_SEQUENCE = (
    "LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQE"
    "VAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEIL"
    "HGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGE"
    "ENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTC"
    "PPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVR"
    "KCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHT"
    "PPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLN"
    "ITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQ"
    "VCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLP"
    "QAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNC"
    "TYGCTGPGLEGCPTNGPKIPS"
)

_FASTA = Path(__file__).resolve().parents[2] / "data" / (
    "egfr_ectodomain_P00533_25-645.fasta"
)


def _read_cached_fasta() -> str | None:
    """The on-disk copy, for cross-checking locally. Absent in containers."""
    if not _FASTA.exists():
        return None
    lines = _FASTA.read_text().splitlines()
    return "".join(ln.strip() for ln in lines if not ln.startswith(">"))


def target_sequence() -> str:
    """Return the pinned 621-residue EGFR ectodomain, checksum-verified.

    Raises loudly rather than returning a sequence that does not match the
    recorded checksum. A silent mismatch here is the failure mode this whole
    module exists to prevent.
    """
    seq = _SEQUENCE

    if len(seq) != EXPECTED_LENGTH:
        raise ValueError(
            f"EGFR construct length is {len(seq)}, expected {EXPECTED_LENGTH}. "
            f"Adaptyv's pae_interaction depends on this being exactly 621; a "
            f"different length means every interface metric would be computed "
            f"over the wrong block of the PAE matrix. Refusing to continue."
        )

    got = hashlib.sha256(seq.encode()).hexdigest()
    if got != EXPECTED_SHA256:
        raise ValueError(
            f"EGFR construct checksum mismatch.\n"
            f"  expected {EXPECTED_SHA256}\n"
            f"  got      {got}\n"
            f"The cached FASTA has changed. Do not fold against it until the "
            f"change is understood and DECISIONS.md is updated."
        )

    on_disk = _read_cached_fasta()
    if on_disk is not None and on_disk != seq:
        raise ValueError(
            "The embedded EGFR sequence and data/egfr_ectodomain_P00533_25-645.fasta "
            "disagree. One of them has been edited. Resolve before folding."
        )

    return seq


def split_index(total_length: int) -> int:
    """Index where the target begins, given a complex of ``total_length``.

    Chain order is BINDER FIRST, TARGET SECOND. This is not our convention, it
    is Adaptyv's, stated in a comment in their own code: "note that we are
    assuming that the binder always comes first in the sequence!". Matching it
    is what makes our numbers comparable to theirs.
    """
    binder_len = total_length - EXPECTED_LENGTH
    if binder_len <= 0:
        raise ValueError(
            f"Complex length {total_length} leaves a binder of {binder_len} "
            f"residues after removing the {EXPECTED_LENGTH}-residue target. "
            f"Either the chains are ordered target-first or the wrong construct "
            f"was used. Refusing to guess."
        )
    return binder_len


def assert_chain_layout(total_length: int, expected_binder_length: int) -> int:
    """Verify per design that the complex is laid out as we believe.

    SPEC.md calls this out as a named check: interface metrics are computed over
    interchain residue pairs, so they depend on knowing which chain is which. If
    the order flips between designs, or between our parser and the model's
    output convention, the metric gets computed on the wrong block of the matrix
    and returns numbers that look plausible and mean nothing. Fail loudly, never
    default.
    """
    binder_len = split_index(total_length)
    if binder_len != expected_binder_length:
        raise ValueError(
            f"Chain layout mismatch. Complex of {total_length} residues implies "
            f"a binder of {binder_len}, but the design's sequence is "
            f"{expected_binder_length} residues. Interface metrics would be "
            f"computed over the wrong PAE block. Refusing to continue."
        )
    return binder_len
