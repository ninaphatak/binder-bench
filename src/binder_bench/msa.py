"""Build the ColabFold paired A3M for one binder-target complex.

Pure Python. No Modal imports, no network, no file reads at import time, so
this module is importable and testable offline.

WHY THIS MODULE IS WRITTEN SO DEFENSIVELY
-----------------------------------------
A silently wrong A3M does not crash. It folds, it produces five models, it
produces an ipTM and a PAE matrix, and every number is meaningless. That is the
exact failure mode this project exists to study, so it must not be the failure
mode of our own pipeline. Every function here refuses rather than defaults.

THE FORMAT, READ OUT OF ColabFold 1.6.2 ITSELF
----------------------------------------------
Everything below was established from the source dumped out of the pinned
container image (ghcr.io/sokrypton/colabfold 1.6.2-cuda12), not from
documentation and not from the GitHub main branch, which has drifted.

``colabfold_batch`` accepts an A3M file as input (positional help: "4) A3M
file"), and ``--msa-mode``'s help says "Using an A3M file as input overwrites
this option", so a precomputed A3M is the supported path for a cached MSA.

1. ``input.get_queries`` decides whether the job is a complex *from the A3M's
   first line*: it must start with ``#`` and split into exactly two
   tab-separated fields, the first a comma-separated list of chain lengths and
   the second a comma-separated list of copy counts. More than one length means
   ``is_complex = True``, which is what makes ``batch.set_model_type`` select a
   multimer model. So the header for our two-chain job is

       #<binder_len>,621<TAB>1,1

   Our cached target-only MSA, from ``colabfold_batch --msa-only``, opens with
   ``#621<TAB>1``: one chain, one copy. Same grammar.

2. ``input.get_queries`` also sets ``query_sequence = seqs[0]``, the first
   sequence in the file via ``parse_fasta`` (which skips ``#`` lines). And
   ``batch.unserialize_msa`` slices ``a3m_lines[2]``, the first sequence row,
   by raw character offsets taken from the header lengths:

       for qlen in query_seq_len:
           query_seqs_unique.append(qcat[prev:prev + qlen])

   There is no gap or insertion handling in that slice. So the first sequence
   row MUST be the plain concatenated query, binder immediately followed by
   target, with no gaps, no lowercase and no separator. A file whose first row
   is padded with gaps (which is what ColabFold's own ``pad_sequences`` emits
   for an unpaired-only complex) would hand AF2 a 621-residue "chain" made of
   dashes. It would still fold.

3. Every later row is assigned to chains by ``unserialize_msa._split_by_aln``,
   which walks the row and counts a column as *aligned* when it is NOT a
   lowercase letter (``if not ('a' <= c <= 'z')``). Lowercase letters are A3M
   insertions relative to the query and do not count; ``-`` does count. A
   segment closes as soon as its aligned-column count reaches that chain's
   declared length. Consequences we depend on:

   - a row for the target chain is written as ``'-' * binder_len + row``, and
     left-padding with gaps is safe even when the row opens with lowercase
     insertions, because the binder segment closes on its last dash;
   - a row whose aligned length is not exactly ``binder_len + 621`` is
     silently truncated or silently dropped, never reported. Hence the
     validation here.

4. ``_split_by_aln`` also returns ``has_aa`` per segment, true when the segment
   holds at least one uppercase letter. A row with amino acids in two or more
   segments is treated as a PAIRED row; otherwise it goes to the single chain
   that has residues, as that chain's unpaired MSA. The binder is de novo and
   single-sequence, so pairing across chains is meaningless for it. We
   therefore emit exactly one row with residues in both segments, the query
   pair itself, which is the standard no-pairing case: each chain's paired MSA
   is its own query and nothing else.

5. Each chain's unpaired MSA must be non-empty. ``generate_input_feature``
   passes ``unpaired_msa[i]`` straight into ``build_monomer_feature`` ->
   ``parsers.parse_a3m``, and AlphaFold's ``make_msa_features`` raises on an
   empty MSA. The query-pair row from point 4 lands in the PAIRED block for
   both chains, not the unpaired one, so the binder needs its own gap-padded
   row as well. That row is also exactly what ColabFold would emit for a
   single-sequence chain.

Put together, the text produced by :func:`build_complex_a3m` is byte-identical
to what ColabFold's own ``input.msa_to_str`` would write for

    query_seqs_unique      = [binder, target]
    query_seqs_cardinality = [1, 1]
    paired_msa             = [">101\\n<binder>", ">102\\n<target>"]
    unpaired_msa           = [">101\\n<binder>", <the cached target a3m rows>]

which is the point: we are not inventing a format, we are assembling the one
ColabFold serialises, with the binder's half deliberately kept single-sequence.

CHAIN ORDER IS BINDER FIRST, TARGET SECOND
------------------------------------------
Not our choice. It is Adaptyv's convention, stated in their own metrics code
("note that we are assuming that the binder always comes first in the
sequence!"), and every interface metric we compare against theirs depends on
it. See ``target.assert_chain_layout`` and ``metrics.interface_metrics``.

ONE INHERITED LOSS, STATED RATHER THAN HIDDEN
---------------------------------------------
Because the target is the LAST chain, any lowercase insertion columns trailing
the target's final aligned column are dropped by ``_split_by_aln`` (the segment
closes on the last aligned column and the slice ends there). That is
ColabFold's behaviour for any complex A3M, not something this module causes or
can avoid. It touches only C-terminal insertions in some hits, and it is
identical for all 378 designs because the target MSA is identical, so it cannot
bias a between-design comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Query labels. ColabFold numbers query sequences from 101 upward
#: (``colabfold.run_mmseqs2``: ``N = 101``; ``Ms = [N + index]``), so in a real
#: two-chain job chain 0's query header is ``>101`` and chain 1's is ``>102``.
#: Our cached target MSA was computed with the target alone, where it was chain
#: 0, so its query row is headed ``>101``; rewriting that one header to ``>102``
#: is what makes our file identical to the two-chain serialisation.
QUERY_LABEL_BINDER = "101"
QUERY_LABEL_TARGET = "102"

#: A target MSA of depth 1 would silently turn this experiment into
#: single-sequence prediction for BOTH chains while still producing confident
#: looking numbers. Staging checks depth explicitly for that reason; so does
#: this module, at the point of use.
MIN_TARGET_DEPTH = 2

#: Characters allowed in an alignment row: uppercase residues, lowercase
#: insertions, and the gap. Anything else (``.``, ``*``, digits) is counted as
#: an ALIGNED column by ColabFold's splitter and would shift the chain
#: boundary, so it is refused rather than passed through.
_ROW_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-")
_QUERY_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ---------------------------------------------------------------------------
# ColabFold's own primitives, re-implemented exactly
# ---------------------------------------------------------------------------


def aligned_length(seq: str) -> int:
    """Number of aligned columns in an A3M row.

    ColabFold's rule, from ``unserialize_msa._split_by_aln``: a column counts
    unless it is a lowercase letter. Lowercase letters are insertions relative
    to the query and carry no column of their own; the gap ``-`` does count.
    """
    return sum(1 for c in seq if not ("a" <= c <= "z"))


def split_chain_segments(
    seq: str, lengths: tuple[int, ...] | list[int]
) -> tuple[list[str], list[bool]]:
    """Split one A3M row into per-chain segments, as ColabFold does.

    A faithful re-implementation of ``unserialize_msa._split_by_aln``, used to
    check our own output against the parser that will consume it.

    Returns the segments and whether each segment holds at least one residue. A
    segment left empty means the row ran out of aligned columns before that
    chain was filled, which is how ColabFold silently drops a malformed row.
    """
    n = len(lengths)
    segments = [""] * n
    has_aa = [False] * n
    seg_idx = 0
    aln_count = 0
    start = 0
    curr_has_aa = False

    for i, c in enumerate(seq):
        if seg_idx >= n:
            break
        if not ("a" <= c <= "z"):
            if "A" <= c <= "Z":
                curr_has_aa = True
            aln_count += 1
            if aln_count == lengths[seg_idx]:
                segments[seg_idx] = seq[start : i + 1]
                has_aa[seg_idx] = curr_has_aa
                seg_idx += 1
                aln_count = 0
                start = i + 1
                curr_has_aa = False

    return segments, has_aa


def colabfold_jobname(design_id: str) -> str:
    """The job name ColabFold will derive from a file named after ``design_id``.

    ``batch.run`` sets ``jobname = safe_filename(input_path.stem)`` and names
    every output file after it, so this is the key our results are actually
    written under. 120 of the 402 competition design_ids contain spaces, which
    ``input.safe_filename`` rewrites to underscores; a caller that assumed the
    filename equalled the design_id would look for outputs that do not exist.
    Verbatim copy of ``input.safe_filename``.
    """
    if not design_id:
        raise ValueError("design_id is empty; ColabFold would produce an unnamed job.")
    return "".join(c if c.isalnum() or c in ["_", ".", "-"] else "_" for c in design_id)


def a3m_filename(design_id: str) -> str:
    """File name to write for ``design_id`` so outputs come back under its job name."""
    return colabfold_jobname(design_id) + ".a3m"


def assert_unique_jobnames(design_ids: list[str]) -> dict[str, str]:
    """Refuse a batch whose design_ids collapse onto the same ColabFold job name.

    Two designs sharing a job name would overwrite each other's outputs in the
    results directory, and the survivor's numbers would be attributed to both.
    """
    mapping: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}
    for design_id in design_ids:
        job = colabfold_jobname(design_id)
        if job in mapping and mapping[job] != design_id:
            collisions.setdefault(job, [mapping[job]]).append(design_id)
        mapping[job] = design_id
    if collisions:
        detail = "; ".join(f"{job} <- {ids}" for job, ids in collisions.items())
        raise ValueError(
            f"design_ids collide after ColabFold's safe_filename: {detail}. "
            f"Their outputs would overwrite each other in the results volume "
            f"and one design's numbers would be reported for another. Rename "
            f"or key the files differently before folding."
        )
    return {design_id: colabfold_jobname(design_id) for design_id in design_ids}


# ---------------------------------------------------------------------------
# The cardinality header
# ---------------------------------------------------------------------------


def format_cardinality_header(
    lengths: tuple[int, ...] | list[int], cardinality: tuple[int, ...] | list[int]
) -> str:
    """Build ColabFold's chain-cardinality header line: lengths TAB copy counts.

    Mirrors the first two lines of ``input.msa_to_str``.
    """
    if len(lengths) != len(cardinality):
        raise ValueError(
            f"{len(lengths)} chain lengths against {len(cardinality)} copy counts. "
            f"ColabFold reads these as parallel lists and would mis-slice the query."
        )
    return ",".join(str(x) for x in lengths) + "\t" + ",".join(str(x) for x in cardinality)


def parse_cardinality_header(line: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Parse a ``#lengths<TAB>counts`` header, or raise.

    ``get_queries`` and ``unserialize_msa`` both require the leading ``#`` and
    exactly two tab-separated fields. When either is missing, ColabFold does
    NOT fail: ``unserialize_msa`` falls through to its single-chain branch and
    treats the whole file as one monomer MSA. For a complex that means folding
    a single chain of binder+target joined end to end, with no interface at
    all, and the run still produces scores. So we refuse here instead.
    """
    if not line.startswith("#"):
        raise ValueError(
            f"A3M header line does not start with '#': {line[:60]!r}. ColabFold "
            f"would take the file for a single-chain MSA and fold binder and "
            f"target as ONE chain, producing scores with no interface in them."
        )
    fields = line[1:].split("\t")
    if len(fields) != 2:
        raise ValueError(
            f"A3M header has {len(fields)} tab-separated fields, expected 2 "
            f"(lengths then copy counts): {line[:80]!r}. ColabFold requires "
            f"exactly 2 or it silently treats the file as a single-chain MSA."
        )
    try:
        lengths = tuple(int(x) for x in fields[0].split(","))
        cardinality = tuple(int(x) for x in fields[1].split(","))
    except ValueError as exc:
        raise ValueError(
            f"A3M header fields are not integers: {line[:80]!r} ({exc})."
        ) from None
    if len(lengths) != len(cardinality):
        raise ValueError(
            f"A3M header declares {len(lengths)} chain lengths but "
            f"{len(cardinality)} copy counts: {line[:80]!r}."
        )
    if any(n <= 0 for n in lengths) or any(n <= 0 for n in cardinality):
        raise ValueError(
            f"A3M header contains a non-positive value: {line[:80]!r}. "
            f"ColabFold clamps a negative length to zero and would emit an "
            f"empty chain."
        )
    return lengths, cardinality


# ---------------------------------------------------------------------------
# Reading the cached target MSA
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetMSA:
    """The cached target A3M, parsed and validated."""

    declared_length: int
    rows: tuple[tuple[str, str], ...]
    """``(header_without_'>', sequence)`` in file order. Row 0 is the query."""

    @property
    def query_header(self) -> str:
        return self.rows[0][0]

    @property
    def query_sequence(self) -> str:
        return self.rows[0][1]

    @property
    def depth(self) -> int:
        return len(self.rows)

    @property
    def hits(self) -> tuple[tuple[str, str], ...]:
        return self.rows[1:]


def _normalize_a3m(text: str) -> tuple[str | None, tuple[tuple[str, str], ...]]:
    """Split A3M text into its header line and ``(header, sequence)`` rows.

    Follows ``batch.normalize_a3m``: the leading ``#`` line is kept, wrapped
    sequence lines are joined, whitespace inside sequences is stripped, and
    ``\\x00`` separators (which the MSA server emits between chains) are
    removed as ``unserialize_msa`` does.
    """
    lines = text.replace("\x00", "").splitlines()
    header_line: str | None = None
    rows: list[tuple[str, str]] = []
    current: str | None = None
    chunks: list[str] = []

    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if idx == 0:
                header_line = line
                continue
            # normalize_a3m keeps ONLY lines[0]; a later '#' line is treated as
            # sequence text and concatenated onto the preceding row.
            raise ValueError(
                f"Comment line at line {idx + 1} of the A3M: {line[:60]!r}. "
                f"ColabFold's normalize_a3m keeps only the FIRST line as a "
                f"header and would splice this one into the sequence above it, "
                f"corrupting that alignment row. Refusing to pass it on."
            )
        if line.startswith(">"):
            if current is not None:
                rows.append((current, "".join(chunks)))
            current = line[1:]
            chunks = []
        else:
            if current is None:
                raise ValueError(
                    f"A3M has a sequence at line {idx + 1} before any '>' "
                    f"header: {line[:60]!r}. The file is not valid A3M."
                )
            chunks.append("".join(line.split()))

    if current is not None:
        rows.append((current, "".join(chunks)))
    return header_line, tuple(rows)


def _validate_query_sequence(seq: str, what: str) -> None:
    if not seq:
        raise ValueError(f"The {what} sequence is empty. Refusing to build an A3M.")
    bad = sorted(set(seq) - _QUERY_CHARS)
    if bad:
        raise ValueError(
            f"The {what} sequence contains {bad!r}, which is not an uppercase "
            f"residue letter. ColabFold slices the first A3M row by raw "
            f"character offsets with no gap or insertion handling, so a gap, a "
            f"lowercase insertion or a ':' separator in a query sequence shifts "
            f"the chain boundary and every interface metric is then computed "
            f"over the wrong block."
        )


def parse_target_a3m(
    text: str,
    target_sequence: str,
    *,
    min_depth: int = MIN_TARGET_DEPTH,
) -> TargetMSA:
    """Parse and fully validate the cached target MSA.

    ``target_sequence`` is the construct we intend to fold, normally
    ``binder_bench.target.target_sequence()``. Every check here exists because
    its failure mode is a plausible number rather than a crash.
    """
    _validate_query_sequence(target_sequence, "target")

    if not text or not text.strip():
        raise ValueError(
            "The cached target A3M is empty. Nothing was staged, or the wrong "
            "path was read. Folding with no target MSA would quietly become a "
            "single-sequence prediction for both chains, which is a different "
            "experiment that still produces confident looking numbers."
        )

    header_line, rows = _normalize_a3m(text)
    if header_line is None:
        raise ValueError(
            "The cached target A3M has no '#' cardinality header on its first "
            "line. colabfold_batch --msa-only writes one; a file without it is "
            "not the artifact we staged. Refusing to guess its chain layout."
        )
    lengths, cardinality = parse_cardinality_header(header_line)

    if len(lengths) != 1 or cardinality != (1,):
        raise ValueError(
            f"The cached target A3M declares {len(lengths)} chains with copy "
            f"counts {cardinality}, expected exactly one chain with one copy "
            f"(header '#{len(target_sequence)}\\t1'). This file is not a "
            f"target-only MSA, so pasting the binder in front of it would "
            f"produce a complex with the wrong number of chains."
        )

    declared = lengths[0]
    if declared != len(target_sequence):
        raise ValueError(
            f"The cached target A3M declares a chain length of {declared} but "
            f"the target construct is {len(target_sequence)} residues. One of "
            f"the two is not the EGFR ectodomain we pinned. Folding anyway "
            f"would align the MSA against the wrong construct and every ipTM "
            f"and PAE value would be plausible and meaningless."
        )

    if not rows:
        raise ValueError(
            "The cached target A3M has a header but no sequences. The MSA stage "
            "produced nothing usable."
        )

    query_header, query_seq = rows[0]
    if query_seq != target_sequence:
        n_shared = sum(1 for a, b in zip(query_seq, target_sequence) if a == b)
        raise ValueError(
            f"The first sequence in the cached target A3M is not the target "
            f"construct.\n"
            f"  A3M query ({len(query_seq)} chars, header {query_header!r}): "
            f"{query_seq[:40]}...{query_seq[-20:] if len(query_seq) > 60 else ''}\n"
            f"  pinned target ({len(target_sequence)} chars): "
            f"{target_sequence[:40]}...\n"
            f"  identical positions: {n_shared}\n"
            f"ColabFold takes the query row as the sequence it folds, so the "
            f"MSA would be aligned to one construct and the structure "
            f"predicted for another. Refusing to continue."
        )

    if min_depth > 0 and len(rows) < min_depth:
        raise ValueError(
            f"The cached target A3M has depth {len(rows)}, below the required "
            f"minimum of {min_depth}. A depth-1 alignment is the query alone, "
            f"which silently converts this experiment into single-sequence "
            f"prediction for BOTH chains while still returning confident "
            f"looking scores. Re-stage the target MSA."
        )

    for i, (hdr, seq) in enumerate(rows):
        if not seq:
            raise ValueError(
                f"Row {i} of the cached target A3M (header {hdr!r}) has no "
                f"sequence."
            )
        bad = sorted(set(seq) - _ROW_CHARS)
        if bad:
            raise ValueError(
                f"Row {i} of the cached target A3M (header {hdr!r}) contains "
                f"{bad!r}. Only residues, lowercase insertions and '-' are "
                f"allowed: ColabFold counts any non-lowercase character as an "
                f"ALIGNED column, so an unexpected character shifts the "
                f"binder/target boundary for that row."
            )
        n_aligned = aligned_length(seq)
        if n_aligned != declared:
            raise ValueError(
                f"Row {i} of the cached target A3M (header {hdr!r}) has "
                f"{n_aligned} aligned columns, expected {declared}. Lowercase "
                f"insertions do not count toward that total. ColabFold splits "
                f"chains by counting aligned columns and would silently "
                f"truncate or drop this row rather than report it."
            )

    return TargetMSA(declared_length=declared, rows=rows)


# ---------------------------------------------------------------------------
# Building the complex A3M
# ---------------------------------------------------------------------------


def build_complex_a3m(
    binder_sequence: str,
    target_sequence: str,
    target_a3m_text: str,
    *,
    min_target_depth: int = MIN_TARGET_DEPTH,
) -> str:
    """Return the ColabFold A3M text for one binder-target complex.

    The binder comes FIRST and is single-sequence; the target comes second and
    carries the cached deep MSA. Write the returned text to
    ``<design_id>.a3m`` (see :func:`a3m_filename`) and hand that path, or the
    directory holding a chunk of them, to ``colabfold_batch``.

    Raises rather than returning anything questionable. See
    :func:`parse_target_a3m` for the checks on the cached MSA.
    """
    _validate_query_sequence(binder_sequence, "binder")
    msa = parse_target_a3m(target_a3m_text, target_sequence, min_depth=min_target_depth)

    binder_len = len(binder_sequence)
    target_len = msa.declared_length
    binder_gaps = "-" * binder_len
    target_gaps = "-" * target_len

    out: list[str] = [
        # 1. The chain-cardinality header. Two lengths is what makes
        #    get_queries set is_complex and select a multimer model.
        "#" + format_cardinality_header((binder_len, target_len), (1, 1)),
        # 2. The paired block: exactly one row, the query pair. Its sequence
        #    row is the plain concatenated query, which unserialize_msa slices
        #    by the header lengths. Headers are joined with a TAB, as
        #    input.pair_sequences does (it rewrites chain 1's '>' to a tab).
        f">{QUERY_LABEL_BINDER}\t{QUERY_LABEL_TARGET}",
        binder_sequence + target_sequence,
        # 3. The unpaired block, chain 0: the binder alone, gap-padded over the
        #    target's columns. Needed because the row above is classified as
        #    PAIRED for both chains, which would leave the binder's unpaired
        #    MSA empty and make AlphaFold's make_msa_features raise.
        f">{QUERY_LABEL_BINDER}",
        binder_sequence + target_gaps,
    ]

    # 4. The unpaired block, chain 1: every row of the cached target MSA,
    #    gap-padded over the binder's columns. The query row's header is
    #    rewritten from >101 to >102 because in a two-chain job chain 1's query
    #    is numbered 102 (colabfold.run_mmseqs2 numbers queries from 101).
    for i, (hdr, seq) in enumerate(msa.rows):
        out.append(">" + (QUERY_LABEL_TARGET if i == 0 else hdr))
        out.append(binder_gaps + seq)

    text = "\n".join(out) + "\n"

    # Self-check against a re-implementation of the parser that will consume
    # this file. Cheap, and it is the only thing standing between a subtle
    # construction bug and 378 confident, meaningless folds.
    validate_complex_a3m(text, binder_sequence, target_sequence)
    return text


def validate_complex_a3m(
    text: str, binder_sequence: str, target_sequence: str
) -> dict[str, int]:
    """Re-parse a complex A3M the way ColabFold will, and refuse if it is wrong.

    Reproduces ``get_queries``' complex detection and ``unserialize_msa``'s
    chain splitting, then asserts that chain 0 recovers the binder, chain 1
    recovers the target, exactly one row is treated as paired (the query pair),
    and neither chain's unpaired MSA is empty.

    Returns a small summary: chain lengths, and per-chain paired and unpaired
    depths.
    """
    header_line, rows = _normalize_a3m(text)
    if header_line is None:
        raise ValueError(
            "Complex A3M has no '#' header line, so ColabFold would not treat "
            "it as a complex at all and would fold binder and target as one "
            "chain."
        )
    lengths, cardinality = parse_cardinality_header(header_line)

    expected = (len(binder_sequence), len(target_sequence))
    if lengths != expected:
        raise ValueError(
            f"Complex A3M header declares chain lengths {lengths}, expected "
            f"{expected} (binder first, target second). A flipped or wrong "
            f"header means every interface metric is computed over the wrong "
            f"block of the PAE matrix."
        )
    if cardinality != (1, 1):
        raise ValueError(
            f"Complex A3M header declares copy counts {cardinality}, expected "
            f"(1, 1). Anything else would fold extra copies of a chain."
        )
    if len(rows) < 3:
        raise ValueError(
            f"Complex A3M has {len(rows)} sequence rows. At least 3 are needed "
            f"(the query pair, the binder's own row, and the target query), or "
            f"a chain ends up with an empty MSA. unserialize_msa also raises "
            f"'Unknown file format a3m' below 3 lines."
        )

    total = sum(lengths)
    concatenated = rows[0][1]
    if concatenated != binder_sequence + target_sequence:
        raise ValueError(
            "The first sequence row of the complex A3M is not the plain "
            "concatenation of binder and target. get_queries takes that row as "
            "the sequence to fold and unserialize_msa slices it by raw "
            "character offset, so a padded or reordered first row silently "
            "folds the wrong thing."
        )

    n_paired = 0
    unpaired_depth = [0, 0]
    for i, (hdr, seq) in enumerate(rows):
        n_aligned = aligned_length(seq)
        if n_aligned != total:
            raise ValueError(
                f"Row {i} of the complex A3M (header {hdr!r}) has {n_aligned} "
                f"aligned columns, expected {total} = {lengths[0]} + "
                f"{lengths[1]}. ColabFold would truncate or drop it silently."
            )
        segments, has_aa = split_chain_segments(seq, lengths)
        if "" in segments:
            raise ValueError(
                f"Row {i} of the complex A3M (header {hdr!r}) does not split "
                f"into both chains; ColabFold would drop part of it."
            )
        if i == 0:
            if segments[0] != binder_sequence or segments[1] != target_sequence:
                raise ValueError(
                    "Splitting the query row by the header lengths does not "
                    "recover the binder and the target. The header and the "
                    "query row disagree."
                )
            if not all(has_aa):
                raise ValueError(
                    "The query row does not carry residues in both chains, so "
                    "ColabFold would not place it in the paired MSA and one "
                    "chain would be left with an empty paired alignment."
                )
        if sum(has_aa) > 1:
            n_paired += 1
        else:
            for j, flag in enumerate(has_aa):
                if flag:
                    unpaired_depth[j] += 1

    if n_paired != 1:
        raise ValueError(
            f"{n_paired} rows of the complex A3M carry residues in both chains, "
            f"expected exactly 1 (the query pair). The binder is de novo and "
            f"single-sequence, so cross-chain pairing is meaningless for it; "
            f"extra paired rows would feed AF2 an alignment that claims "
            f"co-evolutionary evidence we do not have."
        )
    for j, depth in enumerate(unpaired_depth):
        if depth == 0:
            raise ValueError(
                f"Chain {j} of the complex A3M has an empty unpaired MSA. "
                f"ColabFold passes it straight to AlphaFold's "
                f"make_msa_features, which raises on an empty MSA, so the fold "
                f"would fail after the weights were loaded and paid for."
            )

    return {
        "binder_length": lengths[0],
        "target_length": lengths[1],
        "n_rows": len(rows),
        "n_paired_rows": n_paired,
        "binder_unpaired_depth": unpaired_depth[0],
        "target_unpaired_depth": unpaired_depth[1],
    }


def write_complex_a3m(
    out_dir: Path | str,
    design_id: str,
    binder_sequence: str,
    target_sequence: str,
    target_a3m_text: str,
    *,
    min_target_depth: int = MIN_TARGET_DEPTH,
) -> Path:
    """Write the complex A3M for ``design_id`` into ``out_dir`` and return its path.

    The file name is ``colabfold_jobname(design_id) + '.a3m'`` because
    ColabFold names every output after the input's stem, so this is what keeps
    results keyed by design and re-runs resumable.
    """
    text = build_complex_a3m(
        binder_sequence,
        target_sequence,
        target_a3m_text,
        min_target_depth=min_target_depth,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / a3m_filename(design_id)
    path.write_text(text)
    return path
