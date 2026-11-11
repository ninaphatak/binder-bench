"""Tests for the paired A3M construction.

Written the same way as tests/test_metrics.py: the point is that the guards
FAIL, not that the happy path succeeds. A wrong A3M does not crash ColabFold.
It folds, returns five models, an ipTM and a PAE matrix, and every number is
meaningless. So most of what follows checks that we refuse.

The expected format was read out of ColabFold 1.6.2's own source (dumped from
the pinned container image), and the construction was additionally verified
offline against that source's ``unserialize_msa`` and ``msa_to_str``: the text
produced here is byte-identical to ColabFold's own serialisation of
(binder single-sequence, target deep MSA). That check cannot live in this suite
because it needs the container's source tree, so the invariants it established
are asserted directly below instead.
"""

from __future__ import annotations

import pytest

from binder_bench.msa import (
    MIN_TARGET_DEPTH,
    QUERY_LABEL_BINDER,
    QUERY_LABEL_TARGET,
    a3m_filename,
    aligned_length,
    assert_unique_jobnames,
    build_complex_a3m,
    colabfold_jobname,
    format_cardinality_header,
    parse_cardinality_header,
    parse_target_a3m,
    split_chain_segments,
    validate_complex_a3m,
    write_complex_a3m,
)
from binder_bench.target import EXPECTED_LENGTH, target_sequence

# A short synthetic target keeps the fixtures readable. The real 621-residue
# construct gets its own test further down.
TARGET = "ACDEFGHIKLMNPQRSTVWY"  # 20 residues
BINDER = "MKTWYQ"  # 6 residues

HITS = [
    (">UniRef100_A\tdescription with a tab", "ACDEFGHIKLMNPQRSTVWY"),
    (">UniRef100_B", "acdACDEFGHIKLMNPQRSTVW-"),  # leading insertion
    (">UniRef100_C", "ACDEFghiGHIKLMNPQRSTVWY"),  # interior insertion
    (">UniRef100_D", "-CDEFGHIKLMNPQRSTVW-"),  # gaps at both ends
]


def cached_a3m(
    rows: list[tuple[str, str]] | None = None,
    *,
    declared: int | None = None,
    cardinality: str = "1",
    query: str | None = None,
) -> str:
    """A cached target-only A3M of the shape colabfold_batch --msa-only writes."""
    if rows is None:
        rows = [(f">{QUERY_LABEL_BINDER}", TARGET if query is None else query)] + HITS
    n = len(TARGET) if declared is None else declared
    body = "\n".join(f"{h}\n{s}" for h, s in rows)
    return f"#{n}\t{cardinality}\n{body}\n"


def rows_of(text: str) -> list[tuple[str, str]]:
    """Header/sequence pairs of an A3M, skipping the '#' line."""
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    return [(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)]


# ---------------------------------------------------------------------------
# The header: what makes ColabFold treat the job as a two-chain complex
# ---------------------------------------------------------------------------


def test_header_declares_two_chains_binder_first():
    """#<binder_len>,<target_len> TAB 1,1 . Exactly this, nothing else."""
    text = build_complex_a3m(BINDER, TARGET, cached_a3m())
    assert text.splitlines()[0] == "#6,20\t1,1"


def test_header_for_the_real_target_puts_621_second():
    target = target_sequence()
    text = build_complex_a3m(
        BINDER, target, cached_a3m(rows=[(">101", target)] + [(">h", target)], declared=EXPECTED_LENGTH)
    )
    assert text.splitlines()[0] == f"#{len(BINDER)},{EXPECTED_LENGTH}\t1,1"
    # Target first would be the silent catastrophe. Assert it is not that.
    assert text.splitlines()[0] != f"#{EXPECTED_LENGTH},{len(BINDER)}\t1,1"


def test_header_separator_is_a_tab_not_a_space():
    """get_queries splits on '\\t'. A space would make it a single-chain file."""
    header = build_complex_a3m(BINDER, TARGET, cached_a3m()).splitlines()[0]
    assert "\t" in header
    assert " " not in header
    assert len(header[1:].split("\t")) == 2


def test_header_round_trips_through_colabfolds_own_grammar():
    text = build_complex_a3m(BINDER, TARGET, cached_a3m())
    lengths, cardinality = parse_cardinality_header(text.splitlines()[0])
    assert lengths == (len(BINDER), len(TARGET))
    assert cardinality == (1, 1)
    # len(lengths) > 1 is precisely what sets is_complex in get_queries.
    assert len(lengths) > 1


def test_format_cardinality_header_rejects_mismatched_lists():
    with pytest.raises(ValueError, match="parallel lists"):
        format_cardinality_header((10, 20), (1,))


# ---------------------------------------------------------------------------
# Binder-first ordering, reflected in the file and not just the header
# ---------------------------------------------------------------------------


def test_first_sequence_row_is_the_plain_concatenated_query():
    """get_queries takes seqs[0] as the sequence to fold and unserialize_msa
    slices it by raw character offset, so it must be binder+target with no
    gaps, no lowercase and no separator."""
    text = build_complex_a3m(BINDER, TARGET, cached_a3m())
    rows = rows_of(text)
    assert rows[0][0] == f">{QUERY_LABEL_BINDER}\t{QUERY_LABEL_TARGET}"
    assert rows[0][1] == BINDER + TARGET
    assert "-" not in rows[0][1]
    assert rows[0][1] == rows[0][1].upper()


def test_binder_occupies_the_leading_columns_of_every_row():
    text = build_complex_a3m(BINDER, TARGET, cached_a3m())
    rows = rows_of(text)
    # The binder's own row: residues first, then the target's columns gapped.
    assert rows[1][1] == BINDER + "-" * len(TARGET)
    # Every target row: the binder's columns gapped, then the alignment.
    for _, seq in rows[2:]:
        assert seq.startswith("-" * len(BINDER))
        # The binder's columns are ALL gap in a target row: no residue of the
        # target may land in them, or the chain boundary has moved.
        assert aligned_length(seq[: len(BINDER)]) == len(BINDER)
        assert set(seq[: len(BINDER)]) == {"-"}


def test_splitting_the_query_row_recovers_binder_then_target_not_the_reverse():
    text = build_complex_a3m(BINDER, TARGET, cached_a3m())
    lengths, _ = parse_cardinality_header(text.splitlines()[0])
    segments, has_aa = split_chain_segments(rows_of(text)[0][1], lengths)
    assert segments == [BINDER, TARGET]
    assert segments != [TARGET, BINDER]
    assert has_aa == [True, True]


def test_round_trip_recovers_the_whole_target_msa_against_the_binder():
    """The full round trip: build, then split every row the way ColabFold does."""
    text = build_complex_a3m(BINDER, TARGET, cached_a3m())
    lengths, _ = parse_cardinality_header(text.splitlines()[0])

    recovered_binder = []
    recovered_target = []
    for _, seq in rows_of(text):
        segments, has_aa = split_chain_segments(seq, lengths)
        if has_aa[0]:
            recovered_binder.append(segments[0])
        if has_aa[1]:
            recovered_target.append(segments[1])

    # The binder appears in the query pair and in its own single-sequence row.
    assert recovered_binder == [BINDER, BINDER]
    # The target appears in the query pair plus every row of the cached MSA.
    assert recovered_target[0] == TARGET  # from the query pair
    assert recovered_target[1] == TARGET  # the cached MSA's own query row
    assert len(recovered_target) == 2 + len(HITS)
    assert recovered_target[2] == HITS[0][1]
    assert recovered_target[3] == HITS[1][1]  # leading insertion preserved


def test_target_msa_depth_is_preserved_not_truncated():
    text = build_complex_a3m(BINDER, TARGET, cached_a3m())
    # One header row for the query pair, one for the binder, then the whole MSA.
    assert len(rows_of(text)) == 2 + (1 + len(HITS))


def test_hit_headers_are_passed_through_verbatim_tabs_included():
    text = build_complex_a3m(BINDER, TARGET, cached_a3m())
    headers = [h for h, _ in rows_of(text)]
    assert HITS[0][0] in headers


def test_target_query_row_is_renumbered_to_the_second_chain():
    """In a two-chain job ColabFold numbers chain 1's query >102."""
    text = build_complex_a3m(BINDER, TARGET, cached_a3m())
    headers = [h for h, _ in rows_of(text)]
    assert headers[1] == f">{QUERY_LABEL_BINDER}"
    assert headers[2] == f">{QUERY_LABEL_TARGET}"


def test_binder_gets_its_own_unpaired_row():
    """Without it the binder's unpaired MSA is empty and AlphaFold raises
    after the weights have been loaded and paid for."""
    summary = validate_complex_a3m(
        build_complex_a3m(BINDER, TARGET, cached_a3m()), BINDER, TARGET
    )
    assert summary["binder_unpaired_depth"] == 1
    assert summary["target_unpaired_depth"] == 1 + len(HITS)


def test_exactly_one_row_is_treated_as_paired():
    """Pairing is meaningless for a de novo binder, so only the query pair
    may carry residues in both chains."""
    summary = validate_complex_a3m(
        build_complex_a3m(BINDER, TARGET, cached_a3m()), BINDER, TARGET
    )
    assert summary["n_paired_rows"] == 1


def test_build_works_with_the_real_621_residue_construct():
    target = target_sequence()
    cached = (
        f"#{EXPECTED_LENGTH}\t1\n"
        f">101\n{target}\n"
        f">UniRef100_X\n{target[:-1]}-\n"
        f">UniRef100_Y\nqq{target}\n"
    )
    text = build_complex_a3m(BINDER, target, cached)
    summary = validate_complex_a3m(text, BINDER, target)
    assert summary == {
        "binder_length": len(BINDER),
        "target_length": EXPECTED_LENGTH,
        "n_rows": 5,
        "n_paired_rows": 1,
        "binder_unpaired_depth": 1,
        "target_unpaired_depth": 3,
    }


# ---------------------------------------------------------------------------
# Lowercase insertion characters
# ---------------------------------------------------------------------------


def test_aligned_length_ignores_lowercase_but_counts_gaps():
    """ColabFold's rule: a column counts unless it is a lowercase letter."""
    assert aligned_length("ACDE") == 4
    assert aligned_length("ACacDE") == 4
    assert aligned_length("A-C-") == 4
    assert aligned_length("abc") == 0
    assert aligned_length("") == 0


def test_a_row_with_insertions_is_not_rejected_for_being_longer():
    """Its aligned length matches even though its character count does not."""
    row = "acdACDEFGHIKLMNPQRSTVW-"
    assert len(row) != len(TARGET)
    assert aligned_length(row) == len(TARGET)
    msa = parse_target_a3m(cached_a3m(), TARGET)
    assert any(seq == row for _, seq in msa.rows)


def test_gap_padding_a_row_that_starts_with_an_insertion_is_safe():
    """The binder segment closes on its last dash, so a leading lowercase run
    in a target row stays in the target segment."""
    text = build_complex_a3m(BINDER, TARGET, cached_a3m())
    lengths, _ = parse_cardinality_header(text.splitlines()[0])
    row = next(seq for _, seq in rows_of(text) if "acd" in seq)
    segments, has_aa = split_chain_segments(row, lengths)
    assert segments[0] == "-" * len(BINDER)
    assert segments[1] == "acdACDEFGHIKLMNPQRSTVW-"
    assert has_aa == [False, True]


def test_insertions_do_not_shift_the_chain_boundary():
    interior = next(seq for _, seq in rows_of(
        build_complex_a3m(BINDER, TARGET, cached_a3m())
    ) if "ghi" in seq)
    segments, _ = split_chain_segments(interior, (len(BINDER), len(TARGET)))
    assert segments[0] == "-" * len(BINDER)
    assert aligned_length(segments[1]) == len(TARGET)


def test_trailing_insertions_on_the_last_chain_are_dropped_by_colabfold():
    """Documented inherited behaviour, not a bug we introduce: the target is
    the last chain, and _split_by_aln ends its segment on the last aligned
    column, so C-terminal insertions fall off. Identical for every design."""
    trailing = "ACDEFGHIKLMNPQRSTVWYxyz"
    text = build_complex_a3m(
        BINDER, TARGET, cached_a3m(rows=[(">101", TARGET), (">hit", trailing)])
    )
    # We write the row intact...
    assert any(seq.endswith("xyz") for _, seq in rows_of(text))
    # ...and ColabFold's splitter will then ignore the tail.
    segments, _ = split_chain_segments(
        "-" * len(BINDER) + trailing, (len(BINDER), len(TARGET))
    )
    assert segments[1] == "ACDEFGHIKLMNPQRSTVWY"


def test_lowercase_in_the_query_row_of_the_cached_msa_is_rejected():
    """The query is the reference, so it cannot hold insertions; and
    unserialize_msa slices the query row by raw offset, with no insertion
    handling at all."""
    with pytest.raises(ValueError, match="not the target construct"):
        parse_target_a3m(cached_a3m(query=TARGET.lower()), TARGET)


# ---------------------------------------------------------------------------
# Loud failures: the cached target MSA
# ---------------------------------------------------------------------------


def test_declared_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="declares a chain length of 999"):
        build_complex_a3m(BINDER, TARGET, cached_a3m(declared=999))


def test_declared_length_mismatch_names_the_consequence():
    with pytest.raises(ValueError, match="plausible and meaningless"):
        parse_target_a3m(cached_a3m(declared=19), TARGET)


def test_real_target_against_an_msa_staged_for_a_different_construct():
    """The 621-residue construct against a cached MSA that declares 620."""
    target = target_sequence()
    cached = f"#620\t1\n>101\n{target}\n>hit\n{target}\n"
    with pytest.raises(ValueError, match=r"declares a chain length of 620 but"):
        build_complex_a3m(BINDER, target, cached)


def test_wrong_query_sequence_is_rejected_even_at_the_right_length():
    """Same length, different protein: the single most dangerous case, because
    nothing downstream would notice."""
    wrong = TARGET[::-1]
    assert len(wrong) == len(TARGET)
    with pytest.raises(ValueError, match="not the target construct"):
        build_complex_a3m(BINDER, TARGET, cached_a3m(query=wrong))


def test_wrong_query_sequence_error_states_the_consequence():
    with pytest.raises(ValueError, match="predicted for another"):
        parse_target_a3m(cached_a3m(query=TARGET[::-1]), TARGET)


def test_query_sequence_off_by_one_residue_is_rejected():
    mutated = "W" + TARGET[1:]
    with pytest.raises(ValueError, match="not the target construct"):
        build_complex_a3m(BINDER, TARGET, cached_a3m(query=mutated))


def test_empty_cached_a3m_is_rejected():
    with pytest.raises(ValueError, match="cached target A3M is empty"):
        build_complex_a3m(BINDER, TARGET, "")


def test_whitespace_only_cached_a3m_is_rejected():
    with pytest.raises(ValueError, match="cached target A3M is empty"):
        build_complex_a3m(BINDER, TARGET, "\n  \n\t\n")


def test_header_only_cached_a3m_is_rejected():
    with pytest.raises(ValueError, match="no sequences"):
        build_complex_a3m(BINDER, TARGET, f"#{len(TARGET)}\t1\n")


def test_missing_cardinality_header_is_rejected():
    body = f">101\n{TARGET}\n>hit\n{TARGET}\n"
    with pytest.raises(ValueError, match="no '#' cardinality header"):
        build_complex_a3m(BINDER, TARGET, body)


def test_header_with_one_field_is_rejected():
    with pytest.raises(ValueError, match="tab-separated fields, expected 2"):
        parse_cardinality_header("#621")


def test_header_with_three_fields_is_rejected():
    with pytest.raises(ValueError, match="tab-separated fields, expected 2"):
        parse_cardinality_header("#621\t1\t1")


def test_header_without_hash_is_rejected_with_the_consequence_named():
    with pytest.raises(ValueError, match="fold binder and target as ONE chain"):
        parse_cardinality_header("621\t1")


def test_non_integer_header_is_rejected():
    with pytest.raises(ValueError, match="not integers"):
        parse_cardinality_header("#621x\t1")


def test_non_positive_length_in_header_is_rejected():
    with pytest.raises(ValueError, match="non-positive"):
        parse_cardinality_header("#-621\t1")


def test_cached_msa_declaring_two_chains_is_rejected():
    text = f"#10,10\t1,1\n>101\t102\n{TARGET}\n>hit\n{TARGET}\n"
    with pytest.raises(ValueError, match="declares 2 chains"):
        build_complex_a3m(BINDER, TARGET, text)


def test_cached_msa_declaring_a_homooligomer_is_rejected():
    with pytest.raises(ValueError, match="one chain with one copy"):
        build_complex_a3m(BINDER, TARGET, cached_a3m(cardinality="2"))


def test_row_with_too_few_aligned_columns_is_rejected():
    short = TARGET[:-3]
    with pytest.raises(ValueError, match="aligned columns, expected 20"):
        build_complex_a3m(
            BINDER, TARGET, cached_a3m(rows=[(">101", TARGET), (">hit", short)])
        )


def test_row_with_too_many_aligned_columns_is_rejected():
    long = TARGET + "ACD"
    with pytest.raises(ValueError, match="aligned columns, expected 20"):
        build_complex_a3m(
            BINDER, TARGET, cached_a3m(rows=[(">101", TARGET), (">hit", long)])
        )


def test_row_length_error_explains_the_silent_failure():
    with pytest.raises(ValueError, match="silently"):
        parse_target_a3m(
            cached_a3m(rows=[(">101", TARGET), (">hit", TARGET + "A")]), TARGET
        )


def test_depth_one_target_msa_is_rejected():
    """A depth-1 alignment turns the run into single-sequence prediction for
    both chains while still returning confident looking scores."""
    with pytest.raises(ValueError, match="depth 1, below the required minimum"):
        build_complex_a3m(BINDER, TARGET, cached_a3m(rows=[(">101", TARGET)]))


def test_depth_requirement_is_configurable_but_not_silent():
    text = build_complex_a3m(
        BINDER, TARGET, cached_a3m(rows=[(">101", TARGET)]), min_target_depth=1
    )
    assert validate_complex_a3m(text, BINDER, TARGET)["target_unpaired_depth"] == 1
    assert MIN_TARGET_DEPTH == 2


def test_illegal_character_in_a_row_is_rejected():
    """'.' is not lowercase, so ColabFold counts it as an ALIGNED column and
    the chain boundary moves."""
    with pytest.raises(ValueError, match=r"contains \['\.'\]"):
        build_complex_a3m(
            BINDER,
            TARGET,
            cached_a3m(rows=[(">101", TARGET), (">hit", TARGET[:-1] + ".")]),
        )


def test_star_in_a_row_is_rejected():
    with pytest.raises(ValueError, match=r"contains \['\*'\]"):
        parse_target_a3m(
            cached_a3m(rows=[(">101", TARGET), (">hit", TARGET[:-1] + "*")]), TARGET
        )


def test_second_comment_line_is_rejected():
    """normalize_a3m keeps only the first line as a header and splices any
    later '#' line into the sequence above it."""
    text = cached_a3m() + "# a note\n"
    with pytest.raises(ValueError, match="Comment line at line"):
        build_complex_a3m(BINDER, TARGET, text)


def test_sequence_before_any_header_is_rejected():
    text = f"#{len(TARGET)}\t1\n{TARGET}\n>101\n{TARGET}\n"
    with pytest.raises(ValueError, match="before any '>' header"):
        build_complex_a3m(BINDER, TARGET, text)


def test_empty_sequence_row_is_rejected():
    text = f"#{len(TARGET)}\t1\n>101\n{TARGET}\n>hit\n>hit2\n{TARGET}\n"
    with pytest.raises(ValueError, match="has no\\s+sequence"):
        build_complex_a3m(BINDER, TARGET, text)


def test_wrapped_sequence_lines_are_joined():
    text = f"#{len(TARGET)}\t1\n>101\n{TARGET[:9]}\n{TARGET[9:]}\n>hit\n{TARGET}\n"
    msa = parse_target_a3m(text, TARGET)
    assert msa.query_sequence == TARGET
    assert msa.depth == 2


def test_null_separators_and_crlf_are_tolerated():
    text = (f"#{len(TARGET)}\t1\r\n>101\r\n{TARGET}\r\n>hit\r\n{TARGET}\x00\r\n")
    msa = parse_target_a3m(text, TARGET)
    assert msa.depth == 2
    assert msa.hits[0][1] == TARGET


# ---------------------------------------------------------------------------
# Loud failures: the query sequences themselves
# ---------------------------------------------------------------------------


def test_empty_binder_is_rejected():
    with pytest.raises(ValueError, match="binder sequence is empty"):
        build_complex_a3m("", TARGET, cached_a3m())


def test_binder_with_a_gap_is_rejected():
    with pytest.raises(ValueError, match="not an uppercase residue letter"):
        build_complex_a3m("MKT-WYQ", TARGET, cached_a3m())


def test_binder_with_lowercase_is_rejected():
    with pytest.raises(ValueError, match="not an uppercase residue letter"):
        build_complex_a3m("MKTwYQ", TARGET, cached_a3m())


def test_binder_with_a_chain_separator_is_rejected():
    """':' is ColabFold's FASTA complex separator and has no meaning here."""
    with pytest.raises(ValueError, match="not an uppercase residue letter"):
        build_complex_a3m("MKT:WYQ", TARGET, cached_a3m())


def test_binder_with_whitespace_is_rejected():
    with pytest.raises(ValueError, match="not an uppercase residue letter"):
        build_complex_a3m("MKT WYQ", TARGET, cached_a3m())


def test_empty_target_is_rejected():
    with pytest.raises(ValueError, match="target sequence is empty"):
        build_complex_a3m(BINDER, "", cached_a3m())


def test_target_with_gaps_is_rejected():
    with pytest.raises(ValueError, match="not an uppercase residue letter"):
        build_complex_a3m(BINDER, TARGET[:-1] + "-", cached_a3m())


# ---------------------------------------------------------------------------
# validate_complex_a3m catches hand-built or mutated files too
# ---------------------------------------------------------------------------


def test_validation_rejects_a_target_first_file():
    """The flip that would make every interface metric meaningless."""
    text = (
        f"#{len(TARGET)},{len(BINDER)}\t1,1\n"
        f">102\t101\n{TARGET}{BINDER}\n"
        f">101\n{'-' * len(TARGET)}{BINDER}\n"
        f">102\n{TARGET}{'-' * len(BINDER)}\n"
    )
    with pytest.raises(ValueError, match="binder first, target second"):
        validate_complex_a3m(text, BINDER, TARGET)


def test_validation_rejects_a_gap_padded_first_row():
    """ColabFold's own pad_sequences would emit this for an unpaired-only
    complex, and unserialize_msa would then slice a 'chain' made of dashes."""
    text = build_complex_a3m(BINDER, TARGET, cached_a3m())
    rows = text.splitlines()
    # Drop the paired query pair, leaving the binder's padded row first.
    mutated = "\n".join([rows[0]] + rows[3:]) + "\n"
    with pytest.raises(ValueError, match="not the plain concatenation"):
        validate_complex_a3m(mutated, BINDER, TARGET)


def test_validation_rejects_a_file_with_no_paired_query_row():
    text = (
        f"#{len(BINDER)},{len(TARGET)}\t1,1\n"
        f">101\n{BINDER}{TARGET}\n"
        f">101\n{BINDER}{'-' * len(TARGET)}\n"
    )
    with pytest.raises(ValueError, match="At least 3 are needed"):
        validate_complex_a3m(text, BINDER, TARGET)


def test_validation_rejects_an_empty_binder_alignment():
    """Query pair plus target rows only: the binder's unpaired MSA is empty
    and AlphaFold raises after the weights are loaded."""
    text = (
        f"#{len(BINDER)},{len(TARGET)}\t1,1\n"
        f">101\t102\n{BINDER}{TARGET}\n"
        f">102\n{'-' * len(BINDER)}{TARGET}\n"
        f">hit\n{'-' * len(BINDER)}{TARGET}\n"
    )
    with pytest.raises(ValueError, match="Chain 0 .* empty unpaired MSA"):
        validate_complex_a3m(text, BINDER, TARGET)


def test_validation_rejects_spurious_cross_chain_pairing():
    """A row with residues in both chains claims co-evolutionary evidence
    that a de novo binder cannot have."""
    text = (
        f"#{len(BINDER)},{len(TARGET)}\t1,1\n"
        f">101\t102\n{BINDER}{TARGET}\n"
        f">101\n{BINDER}{'-' * len(TARGET)}\n"
        f">102\n{'-' * len(BINDER)}{TARGET}\n"
        f">bogus\tpair\n{BINDER}{TARGET}\n"
    )
    with pytest.raises(ValueError, match="carry residues in both chains"):
        validate_complex_a3m(text, BINDER, TARGET)


def test_validation_rejects_a_wrong_row_width():
    text = build_complex_a3m(BINDER, TARGET, cached_a3m())
    mutated = text.replace("-" * len(BINDER) + TARGET, "-" * len(BINDER) + TARGET[:-2], 1)
    with pytest.raises(ValueError, match="aligned columns, expected 26"):
        validate_complex_a3m(mutated, BINDER, TARGET)


def test_validation_rejects_a_copy_count_other_than_one():
    text = build_complex_a3m(BINDER, TARGET, cached_a3m()).replace(
        "\t1,1", "\t1,2", 1
    )
    with pytest.raises(ValueError, match=r"copy counts \(1, 2\)"):
        validate_complex_a3m(text, BINDER, TARGET)


def test_split_chain_segments_reports_a_row_that_runs_out_of_columns():
    segments, has_aa = split_chain_segments("MKTWYQ" + "ACDE", (6, 20))
    assert segments[0] == "MKTWYQ"
    assert segments[1] == ""  # ColabFold would silently drop this chain
    assert has_aa == [True, False]


# ---------------------------------------------------------------------------
# File naming: keeping results keyed by design_id
# ---------------------------------------------------------------------------


def test_jobname_matches_colabfolds_safe_filename():
    """120 of the 402 competition design_ids contain spaces."""
    assert (
        colabfold_jobname("Aurelia Bustos.aureliabustos.bce_var433")
        == "Aurelia_Bustos.aureliabustos.bce_var433"
    )
    assert colabfold_jobname("control.Cetuximab_scFv") == "control.Cetuximab_scFv"
    assert colabfold_jobname("a/b") == "a_b"


def test_a3m_filename_stem_is_the_jobname():
    """ColabFold names every output after the input file's stem."""
    name = a3m_filename("Alan Blakely.alan.blakely.12_6")
    assert name == "Alan_Blakely.alan.blakely.12_6.a3m"
    assert name.removesuffix(".a3m") == colabfold_jobname(
        "Alan Blakely.alan.blakely.12_6"
    )


def test_empty_design_id_is_rejected():
    with pytest.raises(ValueError, match="design_id is empty"):
        a3m_filename("")


def test_colliding_design_ids_are_rejected():
    with pytest.raises(ValueError, match="collide after ColabFold's safe_filename"):
        assert_unique_jobnames(["a b", "a_b"])


def test_unique_design_ids_map_cleanly():
    mapping = assert_unique_jobnames(["a b", "c.d", "e-f"])
    assert mapping == {"a b": "a_b", "c.d": "c.d", "e-f": "e-f"}


def test_write_complex_a3m_writes_the_text_under_the_jobname(tmp_path):
    path = write_complex_a3m(
        tmp_path / "msas", "Alan Blakely.alan.blakely.12_6", BINDER, TARGET, cached_a3m()
    )
    assert path.name == "Alan_Blakely.alan.blakely.12_6.a3m"
    assert path.read_text() == build_complex_a3m(BINDER, TARGET, cached_a3m())
    assert path.read_text().endswith("\n")


def test_write_complex_a3m_does_not_write_a_rejected_design(tmp_path):
    out = tmp_path / "msas"
    with pytest.raises(ValueError):
        write_complex_a3m(out, "d1", "MKT-WYQ", TARGET, cached_a3m())
    assert not out.exists() or list(out.iterdir()) == []


def test_two_designs_in_one_directory_share_the_cached_msa(tmp_path):
    """One MSA, reused. The directory is what gets handed to colabfold_batch."""
    cached = cached_a3m()
    p1 = write_complex_a3m(tmp_path, "d1", BINDER, TARGET, cached)
    p2 = write_complex_a3m(tmp_path, "d2", "WWWWWWWW", TARGET, cached)
    assert p1 != p2
    assert p1.read_text().splitlines()[0] == "#6,20\t1,1"
    assert p2.read_text().splitlines()[0] == "#8,20\t1,1"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["d1.a3m", "d2.a3m"]
