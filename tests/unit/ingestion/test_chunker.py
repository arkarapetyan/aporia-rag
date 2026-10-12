"""Tests for aporia_rag.ingestion.chunker (test plan: Component 2).

Example-based tests pin down *which* boundaries the chunker picks; the
chunk sizes are chosen so that a naive greedy word-packer would cut in a
different place, so these tests fail unless the boundary logic is real.
Property-based tests then check invariants that must hold for any input.
"""

from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from aporia_rag.domain.models import Chunk, Document, SourceMetadata
from aporia_rag.ingestion.chunker import RecursiveChunker

METADATA = SourceMetadata(author="Plato", title="The Republic", source_id="gutenberg:1497")

FOUR_SENTENCES = "Alpha is first. Beta is second. Gamma is third. Delta is fourth."


def _document(text: str, doc_id: str = "republic") -> Document:
    return Document(id=doc_id, text=text, metadata=METADATA)


def _chunk(text: str, *, chunk_size: int, chunk_overlap: int = 0) -> list[Chunk]:
    chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk(_document(text))


def _texts(chunks: list[Chunk]) -> list[str]:
    return [chunk.text for chunk in chunks]


class TestConfiguration:
    @pytest.mark.parametrize(
        ("chunk_size", "chunk_overlap", "message"),
        [
            (0, 0, "chunk_size must be positive"),  # 2.1
            (-10, 0, "chunk_size must be positive"),  # 2.1
            (100, -1, "chunk_overlap must not be negative"),  # 2.2
            (100, 100, "chunk_overlap must be smaller than chunk_size"),  # 2.3
            (100, 150, "chunk_overlap must be smaller than chunk_size"),  # 2.3
        ],
    )
    def test_rejects_invalid_sizes(self, chunk_size: int, chunk_overlap: int, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


class TestDegenerateAndShortInput:
    @pytest.mark.parametrize("text", ["", "   ", "\n\n\t \n"], ids=["empty", "spaces", "mixed"])
    def test_blank_text_produces_no_chunks(self, text: str) -> None:  # 2.4
        assert _chunk(text, chunk_size=100) == []

    def test_text_that_fits_becomes_one_identical_chunk(self) -> None:  # 2.5
        assert _texts(_chunk("A short passage.", chunk_size=100)) == ["A short passage."]

    def test_surrounding_whitespace_is_trimmed_with_offsets_into_original(self) -> None:  # 2.6
        [chunk] = _chunk("  A short passage.\n", chunk_size=100)

        assert chunk.text == "A short passage."
        assert (chunk.start_char, chunk.end_char) == (2, 18)

    def test_paragraphs_that_fit_together_stay_in_one_chunk(self) -> None:  # 2.7
        text = "First paragraph.\n\nSecond paragraph."

        assert _texts(_chunk(text, chunk_size=100)) == [text]


class TestBoundaryPreference:
    def test_splits_between_paragraphs(self) -> None:  # 2.8
        text = "Socrates asks what justice is.\n\nThrasymachus scoffs at him.\n\nGlaucon objects."

        assert _texts(_chunk(text, chunk_size=70)) == [
            "Socrates asks what justice is.\n\nThrasymachus scoffs at him.",
            "Glaucon objects.",
        ]

    def test_spaces_before_a_paragraph_break_belong_to_the_break(self) -> None:  # 2.35
        text = "Socrates asks.  \n\nGlaucon objects."

        assert _texts(_chunk(text, chunk_size=20)) == ["Socrates asks.", "Glaucon objects."]

    def test_falls_back_to_sentences_inside_a_long_paragraph(self) -> None:  # 2.9
        assert _texts(_chunk(FOUR_SENTENCES, chunk_size=55)) == [
            "Alpha is first. Beta is second. Gamma is third.",
            "Delta is fourth.",
        ]

    def test_recognises_a_sentence_ending_inside_closing_quotes(self) -> None:  # 2.10
        text = "\u201cJustice is the interest of the stronger.\u201d Socrates was not convinced."

        assert _texts(_chunk(text, chunk_size=55)) == [
            "\u201cJustice is the interest of the stronger.\u201d",
            "Socrates was not convinced.",
        ]

    def test_falls_back_to_clauses_inside_a_long_sentence(self) -> None:  # 2.11
        text = (
            "The soul has three parts; reason ought to rule; "
            "spirit is its ally; appetite must obey."
        )

        assert _texts(_chunk(text, chunk_size=55)) == [
            "The soul has three parts; reason ought to rule;",
            "spirit is its ally; appetite must obey.",
        ]

    def test_falls_back_to_words_without_punctuation(self) -> None:  # 2.12
        text = "one two three four five six seven eight nine ten"

        assert _texts(_chunk(text, chunk_size=15)) == [
            "one two three",
            "four five six",
            "seven eight",
            "nine ten",
        ]

    def test_hard_splits_only_a_token_longer_than_chunk_size(self) -> None:  # 2.13
        text = "tiny supercalifragilistic tiny"

        assert _texts(_chunk(text, chunk_size=10)) == ["tiny", "supercalif", "ragilistic", "tiny"]


class TestOverlap:
    def test_zero_overlap_produces_disjoint_chunks(self) -> None:  # 2.14
        first, second = _chunk(FOUR_SENTENCES, chunk_size=55, chunk_overlap=0)

        assert second.text == "Delta is fourth."
        assert second.start_char >= first.end_char

    def test_repeats_the_trailing_sentence(self) -> None:  # 2.15
        _, second = _chunk(FOUR_SENTENCES, chunk_size=55, chunk_overlap=25)

        assert second.text == "Gamma is third. Delta is fourth."

    def test_repeats_as_many_whole_units_as_the_budget_allows(self) -> None:  # 2.16
        _, second = _chunk(FOUR_SENTENCES, chunk_size=55, chunk_overlap=31)

        assert second.text == "Beta is second. Gamma is third. Delta is fourth."

    def test_never_repeats_a_partial_unit(self) -> None:  # 2.17
        _, second = _chunk(FOUR_SENTENCES, chunk_size=55, chunk_overlap=10)

        assert second.text == "Delta is fourth."

    def test_shrinks_overlap_so_the_next_unit_still_fits(self) -> None:  # 2.18
        first, second = _chunk(FOUR_SENTENCES, chunk_size=47, chunk_overlap=31)

        assert first.text == "Alpha is first. Beta is second. Gamma is third."
        assert second.text == "Gamma is third. Delta is fourth."


class TestOverlapInsideParagraphUnits:
    """Each case is two paragraphs whose first paragraph fits in a chunk on its
    own, so it is packed as a single unit. Whole-unit overlap would start the
    second chunk at paragraph 2 with nothing repeated."""

    def test_repeats_trailing_whole_sentences_of_a_paragraph(self) -> None:  # 2.28
        text = (
            "Alpha is first. Beta is second. Gamma is third.\n\nDelta is fourth. Epsilon is fifth."
        )

        first, second = _chunk(text, chunk_size=55, chunk_overlap=25)

        assert first.text == "Alpha is first. Beta is second. Gamma is third."
        assert second.text == "Gamma is third.\n\nDelta is fourth. Epsilon is fifth."

    def test_prefers_a_whole_sentence_over_a_longer_clause_tail(self) -> None:  # 2.29
        text = "Reason rules; spirit helps. Appetite obeys.\n\nPlato agrees."

        _, second = _chunk(text, chunk_size=50, chunk_overlap=30)

        assert second.text == "Appetite obeys.\n\nPlato agrees."

    def test_falls_back_to_a_clause_when_no_sentence_fits(self) -> None:  # 2.30
        text = "Reason rules. The soul has three parts; appetite obeys.\n\nPlato agrees."

        _, second = _chunk(text, chunk_size=60, chunk_overlap=20)

        assert second.text == "appetite obeys.\n\nPlato agrees."

    def test_never_repeats_fragments_smaller_than_a_clause(self) -> None:  # 2.31
        text = "Reason rules. The soul obeys its rational part.\n\nPlato agrees."

        first, second = _chunk(text, chunk_size=50, chunk_overlap=20)

        assert second.text == "Plato agrees."
        assert second.start_char >= first.end_char

    def test_text_without_punctuation_overlaps_on_whole_words(self) -> None:  # 2.32
        text = "one two three four five six seven eight nine ten"

        assert _texts(_chunk(text, chunk_size=15, chunk_overlap=5)) == [
            "one two three",
            "three four five",
            "five six seven",
            "seven eight",
            "eight nine ten",
        ]


class TestProvenance:
    def test_chunks_carry_document_id_and_metadata(self) -> None:  # 2.19
        chunks = _chunk(FOUR_SENTENCES, chunk_size=20)

        assert all(chunk.document_id == "republic" for chunk in chunks)
        assert all(chunk.metadata == METADATA for chunk in chunks)

    def test_indices_are_sequential(self) -> None:  # 2.20
        chunks = _chunk(FOUR_SENTENCES, chunk_size=20)

        assert [chunk.index for chunk in chunks] == list(range(len(chunks)))

    def test_ids_are_unique_and_deterministic(self) -> None:  # 2.21
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=5)
        document = _document(FOUR_SENTENCES)

        first_run = chunker.chunk(document)
        second_run = chunker.chunk(document)

        assert first_run == second_run
        assert len({chunk.id for chunk in first_run}) == len(first_run)

    def test_ids_do_not_collide_across_documents(self) -> None:  # 2.22
        chunker = RecursiveChunker(chunk_size=20)

        republic = chunker.chunk(_document(FOUR_SENTENCES, doc_id="republic"))
        ethics = chunker.chunk(_document(FOUR_SENTENCES, doc_id="ethics"))

        assert {c.id for c in republic}.isdisjoint({c.id for c in ethics})


# --------------------------------------------------------------------------
# Property-based invariants (2.23 - 2.27)
# --------------------------------------------------------------------------

_WORD = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=14)
_GAP = st.sampled_from([" ", " ", " ", ". ", "; ", "? ", ", ", "\n", "\n\n", "\n \n", ".\u201d "])
_PROSE = st.lists(st.tuples(_WORD, _GAP), max_size=250).map(
    lambda pairs: "".join(word + gap for word, gap in pairs)
)
_RAW = st.text(alphabet="abc .;:!?\u201d\n\t", max_size=600)


@st.composite
def chunking_cases(draw: st.DrawFn) -> tuple[Document, RecursiveChunker]:
    text = draw(st.one_of(_PROSE, _RAW))
    chunk_size = draw(st.integers(min_value=1, max_value=200))
    chunk_overlap = draw(st.integers(min_value=0, max_value=chunk_size - 1))
    return _document(text), RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


class TestInvariants:
    @given(chunking_cases())
    def test_offsets_slice_back_to_chunk_text(
        self, case: tuple[Document, RecursiveChunker]
    ) -> None:  # 2.23
        document, chunker = case
        for chunk in chunker.chunk(document):
            assert document.text[chunk.start_char : chunk.end_char] == chunk.text

    @given(chunking_cases())
    def test_no_chunk_exceeds_chunk_size(
        self, case: tuple[Document, RecursiveChunker]
    ) -> None:  # 2.24
        document, chunker = case
        assert all(len(chunk.text) <= chunker.chunk_size for chunk in chunker.chunk(document))

    @given(chunking_cases())
    def test_every_non_whitespace_character_is_covered(
        self, case: tuple[Document, RecursiveChunker]
    ) -> None:  # 2.25
        document, chunker = case
        covered: set[int] = set()
        for chunk in chunker.chunk(document):
            covered.update(range(chunk.start_char, chunk.end_char))

        uncovered = [
            i for i, char in enumerate(document.text) if not char.isspace() and i not in covered
        ]
        assert uncovered == []

    @given(chunking_cases())
    def test_chunks_advance_monotonically(
        self, case: tuple[Document, RecursiveChunker]
    ) -> None:  # 2.26
        document, chunker = case
        chunks = chunker.chunk(document)
        for previous, current in pairwise(chunks):
            assert current.start_char > previous.start_char
            assert current.end_char > previous.end_char

    @given(chunking_cases())
    def test_overlap_never_exceeds_budget(
        self, case: tuple[Document, RecursiveChunker]
    ) -> None:  # 2.27
        document, chunker = case
        chunks = chunker.chunk(document)
        for previous, current in pairwise(chunks):
            shared = max(0, previous.end_char - current.start_char)
            assert shared <= chunker.chunk_overlap

    @given(chunking_cases())
    def test_chunks_never_start_or_end_with_whitespace(
        self, case: tuple[Document, RecursiveChunker]
    ) -> None:  # 2.34
        document, chunker = case
        for chunk in chunker.chunk(document):
            assert not chunk.text[0].isspace()
            assert not chunk.text[-1].isspace()


# Prose where overlap is always achievable: every sentence (<= 28 chars) fits the
# smallest budget (30), and paragraphs (<= 144 chars) are small enough that two of
# them plus a repeated sentence always fit the smallest chunk_size (300).
_SHORT_WORD = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=6)
_SHORT_SENTENCE = st.lists(_SHORT_WORD, min_size=1, max_size=4).map(lambda w: " ".join(w) + ".")
_SHORT_PARAGRAPH = st.lists(_SHORT_SENTENCE, min_size=1, max_size=5).map(" ".join)


@st.composite
def overlap_friendly_cases(draw: st.DrawFn) -> tuple[Document, RecursiveChunker]:
    text = "\n\n".join(draw(st.lists(_SHORT_PARAGRAPH, min_size=1, max_size=40)))
    chunk_size = draw(st.integers(min_value=300, max_value=400))
    chunk_overlap = draw(st.integers(min_value=30, max_value=100))
    return _document(text), RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


class TestOverlapIsAppliedWhenPossible:
    @given(overlap_friendly_cases())
    def test_every_boundary_overlaps(self, case: tuple[Document, RecursiveChunker]) -> None:  # 2.33
        document, chunker = case
        chunks = chunker.chunk(document)
        for previous, current in pairwise(chunks):
            assert current.start_char < previous.end_char
