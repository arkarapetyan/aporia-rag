"""Structure-aware recursive chunking with exact source offsets.

Design:

* **Largest natural boundary first.** Text is split into *units* at the
  coarsest boundary that makes every unit fit: paragraph, then sentence,
  then clause (``;`` / ``:``), then word, and only as a last resort raw
  characters. A philosophical argument usually lives in a paragraph, so
  cutting one in half when we don't have to throws away the unit of
  reasoning the graders later have to judge.
* **Spans, not strings.** Units and chunks are ``[start, end)`` offsets into
  the original text and chunk text is a slice of it, never a re-joined
  string. Original whitespace is preserved and every chunk can be traced
  to its exact position for citations.
* **Boundary-aligned overlap.** Each chunk after the first repeats the
  longest tail of the previous chunk that fits in ``chunk_overlap`` and
  starts at a whole-sentence boundary; failing that, at a clause boundary;
  failing that (text with no sentence or clause punctuation), at a unit
  boundary. This looks *inside* the last unit, so a paragraph-sized unit
  still contributes its closing sentence(s) to the next chunk, which is the
  context a reader needs when an argument runs across the boundary
  ("Therefore...", "To this Glaucon replied..."). Overlap never starts
  mid-word, and never at a fragment smaller than a clause when the text
  has punctuation.

Sizes are measured in characters, which is a proxy for embedding-model
tokens (roughly 4 characters per token for English prose).
"""

import re
from dataclasses import dataclass
from typing import Final, NamedTuple

from aporia_rag.domain.models import Chunk, Document

# Every boundary pattern matches a whole whitespace run, from its first character
# to its last. So the pieces between matches never start or end with whitespace,
# and a match's end is always the first character of the next piece (overlap
# relies on this).
_PARAGRAPH: Final = re.compile(r"\s*\n\s*\n\s*")
# Sentence end, optionally followed by a closing quote or bracket. Python only
# supports fixed-width look-behinds, hence the two alternatives.
_SENTENCE: Final = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"'\u201d\u2019)\]]))\s+")
_CLAUSE: Final = re.compile(r"(?<=[;:])\s+")
_WORD: Final = re.compile(r"\s+")

_BOUNDARIES: Final = (_PARAGRAPH, _SENTENCE, _CLAUSE, _WORD)


class _Span(NamedTuple):
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class RecursiveChunker:
    chunk_size: int
    chunk_overlap: int = 0

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must not be negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size, "
                f"got {self.chunk_overlap} >= {self.chunk_size}"
            )

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        whole = _strip(text, _Span(0, len(text)))
        if whole is None:
            return []

        units = self._split_into_units(text, whole, level=0)
        return [
            Chunk(
                id=f"{document.id}#{index}",
                document_id=document.id,
                index=index,
                text=text[span.start : span.end],
                start_char=span.start,
                end_char=span.end,
                metadata=document.metadata,
            )
            for index, span in enumerate(self._pack(text, units))
        ]

    def _split_into_units(self, text: str, span: _Span, level: int) -> list[_Span]:
        """Recursively split `span` until every resulting unit fits in chunk_size."""
        if span.length <= self.chunk_size:
            return [span]
        if level == len(_BOUNDARIES):
            # A single whitespace-free token longer than chunk_size: hard split.
            return [
                _Span(start, min(start + self.chunk_size, span.end))
                for start in range(span.start, span.end, self.chunk_size)
            ]
        units: list[_Span] = []
        for piece in _split_at(text, span, _BOUNDARIES[level]):
            units.extend(self._split_into_units(text, piece, level + 1))
        return units

    def _pack(self, text: str, units: list[_Span]) -> list[_Span]:
        """Greedily merge consecutive units into chunks of at most chunk_size.

        A chunk spans from `start` (a character position) to the end of its
        last unit. The first chunk starts at the first unit; every later one
        starts wherever `_overlap_start` decides, possibly inside the previous
        chunk's last unit. `units` is never empty: `chunk` returns early for
        blank documents.
        """
        chunks: list[_Span] = []
        start = units[0].start
        next_unit = 0  # first unit not yet included in any chunk
        while True:
            # units[next_unit] always fits after `start`; _overlap_start guarantees it.
            end = next_unit + 1  # exclusive
            while end < len(units) and units[end].end - start <= self.chunk_size:
                end += 1
            chunk = _Span(start, units[end - 1].end)
            chunks.append(chunk)
            if end == len(units):
                return chunks
            start = self._overlap_start(text, chunk, units[next_unit:end], units[end])
            next_unit = end

    def _overlap_start(
        self, text: str, chunk: _Span, chunk_units: list[_Span], next_unit: _Span
    ) -> int:
        """Pick the character position where the chunk after `chunk` starts.

        Candidate positions come in tiers (whole sentences, then clauses, then
        units). Within the first tier that has a valid candidate, the earliest
        one wins, which is the longest overlap. A position is valid when
        (a) the repeated text, `chunk.end - position`, fits in chunk_overlap
        and (b) `next_unit` still fits after it, so every chunk adds new text
        and packing always terminates. With no valid candidate the next chunk
        starts at `next_unit` with no overlap.
        """
        if self.chunk_overlap:
            for tier in _overlap_candidates(text, chunk, chunk_units):
                for position in tier:
                    repeated = chunk.end - position
                    fits_next = next_unit.end - position <= self.chunk_size
                    if repeated <= self.chunk_overlap and fits_next:
                        return position
        return next_unit.start


def _split_at(text: str, span: _Span, boundary: re.Pattern[str]) -> list[_Span]:
    """Split `span` at every match of `boundary`.

    `span` is already trimmed and every match is a whole whitespace run, so
    every piece is non-blank and trimmed too: no piece needs `_strip`.
    """
    pieces: list[_Span] = []
    cursor = span.start
    for match in boundary.finditer(text, span.start, span.end):
        pieces.append(_Span(cursor, match.start()))
        cursor = match.end()
    pieces.append(_Span(cursor, span.end))
    return pieces


def _overlap_candidates(
    text: str, chunk: _Span, chunk_units: list[_Span]
) -> tuple[list[int], list[int], list[int]]:
    """Positions inside `chunk` where an overlap may start, in preference tiers.

    1. Sentence starts: just after a paragraph break or a sentence end.
    2. Clause starts: just after ``;`` or ``:``.
    3. Unit starts: the only option for text without such punctuation.

    Each tier is sorted ascending and every position is strictly inside the
    chunk and on a non-whitespace character.
    """

    def starts_after(*boundaries: re.Pattern[str]) -> list[int]:
        return sorted(
            {
                match.end()
                for boundary in boundaries
                for match in boundary.finditer(text, chunk.start, chunk.end)
            }
        )

    sentence_starts = starts_after(_PARAGRAPH, _SENTENCE)
    clause_starts = starts_after(_CLAUSE)
    unit_starts = [unit.start for unit in chunk_units if unit.start > chunk.start]
    return sentence_starts, clause_starts, unit_starts


def _strip(text: str, span: _Span) -> _Span | None:
    """Shrink `span` to exclude surrounding whitespace; None if nothing is left."""
    start, end = span
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return _Span(start, end) if start < end else None
