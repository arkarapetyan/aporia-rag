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
* **Whole-unit overlap.** Consecutive chunks repeat as many whole trailing
  units as fit in ``chunk_overlap`` characters, so an argument straddling
  a boundary survives intact in at least one chunk. A unit is never
  repeated partially; a half-sentence of overlap carries no meaning.

Sizes are measured in characters, which is a proxy for embedding-model
tokens (roughly 4 characters per token for English prose).
"""

import re
from dataclasses import dataclass
from typing import Final, NamedTuple

from aporia_rag.domain.models import Chunk, Document

_PARAGRAPH: Final = re.compile(r"\n\s*\n")
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
            for index, span in enumerate(self._pack(units))
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

    def _pack(self, units: list[_Span]) -> list[_Span]:
        """Greedily merge consecutive units into chunks of at most chunk_size.

        `units` is never empty: `chunk` returns early for blank documents.
        """
        chunks: list[_Span] = []
        first = 0
        while True:
            end = first + 1  # exclusive
            while end < len(units) and units[end].end - units[first].start <= self.chunk_size:
                end += 1
            chunks.append(_Span(units[first].start, units[end - 1].end))
            if end == len(units):
                return chunks
            first = self._next_first_unit(units, first, end)

    def _next_first_unit(self, units: list[_Span], first: int, end: int) -> int:
        """Pick where the next chunk starts.

        The earliest unit after `first` such that (a) the repeated tail of the
        current chunk fits in chunk_overlap and (b) the next *new* unit,
        `units[end]`, still fits in the chunk. Falling back to `end` (no
        overlap) guarantees every chunk adds new text, so packing terminates.
        """
        previous_end = units[end - 1].end
        for candidate in range(first + 1, end):
            start = units[candidate].start
            repeated = previous_end - start
            if repeated <= self.chunk_overlap and units[end].end - start <= self.chunk_size:
                return candidate
        return end


def _split_at(text: str, span: _Span, boundary: re.Pattern[str]) -> list[_Span]:
    """Split `span` at every match of `boundary`, trimming each piece.

    With the current boundaries a piece is never blank (each pattern consumes
    a maximal whitespace run and `span` is already trimmed), so the `None`
    checks below are a guard for future boundary patterns, not live branches.
    """
    pieces: list[_Span] = []
    cursor = span.start
    for match in boundary.finditer(text, span.start, span.end):
        if (piece := _strip(text, _Span(cursor, match.start()))) is not None:  # pragma: no branch
            pieces.append(piece)
        cursor = match.end()
    if (last := _strip(text, _Span(cursor, span.end))) is not None:  # pragma: no branch
        pieces.append(last)
    return pieces


def _strip(text: str, span: _Span) -> _Span | None:
    """Shrink `span` to exclude surrounding whitespace; None if nothing is left."""
    start, end = span
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return _Span(start, end) if start < end else None
