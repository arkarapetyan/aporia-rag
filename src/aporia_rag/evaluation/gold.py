"""The gold question set: questions written by hand, with the evidence retrieval should find.

Evidence is a section (a document id) plus a short verbatim quote from the
passage that answers the question. Matching the quote, rather than a chunk
id, keeps the gold set valid when chunking changes: whichever chunk holds
the quote is the right one.

A question may have no evidence at all (``evidence = []``): the corpus does
not answer it, so every retrieved passage is rightly "not relevant" and the
corrective loop must look elsewhere. Retrieval metrics leave such questions
out; the grader evaluation uses them.

The sets live next to this module, like the corpus manifest:
``gold_retrieval.toml`` (the default) and ``gold_loop.toml`` (questions for
the corrective loop). They are validated every time they are loaded.
"""

import tomllib
from importlib.resources import files
from pathlib import Path
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aporia_rag.domain.models import Chunk

_SLUG: Final = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
_NOT_BLANK: Final = r"\S"


class _Gold(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Evidence(_Gold):
    """Where the answer to a gold question is."""

    document: str = Field(
        pattern=_NOT_BLANK, description="Document (section) id, e.g. 'plato-republic/book-iv'."
    )
    quote: str = Field(pattern=_NOT_BLANK, description="Verbatim text of the answering passage.")

    def is_section_hit(self, chunk: Chunk) -> bool:
        """The chunk comes from the evidence's section."""
        return chunk.document_id == self.document

    def is_passage_hit(self, chunk: Chunk) -> bool:
        """The chunk comes from the evidence's section and contains the quote."""
        return self.is_section_hit(chunk) and _flatten(self.quote) in _flatten(chunk.text)


class GoldQuestion(_Gold):
    """A question and the evidence that answers it; finding any one piece counts."""

    id: str = Field(pattern=_SLUG)
    question: str = Field(pattern=_NOT_BLANK)
    evidence: list[Evidence] = Field(
        description="Every passage that answers the question; empty if the corpus does not."
    )

    @property
    def has_evidence(self) -> bool:
        """The corpus answers the question."""
        return bool(self.evidence)

    def is_section_hit(self, chunk: Chunk) -> bool:
        return any(evidence.is_section_hit(chunk) for evidence in self.evidence)

    def is_passage_hit(self, chunk: Chunk) -> bool:
        return any(evidence.is_passage_hit(chunk) for evidence in self.evidence)


class GoldSet(_Gold):
    questions: list[GoldQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_are_unique(self) -> Self:
        ids = [question.id for question in self.questions]
        duplicates = sorted({question_id for question_id in ids if ids.count(question_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate question id: {', '.join(duplicates)}")
        return self


def load_gold_set(path: Path | None = None) -> GoldSet:
    """Load and validate a gold set; defaults to the packaged ``gold_retrieval.toml``."""
    if path is None:
        raw = files("aporia_rag.evaluation").joinpath("gold_retrieval.toml").read_text("utf-8")
    else:
        raw = path.read_text(encoding="utf-8")
    return GoldSet.model_validate(tomllib.loads(raw))


def _flatten(text: str) -> str:
    """Collapse every whitespace run to one space, so line breaks never decide a match."""
    return " ".join(text.split())
