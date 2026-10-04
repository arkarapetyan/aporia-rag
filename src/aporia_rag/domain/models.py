"""Core data shapes shared by every stage of the pipeline.

All models are frozen value objects: a Chunk produced by ingestion is the
same Chunk the retriever returns and the generator cites, and no stage can
mutate it behind another stage's back. `extra="forbid"` turns a misspelled
field into an immediate error instead of a silently dropped value.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _ValueObject(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SourceMetadata(_ValueObject):
    """Provenance of a text: what every answer must be able to cite."""

    author: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_id: str = Field(
        min_length=1, description="Stable upstream identifier, e.g. 'gutenberg:1497'."
    )
    section: str | None = Field(default=None, description="e.g. 'Book I' or 'Part II, Ch. 3'.")


class Document(_ValueObject):
    """A whole source text (typically one work, or one section of a work) before chunking."""

    id: str = Field(min_length=1)
    text: str
    metadata: SourceMetadata


class Chunk(_ValueObject):
    """A retrieval-sized span of a Document.

    `text` is always the exact slice `document.text[start_char:end_char]`,
    so a citation can point to the precise passage in the source.
    """

    id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    index: int = Field(ge=0, description="Position of this chunk within its document.")
    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int
    metadata: SourceMetadata

    @model_validator(mode="after")
    def _span_matches_text(self) -> Self:
        span = self.end_char - self.start_char
        if span != len(self.text):
            raise ValueError(
                f"span [{self.start_char}, {self.end_char}) covers {span} characters "
                f"but text has {len(self.text)}"
            )
        return self
