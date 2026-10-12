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
    translator: str | None = None

    @property
    def citation(self) -> str:
        """'Author, Title, Section' (or 'Author, Title'): how a passage is cited everywhere."""
        return ", ".join(part for part in (self.author, self.title, self.section) if part)


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


class Passage(_ValueObject):
    """A text to embed as a document, with an optional title.

    The title is kept separate because providers use it differently: some
    have a title slot (EmbeddingGemma, Gemini), others get it prepended to
    the text. Either way it lets a chunk be embedded *with* its context
    ("Kant, Groundwork, First Section") while the stored chunk stays the
    exact source text.
    """

    text: str = Field(min_length=1)
    title: str | None = Field(default=None, min_length=1)


class EmbeddingSpace(_ValueObject):
    """Everything that determines what an embedding vector means.

    Two vectors can only be compared if they come from the same space: the
    same provider and model, the same output size, and the same prompt
    format (a query embedded with an instruction is a different vector from
    the same query without one). Indexes record their space and refuse
    vectors from any other.
    """

    provider: str = Field(min_length=1, description="e.g. 'ollama', 'google', 'voyage'.")
    model: str = Field(min_length=1)
    dimensions: int | None = Field(default=None, gt=0, description="None: the model's own size.")
    prompt_format: str = Field(min_length=1, description="Name of the query/passage format.")

    @property
    def id(self) -> str:
        """A readable identifier, e.g. 'ollama/qwen3-embedding:0.6b/native/qwen-instruct'."""
        size = self.dimensions or "native"
        return f"{self.provider}/{self.model}/{size}/{self.prompt_format}"


class RetrievedChunk(_ValueObject):
    """A chunk returned by a search, with its cosine similarity to the query (-1 to 1)."""

    chunk: Chunk
    score: float
