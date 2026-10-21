"""Ports: the interfaces the application needs from the outside world.

This is the "ports" half of ports and adapters (hexagonal architecture).
The indexer, the retriever, the generator and the LangGraph nodes depend
only on these Protocols; `adapters/` holds the implementations (Ollama,
Chroma, ddgs, and later Google, Voyage, Anthropic...). A Protocol is satisfied structurally:
any class with the right methods qualifies, without inheriting from it, so
an adapter needs no import from here beyond the data types it exchanges,
and the in-memory test fakes are plain classes.

The rules that go with these interfaces live here too, written once
instead of once per implementation or caller: `check_filters` and
`common_dimension` for every VectorStore, `check_same_space` for everything
that brings an Embedder and a VectorStore together.
"""

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from pydantic import BaseModel

from aporia_rag.domain.errors import IndexMismatchError, VectorDimensionError
from aporia_rag.domain.models import (
    ChatMessage,
    Chunk,
    Document,
    EmbeddingSpace,
    Passage,
    RetrievedChunk,
    SourceMetadata,
)

type Vector = list[float]
type MetadataFilter = Mapping[str, str]

# Every provenance field is stored with a chunk, so every one can be filtered on.
FILTERABLE_FIELDS: Final = frozenset(SourceMetadata.model_fields)


class Chunker(Protocol):
    """Splits a document into retrieval-sized chunks."""

    def chunk(self, document: Document) -> list[Chunk]: ...


class Embedder(Protocol):
    """Turns text into vectors, keeping queries and passages apart.

    Retrieval models are trained asymmetrically: a short question and the
    passage that answers it are embedded differently (an instruction, a
    prefix, a `task_type` or `input_type` parameter, depending on the
    provider). The interface states the *intent*; each adapter maps it to
    its provider's mechanism.
    """

    @property
    def space(self) -> EmbeddingSpace:
        """What the vectors mean; vectors from different spaces are never compared."""
        ...

    def embed_documents(self, passages: Sequence[Passage]) -> list[Vector]:
        """One vector per passage, in order."""
        ...

    def embed_query(self, text: str) -> Vector:
        """The vector of a search query."""
        ...


class ChatModel(Protocol):
    """A language model that replies to a conversation.

    Providers take the system prompt differently: as a message with the
    "system" role (Ollama, OpenAI), as a separate `system` parameter
    (Anthropic) or as a `system_instruction` (Gemini). The interface states
    the role; each adapter maps it to its provider's mechanism.

    Every call states how long its reply may be (`max_tokens`; Ollama's
    `num_predict`, the APIs' `max_tokens`). A reply cut at that length is a
    `ProviderError`, so a reply that does not stop fails in seconds instead
    of running until the time-out, and a cut reply is never shown as whole.
    """

    def chat(self, messages: Sequence[ChatMessage], *, max_tokens: int) -> str:
        """The model's reply: never empty and never cut (either is a `ProviderError`)."""
        ...

    def chat_structured[M: BaseModel](
        self, messages: Sequence[ChatMessage], schema: type[M], *, max_tokens: int
    ) -> M:
        """The model's reply as an instance of `schema`, for decisions such as verdicts.

        Asked as repeatably as the provider allows (temperature 0): the same
        messages should give the same decision. A reply that does not fit
        the schema is a `ProviderError`.
        """
        ...


@dataclass(frozen=True, slots=True)
class IndexRecord:
    """A chunk ready to store: its vector, and a fingerprint of what was embedded.

    The fingerprint lets the indexer skip chunks that have not changed since
    the last run, without re-embedding them to find out.
    """

    chunk: Chunk
    vector: Vector
    fingerprint: str


class VectorStore(Protocol):
    """Stores chunks with their vectors and finds the nearest ones by cosine similarity."""

    @property
    def space(self) -> EmbeddingSpace:
        """The embedding space of every vector in this store."""
        ...

    def upsert(self, records: Sequence[IndexRecord]) -> None:
        """Insert, or replace by chunk id."""
        ...

    def query(
        self, vector: Vector, *, k: int, filters: MetadataFilter | None = None
    ) -> list[RetrievedChunk]:
        """The `k` most similar chunks, best first, optionally restricted by metadata.

        Filter keys must be in FILTERABLE_FIELDS; several keys combine with AND.
        """
        ...

    def fingerprints(self) -> dict[str, str]:
        """Chunk id -> fingerprint, for every stored chunk."""
        ...

    def delete(self, chunk_ids: Collection[str]) -> None:
        """Remove these chunks; unknown ids are ignored."""
        ...

    def clear(self) -> None:
        """Remove everything."""
        ...

    def count(self) -> int: ...


def check_filters(filters: MetadataFilter | None) -> None:
    """Reject filter keys that are not stored metadata: a typo must not silently match nothing."""
    unknown = sorted(set(filters or {}) - FILTERABLE_FIELDS)
    if unknown:
        raise ValueError(
            f"unknown filter field(s) {', '.join(unknown)}; "
            f"filterable fields: {', '.join(sorted(FILTERABLE_FIELDS))}"
        )


def common_dimension(vectors: Iterable[Vector], expected: int | None = None) -> int | None:
    """The size shared by all `vectors` and by `expected`, if given.

    Returns `expected` (None if not given) when there are no vectors. Raises
    VectorDimensionError if any two sizes differ or a vector is empty.
    """
    dimension = expected
    for vector in vectors:
        size = len(vector)
        if size == 0:
            raise VectorDimensionError("empty vector")
        if dimension is None:
            dimension = size
        elif size != dimension:
            raise VectorDimensionError(f"vector of size {size} where {dimension} was expected")
    return dimension


def check_same_space(embedder: Embedder, store: VectorStore) -> None:
    """Refuse to bring together vectors from two embedding spaces.

    Their similarities would be meaningless, and nothing would visibly fail:
    retrieval would just quietly return the wrong passages.
    """
    if embedder.space != store.space:
        raise IndexMismatchError(
            f"the embedder makes {embedder.space.id} vectors, "
            f"but the store holds {store.space.id} vectors"
        )


class WebSearch(Protocol):
    """Finds pages on the web: the corrective graph's last resort, when the corpus has no answer.

    Pages come back as chunks made with `models.web_passage`, so the grader
    and the generator take them like corpus passages.
    """

    def search(self, query: str, *, limit: int) -> list[Chunk]:
        """At most `limit` pages for `query`, best first; none found is an empty list.

        A search that cannot be made (no network, a rate limit) is a `ProviderError`.
        """
        ...
