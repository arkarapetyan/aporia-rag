"""The shared part of every embedding adapter.

Each provider's API is different, but the work around the call is not:
split passages into batches the provider accepts, send queries and passages
with the right purpose, and check that what comes back is one vector per
input, all of the same size. `BatchingEmbedder` does that once (the
Template Method pattern); a provider adapter only implements `_embed_batch`.

A future adapter, sketched:

    class VoyageEmbedder(BatchingEmbedder):
        def _embed_batch(self, passages, purpose):
            input_type = "query" if purpose is EmbeddingPurpose.QUERY else "document"
            texts = [... title + text ...]
            return self._client.embed(texts, model=..., input_type=input_type).embeddings
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from itertools import batched

from aporia_rag.domain.errors import ProviderError, VectorDimensionError
from aporia_rag.domain.models import EmbeddingSpace, Passage
from aporia_rag.domain.ports import Vector, common_dimension


class EmbeddingPurpose(StrEnum):
    """What a text is for. Providers encode this as a prefix, an instruction or a parameter."""

    QUERY = "query"
    DOCUMENT = "document"


class BatchingEmbedder(ABC):
    """Base class for embedding adapters: batching and answer checking.

    Implements the `Embedder` port. A query is sent as a one-passage batch
    with the QUERY purpose, so subclasses have a single method to write.
    """

    def __init__(self, space: EmbeddingSpace, *, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self._space = space
        self._batch_size = batch_size
        # The vector size: configured, or learned from the first answer. Every
        # later answer must match it, across calls too, or the vectors could
        # not be compared.
        self._dimension = space.dimensions

    @property
    def space(self) -> EmbeddingSpace:
        return self._space

    def embed_documents(self, passages: Sequence[Passage]) -> list[Vector]:
        vectors: list[Vector] = []
        for batch in batched(passages, self._batch_size):
            vectors.extend(self._checked(batch, EmbeddingPurpose.DOCUMENT))
        return vectors

    def embed_query(self, text: str) -> Vector:
        [vector] = self._checked([Passage(text=text)], EmbeddingPurpose.QUERY)
        return vector

    @abstractmethod
    def _embed_batch(self, passages: Sequence[Passage], purpose: EmbeddingPurpose) -> list[Vector]:
        """Ask the provider for one vector per passage, in order.

        Translate the provider's own exceptions into ProviderError (or one of
        its subclasses); the result is checked by the caller.
        """

    def _checked(self, passages: Sequence[Passage], purpose: EmbeddingPurpose) -> list[Vector]:
        vectors = self._embed_batch(passages, purpose)
        if len(vectors) != len(passages):
            raise ProviderError(
                f"{self._space.provider} was sent {len(passages)} passages "
                f"but returned {len(vectors)} vector(s)"
            )
        try:
            self._dimension = common_dimension(vectors, self._dimension)
        except VectorDimensionError as error:
            raise ProviderError(
                f"{self._space.provider} returned vectors of an unexpected size: {error}"
            ) from error
        return vectors
