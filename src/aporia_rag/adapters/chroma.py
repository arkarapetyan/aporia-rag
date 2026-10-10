"""A VectorStore on Chroma, embedded in the process and persisted to disk.

Design points:

- **One collection per embedding space.** The collection name is derived
  from the space, so switching the embedding model builds a new collection
  next to the old one instead of mixing vectors (and both can be compared
  in an evaluation). The space is also recorded in the collection metadata
  and checked on open: a name collision can never mix spaces silently.
- **Cosine distance.** Chroma defaults to squared L2; retrieval models are
  trained for cosine similarity. Chroma returns a distance (1 - cosine),
  turned back into a similarity here so every store reports the same score.
- **We bring the vectors.** Collections have no embedding function: Chroma
  would otherwise download its own default model. Embedding is the
  Embedder's job.
- **Chroma stores flat metadata** (str, int, float, bool; no None), so a
  chunk is flattened on the way in and rebuilt, validated, on the way out.
"""

import hashlib
import re
from collections.abc import Collection, Mapping, Sequence
from itertools import batched
from pathlib import Path
from typing import Any, Final, Self, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection as ChromaCollection
from chromadb.api.types import Metadata, PyEmbedding, Where
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from aporia_rag.domain.errors import IndexMismatchError
from aporia_rag.domain.models import Chunk, EmbeddingSpace, RetrievedChunk
from aporia_rag.domain.ports import (
    IndexRecord,
    MetadataFilter,
    Vector,
    check_filters,
    common_dimension,
)

_SPACE_KEY: Final = "space_id"
_MAX_NAME_SLUG: Final = 54  # + "-" + 8 hex digits = 63, the most Chroma has ever allowed


def client_settings() -> ChromaSettings:
    """Settings for the clients this module creates: no usage telemetry sent anywhere."""
    return ChromaSettings(anonymized_telemetry=False)


def collection_name(space: EmbeddingSpace) -> str:
    """A valid Chroma collection name, readable and unique to the space.

    Chroma allows 3-63 characters of [a-zA-Z0-9._-], starting and ending
    with a letter or digit. Model names allow far more (`:`, `/`, any
    Unicode), so the readable part is a slug and uniqueness comes from a
    hash of the full space.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", space.id.lower()).strip("-")
    slug = slug[:_MAX_NAME_SLUG].rstrip("-") or "space"
    digest = hashlib.sha256(space.model_dump_json().encode()).hexdigest()[:8]
    return f"{slug}-{digest}"


class ChromaVectorStore:
    """A `VectorStore` backed by one Chroma collection."""

    def __init__(
        self, client: ClientAPI, space: EmbeddingSpace, *, batch_size: int | None = None
    ) -> None:
        self._client = client
        self._space = space
        self._name = collection_name(space)
        # Chroma rejects writes larger than this in one call.
        self._batch_size = batch_size or client.get_max_batch_size()
        self._collection = self._open()
        self._dimension: int | None = None  # learned from the stored vectors when needed

    @classmethod
    def persistent(cls, path: Path, space: EmbeddingSpace) -> Self:
        """A store saved under `path` (created if missing)."""
        path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(path), settings=client_settings())
        return cls(client, space)

    @property
    def space(self) -> EmbeddingSpace:
        return self._space

    def upsert(self, records: Sequence[IndexRecord]) -> None:
        # Check every vector before writing any, so a bad batch stores nothing.
        dimension = self._known_dimension()
        self._dimension = common_dimension((record.vector for record in records), dimension)
        for batch in batched(records, self._batch_size):
            embeddings: list[PyEmbedding] = [record.vector for record in batch]
            self._collection.upsert(
                ids=[record.chunk.id for record in batch],
                embeddings=embeddings,
                documents=[record.chunk.text for record in batch],
                metadatas=[_flatten(record) for record in batch],
            )

    def query(
        self, vector: Vector, *, k: int, filters: MetadataFilter | None = None
    ) -> list[RetrievedChunk]:
        check_filters(filters)
        stored = self.count()
        if stored == 0:
            return []
        common_dimension([vector], self._known_dimension())
        query_embeddings: list[PyEmbedding] = [vector]
        result = self._collection.query(
            query_embeddings=query_embeddings,
            n_results=min(k, stored),  # asking for more than exist makes Chroma warn
            where=_where(filters),
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = (result["documents"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]
        return [
            RetrievedChunk(chunk=_rebuild(chunk_id, text, metadata), score=1.0 - distance)
            for chunk_id, text, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]

    def fingerprints(self) -> dict[str, str]:
        result = self._collection.get(include=["metadatas"])
        metadatas = result["metadatas"] or []
        return {
            chunk_id: str(metadata["fingerprint"])
            for chunk_id, metadata in zip(result["ids"], metadatas, strict=True)
        }

    def delete(self, chunk_ids: Collection[str]) -> None:
        # `batched` yields nothing for no ids, and Chroma must not get an
        # empty id list: it rejects it.
        for batch in batched(chunk_ids, self._batch_size):
            self._collection.delete(ids=list(batch))

    def clear(self) -> None:
        self._client.delete_collection(self._name)
        self._collection = self._create()
        self._dimension = None

    def count(self) -> int:
        return self._collection.count()

    def _open(self) -> ChromaCollection:
        try:
            collection = self._client.get_collection(self._name, embedding_function=None)
        except NotFoundError:
            return self._create()
        recorded = (collection.metadata or {}).get(_SPACE_KEY)
        if recorded != self._space.id:
            raise IndexMismatchError(
                f"Chroma collection {self._name!r} holds vectors from {recorded!r}, "
                f"not {self._space.id!r}"
            )
        return collection

    def _create(self) -> ChromaCollection:
        return self._client.create_collection(
            self._name,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={_SPACE_KEY: self._space.id},
            embedding_function=None,
        )

    def _known_dimension(self) -> int | None:
        if self._dimension is None:
            sample = self._collection.get(limit=1, include=["embeddings"])["embeddings"]
            if sample is not None and len(sample) > 0:
                self._dimension = len(sample[0])
        return self._dimension


def _flatten(record: IndexRecord) -> Metadata:
    chunk = record.chunk
    fields: dict[str, Any] = {
        **chunk.metadata.model_dump(),
        "document_id": chunk.document_id,
        "index": chunk.index,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "fingerprint": record.fingerprint,
    }
    return {key: value for key, value in fields.items() if value is not None}


def _rebuild(chunk_id: str, text: str, metadata: Mapping[str, Any]) -> Chunk:
    source_fields = ("author", "title", "source_id", "section", "translator")
    return Chunk.model_validate(
        {
            "id": chunk_id,
            "text": text,
            "document_id": metadata["document_id"],
            "index": metadata["index"],
            "start_char": metadata["start_char"],
            "end_char": metadata["end_char"],
            "metadata": {field: metadata.get(field) for field in source_fields},
        }
    )


def _where(filters: MetadataFilter | None) -> Where | None:
    conditions = [{field: value} for field, value in (filters or {}).items()]
    if not conditions:
        return None
    if len(conditions) == 1:
        return cast(Where, conditions[0])
    return cast(Where, {"$and": conditions})
