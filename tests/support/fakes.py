"""In-memory test doubles for the application's ports.

They implement the same interfaces as the real adapters (`Embedder`,
`VectorStore`, Ollama's HTTP API), so code under test cannot tell them
apart. `InMemoryVectorStore` is held to exactly the same behaviour as
`ChromaVectorStore` by the shared contract suite in
tests/unit/adapters/test_vector_store_contract.py.
"""

import hashlib
import json
import math
import re
from collections.abc import Callable, Collection, Sequence
from typing import Any, Final

import httpx

from aporia_rag.domain.models import EmbeddingSpace, Passage, RetrievedChunk
from aporia_rag.domain.ports import (
    IndexRecord,
    MetadataFilter,
    Vector,
    check_filters,
    common_dimension,
)

_TOKEN: Final = re.compile(r"[a-z]+")


def hash_vector(text: str, dimensions: int = 256) -> Vector:
    """A deterministic bag-of-words vector ("feature hashing").

    Every word is hashed into one of `dimensions` buckets and the counts are
    L2-normalised, so texts sharing words have a high cosine similarity.
    That makes retrieval tests meaningful without a real model. A stable
    hash (blake2b) is used because Python's built-in `hash()` of a string
    changes between processes.
    """
    counts = [0.0] * dimensions
    for token in _TOKEN.findall(text.lower()):
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        counts[int.from_bytes(digest, "big") % dimensions] += 1.0
    norm = math.sqrt(sum(count * count for count in counts))
    if norm == 0:  # no words at all: any fixed non-zero vector will do
        counts[0] = 1.0
        return counts
    return [count / norm for count in counts]


class HashingEmbedder:
    """An `Embedder` built on `hash_vector` that records every call."""

    def __init__(self, dimensions: int = 256, *, model: str = "hashing") -> None:
        self._space = EmbeddingSpace(
            provider="fake", model=model, dimensions=dimensions, prompt_format="plain"
        )
        self.document_calls: list[list[Passage]] = []
        self.query_calls: list[str] = []

    @property
    def space(self) -> EmbeddingSpace:
        return self._space

    @property
    def embedded_passages(self) -> list[Passage]:
        return [passage for call in self.document_calls for passage in call]

    def embed_documents(self, passages: Sequence[Passage]) -> list[Vector]:
        self.document_calls.append(list(passages))
        return [self._vector(f"{passage.title or ''} {passage.text}") for passage in passages]

    def embed_query(self, text: str) -> Vector:
        self.query_calls.append(text)
        return self._vector(text)

    def _vector(self, text: str) -> Vector:
        assert self._space.dimensions is not None
        return hash_vector(text, self._space.dimensions)


class InMemoryVectorStore:
    """A `VectorStore` kept in a dict, with exact (brute-force) cosine search."""

    def __init__(self, space: EmbeddingSpace) -> None:
        self._space = space
        self._records: dict[str, IndexRecord] = {}

    @property
    def space(self) -> EmbeddingSpace:
        return self._space

    def upsert(self, records: Sequence[IndexRecord]) -> None:
        common_dimension((record.vector for record in records), self._dimension())
        for record in records:
            self._records[record.chunk.id] = record

    def query(
        self, vector: Vector, *, k: int, filters: MetadataFilter | None = None
    ) -> list[RetrievedChunk]:
        check_filters(filters)
        if not self._records:
            return []
        common_dimension([vector], self._dimension())
        scored = [
            (cosine(vector, record.vector), record)
            for record in self._records.values()
            if _matches(record, filters)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [RetrievedChunk(chunk=record.chunk, score=score) for score, record in scored[:k]]

    def fingerprints(self) -> dict[str, str]:
        return {chunk_id: record.fingerprint for chunk_id, record in self._records.items()}

    def delete(self, chunk_ids: Collection[str]) -> None:
        for chunk_id in chunk_ids:
            self._records.pop(chunk_id, None)

    def clear(self) -> None:
        self._records.clear()

    def count(self) -> int:
        return len(self._records)

    def _dimension(self) -> int | None:
        first = next(iter(self._records.values()), None)
        return None if first is None else len(first.vector)


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity of two non-zero vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norms


def _matches(record: IndexRecord, filters: MetadataFilter | None) -> bool:
    metadata = record.chunk.metadata.model_dump()
    return all(metadata.get(field) == value for field, value in (filters or {}).items())


class RecordingTransport(httpx.MockTransport):
    """A MockTransport that counts how many clients using it were closed."""

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        super().__init__(handler)
        self.closes = 0

    @property
    def closed(self) -> bool:
        return self.closes > 0

    def close(self) -> None:
        self.closes += 1


class FakeOllamaServer:
    """An `httpx` request handler that answers like Ollama's `/api/embed`.

    Vectors are `hash_vector`s of the exact input strings, so tests can
    check both what was sent and what came back. Use `.transport` as the
    `transport=` of an `ollama.Client` or `httpx.Client`; `.transport.closed`
    tells whether that client was closed afterwards.
    """

    def __init__(
        self,
        *,
        models: Collection[str] = ("qwen3-embedding:0.6b",),
        dimensions: int = 64,
        status: int | None = None,
    ) -> None:
        self.models = set(models)
        self.dimensions = dimensions
        self.status = status
        self.requests: list[dict[str, Any]] = []
        self.transport = RecordingTransport(self)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = json.loads(request.content)
        self.requests.append(body)
        if self.status is not None:
            return httpx.Response(self.status, json={"error": "simulated server failure"})
        if request.url.path != "/api/embed":
            return httpx.Response(404, json={"error": "not found"})
        if body["model"] not in self.models:
            error = f'model "{body["model"]}" not found, try pulling it first'
            return httpx.Response(404, json={"error": error})
        inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
        vectors = [hash_vector(text, self.dimensions) for text in inputs]
        return httpx.Response(200, json={"model": body["model"], "embeddings": vectors})


def refusing_transport() -> httpx.MockTransport:
    """A transport that behaves as if nothing is listening (Ollama not running)."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return httpx.MockTransport(refuse)
