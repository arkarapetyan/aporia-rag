"""Dense retrieval: embed the question, return the nearest chunks.

Deliberately small. This is the baseline the corrective graph improves on
(grading, query rewriting, web fallback), and the evaluation will compare
against it, so it must stay a plain "top-k by cosine similarity".
"""

from collections.abc import Iterable

from aporia_rag.domain.models import RetrievedChunk
from aporia_rag.domain.ports import (
    Embedder,
    MetadataFilter,
    Vector,
    VectorStore,
    check_same_space,
)


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        check_same_space(embedder, store)
        self._embedder = embedder
        self._store = store

    def search(
        self, question: str, *, k: int = 5, filters: MetadataFilter | None = None
    ) -> list[RetrievedChunk]:
        """The `k` chunks most similar to the question, best first."""
        _check_positive("k", k)
        return self._store.query(self._embed(question), k=k, filters=filters)

    def best_per_author(
        self, question: str, authors: Iterable[str], *, limit: int
    ) -> list[RetrievedChunk]:
        """Each author's best chunk, for the `limit` authors whose best chunk scores highest.

        Best first. The question is embedded once, then searched once per
        author, so an author with few passages on a topic still gets a turn
        instead of being crowded out by an author with many. An author with
        no chunks is left out.
        """
        _check_positive("limit", limit)
        vector = self._embed(question)
        best = [
            results[0]
            for author in authors
            if (results := self._store.query(vector, k=1, filters={"author": author}))
        ]
        return sorted(best, key=lambda result: result.score, reverse=True)[:limit]

    def _embed(self, question: str) -> Vector:
        if not question.strip():
            raise ValueError("cannot search for a blank question")
        return self._embedder.embed_query(question)


def _check_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
