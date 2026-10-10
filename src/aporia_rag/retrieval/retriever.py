"""Dense retrieval: embed the question, return the nearest chunks.

Deliberately small. This is the baseline the corrective graph improves on
(grading, query rewriting, web fallback), and the evaluation will compare
against it, so it must stay a plain "top-k by cosine similarity".
"""

from aporia_rag.domain.models import RetrievedChunk
from aporia_rag.domain.ports import Embedder, MetadataFilter, VectorStore, check_same_space


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        check_same_space(embedder, store)
        self._embedder = embedder
        self._store = store

    def search(
        self, question: str, *, k: int = 5, filters: MetadataFilter | None = None
    ) -> list[RetrievedChunk]:
        """The `k` chunks most similar to the question, best first."""
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if not question.strip():
            raise ValueError("cannot search for a blank question")
        return self._store.query(self._embedder.embed_query(question), k=k, filters=filters)
