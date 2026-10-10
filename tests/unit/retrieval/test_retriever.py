"""Tests for aporia_rag.retrieval.retriever (test plan: Component 13)."""

import pytest

from aporia_rag.domain.errors import IndexMismatchError
from aporia_rag.domain.ports import IndexRecord
from aporia_rag.retrieval.retriever import Retriever
from support.builders import make_chunk
from support.fakes import HashingEmbedder, InMemoryVectorStore, hash_vector

KANT = make_chunk(
    "kant/first-section#0",
    "Nothing can be called good without qualification, except a good will. "
    "A good will is good not because of what it performs.",
)
HOBBES = make_chunk(
    "hobbes/chapter-xiii#0",
    "And the life of man, solitary, poor, nasty, brutish, and short.",
    author="Thomas Hobbes",
    title="Leviathan",
    section="Chapter XIII",
)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def retriever(embedder: HashingEmbedder) -> Retriever:
    # The store is filled directly, so the embedder sees only the retriever's calls.
    store = InMemoryVectorStore(embedder.space)
    store.upsert(
        [
            IndexRecord(chunk=chunk, vector=hash_vector(chunk.text), fingerprint="f")
            for chunk in (KANT, HOBBES)
        ]
    )
    return Retriever(embedder, store)


def test_question_is_embedded_as_a_query(
    retriever: Retriever, embedder: HashingEmbedder
) -> None:  # 13.1
    retriever.search("What makes a will good?")

    assert embedder.query_calls == ["What makes a will good?"]
    assert embedder.document_calls == []


def test_returns_the_k_best_chunks_best_first(retriever: Retriever) -> None:  # 13.2
    results = retriever.search("Is a good will good without qualification?", k=2)

    assert [result.chunk for result in results] == [KANT, HOBBES]
    assert results[0].score > results[1].score

    assert [result.chunk for result in retriever.search("good will", k=1)] == [KANT]


def test_filters_are_passed_to_the_store(retriever: Retriever) -> None:  # 13.3
    results = retriever.search("good will", k=5, filters={"author": "Thomas Hobbes"})

    assert [result.chunk for result in results] == [HOBBES]


def test_embedder_and_store_from_different_spaces_are_refused(
    embedder: HashingEmbedder,
) -> None:  # 13.4
    store = InMemoryVectorStore(HashingEmbedder(dimensions=128).space)

    with pytest.raises(IndexMismatchError):
        Retriever(embedder, store)


@pytest.mark.parametrize("k", [0, -1])
def test_k_must_be_positive(retriever: Retriever, k: int) -> None:  # 13.5
    with pytest.raises(ValueError, match="k"):
        retriever.search("good will", k=k)


def test_blank_question_is_rejected(
    retriever: Retriever, embedder: HashingEmbedder
) -> None:  # 13.5
    with pytest.raises(ValueError, match="blank"):
        retriever.search("   ")
    assert embedder.query_calls == []
