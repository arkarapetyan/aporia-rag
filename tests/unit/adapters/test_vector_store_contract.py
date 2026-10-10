"""The VectorStore contract (test plan: 11.1-11.12).

Every test runs twice: against the in-memory fake used by the rest of the
suite, and against real Chroma (in memory, so still fast and hermetic).
This is what makes the fake trustworthy: if it ever behaved differently
from Chroma, these tests would fail for one of them. A future store (say,
pgvector or Qdrant) gets the same suite by adding one fixture parameter.
"""

from collections.abc import Iterator

import pytest
from chromadb.api import ClientAPI

from aporia_rag.adapters.chroma import ChromaVectorStore
from aporia_rag.domain.errors import VectorDimensionError
from aporia_rag.domain.models import EmbeddingSpace
from aporia_rag.domain.ports import IndexRecord, VectorStore
from support.builders import make_chunk, make_record
from support.fakes import InMemoryVectorStore

SPACE = EmbeddingSpace(provider="test", model="contract", dimensions=3, prompt_format="plain")


@pytest.fixture(params=["in-memory", "chroma"])
def store(request: pytest.FixtureRequest) -> Iterator[VectorStore]:
    if request.param == "in-memory":
        yield InMemoryVectorStore(SPACE)
    else:
        client: ClientAPI = request.getfixturevalue("chroma_client")
        yield ChromaVectorStore(client, SPACE)


def _ids(store: VectorStore, vector: list[float], k: int = 10, **filters: str) -> list[str]:
    return [result.chunk.id for result in store.query(vector, k=k, filters=filters or None)]


def test_empty_store(store: VectorStore) -> None:  # 11.1
    assert store.count() == 0
    assert store.fingerprints() == {}
    assert store.query([1.0, 0.0, 0.0], k=3) == []


def test_query_returns_the_k_nearest_best_first(store: VectorStore) -> None:  # 11.2
    store.upsert(
        [
            make_record("far#0", [0.0, 0.0, 1.0]),
            make_record("nearest#0", [1.0, 0.0, 0.0]),
            make_record("middle#0", [0.0, 1.0, 0.0]),
            make_record("near#0", [0.9, 0.1, 0.0]),
        ]
    )

    results = store.query([1.0, 0.05, 0.0], k=2)

    assert [result.chunk.id for result in results] == ["nearest#0", "near#0"]
    assert results[0].score >= results[1].score


def test_scores_are_cosine_similarities(store: VectorStore) -> None:  # 11.3
    store.upsert(
        [
            make_record("same#0", [1.0, 0.0, 0.0]),
            make_record("orthogonal#0", [0.0, 1.0, 0.0]),
            make_record("opposite#0", [-1.0, 0.0, 0.0]),
        ]
    )

    # The query is twice as long as `same`: cosine ignores length, L2 would not.
    scores = {result.chunk.id: result.score for result in store.query([2.0, 0.0, 0.0], k=3)}

    assert scores == pytest.approx(
        {"same#0": 1.0, "orthogonal#0": 0.0, "opposite#0": -1.0}, abs=1e-6
    )


def test_chunks_come_back_intact(store: VectorStore) -> None:  # 11.4
    translated = make_chunk(
        "republic/book-i#3",
        "I went down yesterday to the Piraeus.",
        author="Plato",
        title="The Republic",
        section="Book I",
        translator="Benjamin Jowett",
        index=3,
        start_char=120,
    )
    untranslated = make_chunk("hume/enquiry#0", "Custom is the great guide.", section=None)
    store.upsert(
        [
            IndexRecord(chunk=translated, vector=[1.0, 0.0, 0.0], fingerprint="a"),
            IndexRecord(chunk=untranslated, vector=[0.0, 1.0, 0.0], fingerprint="b"),
        ]
    )

    returned = {result.chunk.id: result.chunk for result in store.query([1.0, 1.0, 0.0], k=2)}

    assert returned == {translated.id: translated, untranslated.id: untranslated}


def test_upserting_an_existing_id_replaces_it(store: VectorStore) -> None:  # 11.5
    store.upsert([make_record("kant#0", [1.0, 0.0, 0.0], text="Old.", fingerprint="old")])

    store.upsert([make_record("kant#0", [0.0, 1.0, 0.0], text="New.", fingerprint="new")])

    assert store.count() == 1
    assert store.fingerprints() == {"kant#0": "new"}
    [result] = store.query([0.0, 1.0, 0.0], k=5)
    assert result.chunk.text == "New."
    assert result.score == pytest.approx(1.0, abs=1e-6)  # the vector was replaced too


def test_k_larger_than_the_store_returns_everything(store: VectorStore) -> None:  # 11.6
    store.upsert([make_record("a#0", [1.0, 0.0, 0.0]), make_record("b#0", [0.0, 1.0, 0.0])])

    assert sorted(_ids(store, [1.0, 0.0, 0.0], k=50)) == ["a#0", "b#0"]


def test_filters_restrict_results_and_combine_with_and(store: VectorStore) -> None:  # 11.7
    store.upsert(
        [
            make_record("kant-1#0", [1.0, 0.0, 0.0], author="Immanuel Kant", section="First"),
            make_record("kant-2#0", [0.9, 0.1, 0.0], author="Immanuel Kant", section="Second"),
            make_record("hobbes#0", [1.0, 0.0, 0.0], author="Thomas Hobbes", section="First"),
        ]
    )
    query = [1.0, 0.0, 0.0]

    assert sorted(_ids(store, query, author="Immanuel Kant")) == ["kant-1#0", "kant-2#0"]
    assert _ids(store, query, author="Immanuel Kant", section="Second") == ["kant-2#0"]
    assert _ids(store, query, author="Thomas Hobbes", section="Second") == []


def test_unknown_filter_field_is_an_error(store: VectorStore) -> None:  # 11.8
    store.upsert([make_record("kant#0", [1.0, 0.0, 0.0])])

    with pytest.raises(ValueError, match="autor"):
        store.query([1.0, 0.0, 0.0], k=1, filters={"autor": "Immanuel Kant"})


def test_fingerprints_map_every_chunk_id(store: VectorStore) -> None:  # 11.9
    store.upsert(
        [
            make_record("a#0", [1.0, 0.0, 0.0], fingerprint="fa"),
            make_record("b#0", [0.0, 1.0, 0.0], fingerprint="fb"),
            make_record("c#0", [0.0, 0.0, 1.0], fingerprint="fc"),
        ]
    )

    assert store.fingerprints() == {"a#0": "fa", "b#0": "fb", "c#0": "fc"}


def test_delete_removes_the_given_ids_and_ignores_unknown_ones(store: VectorStore) -> None:  # 11.10
    store.upsert([make_record("a#0", [1.0, 0.0, 0.0]), make_record("b#0", [0.0, 1.0, 0.0])])

    store.delete(["a#0", "never-stored#0"])
    store.delete([])

    assert store.fingerprints().keys() == {"b#0"}
    assert _ids(store, [1.0, 0.0, 0.0]) == ["b#0"]


def test_clear_empties_the_store_and_it_stays_usable(store: VectorStore) -> None:  # 11.11
    store.upsert([make_record("a#0", [1.0, 0.0, 0.0])])

    store.clear()

    assert store.count() == 0
    store.upsert([make_record("b#0", [0.0, 1.0, 0.0])])
    assert _ids(store, [0.0, 1.0, 0.0]) == ["b#0"]


def test_vectors_of_the_wrong_size_are_rejected(store: VectorStore) -> None:  # 11.12
    with pytest.raises(VectorDimensionError):
        store.upsert([make_record("a#0", [1.0, 0.0, 0.0]), make_record("b#0", [1.0, 0.0])])
    assert store.count() == 0  # nothing from the bad batch was stored

    store.upsert([make_record("a#0", [1.0, 0.0, 0.0])])
    with pytest.raises(VectorDimensionError):
        store.upsert([make_record("b#0", [1.0, 0.0])])
    with pytest.raises(VectorDimensionError):
        store.query([1.0, 0.0], k=1)
