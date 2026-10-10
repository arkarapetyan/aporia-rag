"""Real-model tests for the configured embedder (test plan: L.1-L.3).

Opt-in: `make test-llm`. They use the same settings as the application
(environment variables and `.env`), so they test whatever model you have
configured, and they make no assumption about which one it is: no vector
size or score is hard-coded. Run them after changing the embedding model.
"""

import math
from collections.abc import Iterator

import pytest

from aporia_rag.bootstrap import Container
from aporia_rag.config import load_settings
from aporia_rag.domain.models import Passage
from aporia_rag.domain.ports import Embedder, IndexRecord, Vector
from aporia_rag.indexing.indexer import citation_passage
from aporia_rag.retrieval.retriever import Retriever
from support.builders import make_chunk
from support.fakes import InMemoryVectorStore

pytestmark = pytest.mark.llm

# Abbott's Kant, Hobbes and Mill: three authors on three different subjects.
KANT = make_chunk(
    "kant/first-section#0",
    "Nothing can possibly be conceived in the world, or even out of it, which can be called "
    "good, without qualification, except a Good Will.",
    author="Immanuel Kant",
    title="Fundamental Principles of the Metaphysic of Morals",
    section="First Section",
)
HOBBES = make_chunk(
    "hobbes/chapter-xiii#0",
    "Hereby it is manifest, that during the time men live without a common Power to keep them "
    "all in awe, they are in that condition which is called Warre; and such a warre, as is of "
    "every man, against every man.",
    author="Thomas Hobbes",
    title="Leviathan",
    section="Chapter XIII",
)
MILL = make_chunk(
    "mill/chapter-ii#0",
    "The creed which accepts as the foundation of morals, Utility, or the Greatest Happiness "
    "Principle, holds that actions are right in proportion as they tend to promote happiness.",
    author="John Stuart Mill",
    title="Utilitarianism",
    section="Chapter II",
)


@pytest.fixture(scope="module")
def embedder() -> Iterator[Embedder]:
    with Container(load_settings()) as container:  # closes the connection afterwards
        yield container.embedder


def _cosine(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_vectors_have_one_size_for_every_call(embedder: Embedder) -> None:  # L.1
    passages = [citation_passage(chunk) for chunk in (KANT, HOBBES, MILL)]

    sizes = {len(vector) for vector in embedder.embed_documents(passages)}
    sizes.add(len(embedder.embed_query("What is a good will?")))

    assert len(sizes) == 1
    configured = embedder.space.dimensions
    assert configured is None or sizes == {configured}


def test_query_and_passage_prompts_differ(embedder: Embedder) -> None:  # L.2
    text = "What is a good will?"

    as_query = embedder.embed_query(text)
    [as_passage] = embedder.embed_documents([Passage(text=text)])

    assert _cosine(as_query, as_passage) < 0.9999


def test_question_about_the_good_will_finds_kant(embedder: Embedder) -> None:  # L.3
    store = InMemoryVectorStore(embedder.space)
    chunks = (HOBBES, MILL, KANT)  # Kant last, so insertion order cannot explain a pass
    vectors = embedder.embed_documents([citation_passage(chunk) for chunk in chunks])
    store.upsert(
        [
            IndexRecord(chunk=chunk, vector=vector, fingerprint="live")
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
    )

    [best, *_] = Retriever(embedder, store).search(
        "Is there anything that is good unconditionally, whatever its results?", k=3
    )

    assert best.chunk.id == KANT.id
