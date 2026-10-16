"""Real-model tests for the configured embedder (test plan: L.1-L.3).

Opt-in: `make test-llm`. They use the same settings as the application
(environment variables and `.env`), so they test whatever model you have
configured, and they make no assumption about which one it is: no vector
size or score is hard-coded. Run them after changing the embedding model.
"""

from collections.abc import Iterator

import pytest

from aporia_rag.bootstrap import Container
from aporia_rag.config import load_settings
from aporia_rag.domain.models import Passage
from aporia_rag.domain.ports import Embedder, IndexRecord
from aporia_rag.indexing.indexer import citation_passage
from aporia_rag.retrieval.retriever import Retriever
from support.fakes import InMemoryVectorStore, cosine
from support.passages import HOBBES, KANT, MILL

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def embedder() -> Iterator[Embedder]:
    with Container(load_settings()) as container:  # closes the connection afterwards
        yield container.embedder


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

    assert cosine(as_query, as_passage) < 0.9999


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
