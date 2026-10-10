"""Tests for aporia_rag.indexing.indexer (test plan: Component 12).

The indexer is tested with in-memory fakes for the embedder and the store,
and with a one-chunk-per-paragraph chunker, so every test can say exactly
which chunks exist and which ones had to be embedded.
"""

from collections.abc import Sequence

import pytest

from aporia_rag.domain.errors import IndexMismatchError, ProviderError
from aporia_rag.domain.models import Chunk, Document, Passage, SourceMetadata
from aporia_rag.domain.ports import Vector
from aporia_rag.indexing.indexer import (
    Indexer,
    WorkProgress,
    fingerprint,
)
from support.fakes import HashingEmbedder, InMemoryVectorStore

KANT = SourceMetadata(
    author="Immanuel Kant",
    title="Groundwork",
    source_id="test:kant",
    section="First Section",
    translator="Thomas Kingsmill Abbott",
)
HOBBES = SourceMetadata(
    author="Thomas Hobbes", title="Leviathan", source_id="test:hobbes", section="Chapter XIII"
)


class ParagraphChunker:
    """A `Chunker` that makes one chunk per paragraph: easy to reason about."""

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        start = 0
        for index, paragraph in enumerate(document.text.split("\n\n")):
            chunks.append(
                Chunk(
                    id=f"{document.id}#{index}",
                    document_id=document.id,
                    index=index,
                    text=paragraph,
                    start_char=start,
                    end_char=start + len(paragraph),
                    metadata=document.metadata,
                )
            )
            start += len(paragraph) + 2
        return chunks


def _kant(*paragraphs: str) -> Document:
    text = "\n\n".join(
        paragraphs
        or (
            "Nothing can be called good without qualification, except a good will.",
            "A good will is good not because of what it performs.",
            "Even if it should wholly lack power to accomplish its purpose.",
        )
    )
    return Document(id="kant/first-section", text=text, metadata=KANT)


LEVIATHAN = Document(
    id="hobbes/chapter-xiii",
    text="And the life of man, solitary, poor, nasty, brutish, and short.",
    metadata=HOBBES,
)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def store(embedder: HashingEmbedder) -> InMemoryVectorStore:
    return InMemoryVectorStore(embedder.space)


@pytest.fixture
def indexer(embedder: HashingEmbedder, store: InMemoryVectorStore) -> Indexer:
    return Indexer(ParagraphChunker(), embedder, store)


def test_first_run_embeds_and_stores_every_chunk(
    indexer: Indexer, embedder: HashingEmbedder, store: InMemoryVectorStore
) -> None:  # 12.1
    report = indexer.run([_kant(), LEVIATHAN])

    assert (report.added, report.updated, report.unchanged, report.deleted) == (4, 0, 0, 0)
    assert len(embedder.embedded_passages) == 4
    assert store.count() == 4
    assert report.seconds >= 0


def test_second_run_embeds_nothing(indexer: Indexer, embedder: HashingEmbedder) -> None:  # 12.2
    indexer.run([_kant(), LEVIATHAN])
    embedded_before = len(embedder.embedded_passages)

    report = indexer.run([_kant(), LEVIATHAN])

    assert (report.added, report.updated, report.unchanged, report.deleted) == (0, 0, 4, 0)
    assert len(embedder.embedded_passages) == embedded_before


def test_only_changed_chunks_are_re_embedded(
    indexer: Indexer, embedder: HashingEmbedder, store: InMemoryVectorStore
) -> None:  # 12.3
    indexer.run([_kant(), LEVIATHAN])
    embedder.document_calls.clear()
    edited = _kant(
        "Nothing can be called good without qualification, except a good will.",
        "A good will is good only through its willing.",  # edited
        "Even if it should wholly lack power to accomplish its purpose.",
    )

    report = indexer.run([edited, LEVIATHAN])

    assert (report.added, report.updated, report.unchanged, report.deleted) == (0, 1, 3, 0)
    assert [passage.text for passage in embedder.embedded_passages] == [
        "A good will is good only through its willing."
    ]
    [result] = store.query(embedder.embed_query("only through its willing"), k=1)
    assert result.chunk.text == "A good will is good only through its willing."


def test_chunks_no_longer_produced_are_deleted(
    indexer: Indexer, store: InMemoryVectorStore
) -> None:  # 12.4
    indexer.run([_kant(), LEVIATHAN])
    shorter = _kant("Nothing can be called good without qualification, except a good will.")

    report = indexer.run([shorter])  # one Kant paragraph and all of Hobbes are gone

    assert (report.added, report.updated, report.unchanged, report.deleted) == (0, 0, 1, 3)
    assert store.fingerprints().keys() == {"kant/first-section#0"}


def test_passages_carry_their_citation_but_the_store_keeps_the_raw_text(
    indexer: Indexer, embedder: HashingEmbedder, store: InMemoryVectorStore
) -> None:  # 12.5
    indexer.run([LEVIATHAN])

    [passage] = embedder.embedded_passages
    assert passage == Passage(
        title="Thomas Hobbes, Leviathan, Chapter XIII",
        text="And the life of man, solitary, poor, nasty, brutish, and short.",
    )
    [result] = store.query(embedder.embed_query("life of man"), k=1)
    assert result.chunk.text == LEVIATHAN.text


def test_embedding_happens_in_groups_of_flush_size(
    embedder: HashingEmbedder, store: InMemoryVectorStore
) -> None:  # 12.6
    Indexer(ParagraphChunker(), embedder, store, flush_size=2).run([_kant()])

    assert [len(call) for call in embedder.document_calls] == [2, 1]


def test_flush_size_must_be_positive(
    embedder: HashingEmbedder, store: InMemoryVectorStore
) -> None:  # 12.6
    with pytest.raises(ValueError, match="flush_size"):
        Indexer(ParagraphChunker(), embedder, store, flush_size=0)


def test_progress_is_reported_once_per_work(indexer: Indexer) -> None:  # 12.7
    second_section = Document(
        id="kant/second-section",
        text="Everything in nature works according to laws.",
        metadata=KANT.model_copy(update={"section": "Second Section"}),
    )
    events: list[WorkProgress] = []

    indexer.run([_kant(), second_section, LEVIATHAN], on_progress=events.append)

    assert [(event.author, event.title, event.documents, event.chunks) for event in events] == [
        ("Immanuel Kant", "Groundwork", 2, 4),
        ("Thomas Hobbes", "Leviathan", 1, 1),
    ]


def test_embedder_and_store_from_different_spaces_are_refused() -> None:  # 12.8
    store = InMemoryVectorStore(HashingEmbedder(model="another-model").space)

    with pytest.raises(IndexMismatchError, match="another-model"):
        Indexer(ParagraphChunker(), HashingEmbedder(), store)


def test_fingerprint_changes_with_title_or_text_only() -> None:  # 12.9
    passage = Passage(title="Kant, Groundwork", text="A good will.")

    assert fingerprint(passage) == fingerprint(
        Passage(title="Kant, Groundwork", text="A good will.")
    )
    assert fingerprint(passage) != fingerprint(passage.model_copy(update={"text": "A good will!"}))
    assert fingerprint(passage) != fingerprint(passage.model_copy(update={"title": "Kant"}))
    # Moving characters between title and text is a change too.
    assert fingerprint(Passage(title="ab", text="c")) != fingerprint(Passage(title="a", text="bc"))


def test_two_chunks_with_the_same_id_are_an_error(indexer: Indexer) -> None:  # 12.10
    with pytest.raises(ValueError, match="kant/first-section#0"):
        indexer.run([_kant(), _kant()])


class FailingEmbedder(HashingEmbedder):
    """Embeds `succeed` batches, then fails as if the model server had crashed."""

    def __init__(self, succeed: int) -> None:
        super().__init__()
        self.remaining = succeed

    def embed_documents(self, passages: Sequence[Passage]) -> list[Vector]:
        if self.remaining == 0:
            raise ProviderError("simulated crash")
        self.remaining -= 1
        return super().embed_documents(passages)


def test_interrupted_run_keeps_its_progress(store: InMemoryVectorStore) -> None:  # 12.11
    failing = FailingEmbedder(succeed=1)
    with pytest.raises(ProviderError):
        Indexer(ParagraphChunker(), failing, store, flush_size=2).run([_kant(), LEVIATHAN])
    assert store.count() == 2  # the first flush was stored before the crash

    embedder = HashingEmbedder()
    report = Indexer(ParagraphChunker(), embedder, store, flush_size=2).run([_kant(), LEVIATHAN])

    assert (report.added, report.unchanged) == (2, 2)
    assert len(embedder.embedded_passages) == 2
