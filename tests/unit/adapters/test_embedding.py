"""Tests for aporia_rag.adapters.embedding.BatchingEmbedder (test plan: 10.8-10.13).

`BatchingEmbedder` is the base class every provider adapter extends: it owns
batching and checks the provider's answer, so each adapter only implements
"embed this batch for this purpose". It is tested through a minimal subclass.
"""

from collections.abc import Callable, Sequence

import pytest

from aporia_rag.adapters.embedding import BatchingEmbedder, EmbeddingPurpose
from aporia_rag.domain.errors import ProviderError
from aporia_rag.domain.models import EmbeddingSpace, Passage
from aporia_rag.domain.ports import Vector

type Reply = Callable[[Sequence[Passage]], list[Vector]]


def _three_numbers(passages: Sequence[Passage]) -> list[Vector]:
    return [[float(len(passage.text)), 0.0, 1.0] for passage in passages]


class RecordingEmbedder(BatchingEmbedder):
    """Answers each batch with `reply` and records what it was asked."""

    def __init__(
        self, *, batch_size: int = 2, dimensions: int | None = 3, reply: Reply = _three_numbers
    ) -> None:
        space = EmbeddingSpace(
            provider="test", model="recording", dimensions=dimensions, prompt_format="plain"
        )
        super().__init__(space, batch_size=batch_size)
        self.calls: list[tuple[list[Passage], EmbeddingPurpose]] = []
        self.reply = reply

    def _embed_batch(self, passages: Sequence[Passage], purpose: EmbeddingPurpose) -> list[Vector]:
        self.calls.append((list(passages), purpose))
        return self.reply(passages)


def _passages(*texts: str) -> list[Passage]:
    return [Passage(text=text) for text in texts]


def test_passages_are_sent_in_batches_and_order_is_kept() -> None:  # 10.8
    embedder = RecordingEmbedder(batch_size=2)

    vectors = embedder.embed_documents(_passages("a", "bb", "ccc", "dddd", "eeeee"))

    assert [len(batch) for batch, _ in embedder.calls] == [2, 2, 1]
    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_queries_and_passages_are_sent_with_their_purpose() -> None:  # 10.9
    embedder = RecordingEmbedder()

    embedder.embed_documents(_passages("A passage."))
    vector = embedder.embed_query("A question?")

    assert [purpose for _, purpose in embedder.calls] == [
        EmbeddingPurpose.DOCUMENT,
        EmbeddingPurpose.QUERY,
    ]
    assert embedder.calls[1][0] == [Passage(text="A question?")]
    assert vector == [11.0, 0.0, 1.0]  # len("A question?")


def test_no_passages_means_no_provider_call() -> None:  # 10.10
    embedder = RecordingEmbedder()

    assert embedder.embed_documents([]) == []
    assert embedder.calls == []


@pytest.mark.parametrize("question", ["", "   \n"])
def test_blank_query_is_rejected(question: str) -> None:  # 10.10
    embedder = RecordingEmbedder()

    with pytest.raises(ValueError, match="blank"):
        embedder.embed_query(question)
    assert embedder.calls == []


def test_wrong_number_of_vectors_is_a_provider_error() -> None:  # 10.11
    embedder = RecordingEmbedder(reply=lambda passages: _three_numbers(passages)[:-1])

    with pytest.raises(ProviderError, match=r"2 passages.*1 vector"):
        embedder.embed_documents(_passages("a", "b"))


def test_vectors_of_different_sizes_are_a_provider_error() -> None:  # 10.12
    embedder = RecordingEmbedder(dimensions=None, reply=lambda _: [[1.0, 0.0], [1.0, 0.0, 0.0]])

    with pytest.raises(ProviderError, match="size"):
        embedder.embed_documents(_passages("a", "b"))


def test_vectors_not_of_the_configured_size_are_a_provider_error() -> None:  # 10.12
    embedder = RecordingEmbedder(dimensions=3, reply=lambda passages: [[1.0, 0.0]] * len(passages))

    with pytest.raises(ProviderError, match="size"):
        embedder.embed_query("q")


def test_vector_size_must_stay_the_same_across_calls() -> None:  # 10.12
    sizes = iter([3, 4])
    embedder = RecordingEmbedder(
        dimensions=None, reply=lambda passages: [[1.0] * next(sizes) for _ in passages]
    )
    embedder.embed_query("first")

    with pytest.raises(ProviderError, match="size"):
        embedder.embed_query("second")


def test_batch_size_must_be_positive() -> None:  # 10.13
    with pytest.raises(ValueError, match="batch_size"):
        RecordingEmbedder(batch_size=0)
