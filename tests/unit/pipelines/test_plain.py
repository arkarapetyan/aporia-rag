"""Tests for aporia_rag.pipelines.plain, retrieve then answer (test plan: 30.8)."""

from collections.abc import Iterator

import pytest

from aporia_rag.domain.ports import IndexRecord
from aporia_rag.generation.generator import ANSWER_PROMPT, Generator
from aporia_rag.pipelines.plain import PlainRAG
from aporia_rag.retrieval.retriever import Retriever
from aporia_rag.retrieval.scope import Scope
from support.builders import make_chunk
from support.fakes import HashingEmbedder, InMemoryVectorStore, ScriptedChatModel, hash_vector

KANT = make_chunk(
    "kant/first-section#0", "Nothing can be called good without qualification, except a good will."
)
HOBBES = make_chunk(
    "hobbes/chapter-xiii#0",
    "And the life of man, solitary, poore, nasty, brutish, and short.",
    author="Thomas Hobbes",
    title="Leviathan",
    section="Chapter XIII",
)
AUTHORS = ("Immanuel Kant", "Thomas Hobbes")


def _plain(reply: str) -> tuple[PlainRAG, ScriptedChatModel]:
    embedder = HashingEmbedder()
    store = InMemoryVectorStore(embedder.space)
    store.upsert(
        [
            IndexRecord(chunk=chunk, vector=hash_vector(chunk.text), fingerprint="f")
            for chunk in (KANT, HOBBES)
        ]
    )
    model = ScriptedChatModel(reply=reply)
    return PlainRAG(Retriever(embedder, store), Generator(model)), model


def test_plain_rag_answers_from_every_passage_found_and_checks_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # 30.8
    ticks: Iterator[float] = iter([10.0, 10.5, 16.0])  # start, retrieved, answered
    monkeypatch.setattr("aporia_rag.pipelines.plain.perf_counter", lambda: next(ticks))
    plain, model = _plain("A good will alone is good [3].")  # a citation with no passage

    outcome = plain.answer(
        "What can be called good without qualification?", Scope(), k=5, authors=AUTHORS
    )

    [conversation] = model.conversations  # one call: the answer, no verdict
    assert conversation[0].content == ANSWER_PROMPT
    assert outcome.answer is not None
    assert [source.chunk.id for source in outcome.answer.sources] == [KANT.id, HOBBES.id]
    assert outcome.relevant == outcome.answer.sources
    assert [(problem.kind, problem.detail) for problem in outcome.problems] == [("citation", "[3]")]
    assert outcome.draft_problems == ()  # never written again
    assert outcome.route == "plain"
    assert outcome.seconds == {"retrieval": 0.5, "generation": 5.5}
