"""Tests for aporia_rag.graph.corrective, the corrective loop (test plan: 25.1-25.7).

The graph runs the real retriever, grader, generator and rewriter over
in-memory fakes: the hashing embedder ranks passages by the words they share
with the query, and one `ScriptedChatModel` plays every model role. `Script`
decides its replies by the kind of request, so each test states which
passages the grader keeps and what the rewriter writes, and the recorded
conversations show every model call the graph made.
"""

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import pytest

from aporia_rag.domain.models import Chunk, RetrievedChunk
from aporia_rag.domain.ports import IndexRecord
from aporia_rag.generation.generator import ANSWER_PROMPT, SURVEY_PROMPT, Generator
from aporia_rag.grading.relevance import SYSTEM_PROMPT as GRADING_PROMPT
from aporia_rag.grading.relevance import RelevanceGrader
from aporia_rag.graph.corrective import CorrectiveRAG, Outcome, Scope
from aporia_rag.retrieval.retriever import Retriever
from aporia_rag.retrieval.rewriter import SUBJECT_PROMPT, QueryRewriter
from support.builders import make_chunk
from support.fakes import HashingEmbedder, InMemoryVectorStore, ScriptedChatModel, hash_vector

KANT = make_chunk(
    "kant/first-section#0", "Nothing can be called good without qualification, except a good will."
)
LAW = make_chunk(
    "kant/second-section#0",
    "Act only on that maxim whereby thou canst will that it should become a universal law.",
    section="Second Section",
)
HOBBES = make_chunk(
    "hobbes/chapter-xiii#0",
    "And the life of man, solitary, poore, nasty, brutish, and short.",
    author="Thomas Hobbes",
    title="Leviathan",
    section="Chapter XIII",
)
MILL = make_chunk(
    "mill/chapter-ii#0",
    "Actions are right in proportion as they tend to promote happiness.",
    author="John Stuart Mill",
    title="Utilitarianism",
    section="Chapter II",
)
AUTHORS = ("Immanuel Kant", "John Stuart Mill", "Thomas Hobbes")
GOOD_WILL = "What can be called good without qualification?"  # finds KANT first
REPLY = "The scripted answer [1]."


@dataclass
class Script:
    """The scripted model's replies: which passages it keeps, and what it rewrites."""

    keep: Callable[[Chunk], bool] = lambda chunk: True
    query: str = "nasty brutish short life"  # finds HOBBES first
    subject: str = "the good"

    def reply(self, prompt: str) -> str:
        if prompt.startswith("Passage from "):
            passage = prompt.split(":\n", 1)[1].rsplit("\n\nQuestion: ", 1)[0]
            [chunk] = [chunk for chunk in (KANT, LAW, HOBBES, MILL) if chunk.text == passage]
            return f'{{"relevant": {str(self.keep(chunk)).lower()}}}'
        if prompt.startswith("[1] "):  # a survey: one paragraph per numbered passage
            count = len(re.findall(r"^\[\d+\] ", prompt, re.MULTILINE))
            paragraphs = ", ".join(f'"Paragraph {n}."' for n in range(1, count + 1))
            return f'{{"paragraphs": [{paragraphs}]}}'
        if "\nAuthors compared: " in prompt:
            return f'{{"subject": "{self.subject}"}}'
        return f'{{"query": "{self.query}"}}'


@dataclass
class Setup:
    graph: CorrectiveRAG
    model: ScriptedChatModel
    embedder: HashingEmbedder

    def answer(self, question: str, scope: Scope | None = None, *, k: int = 1) -> Outcome:
        return self.graph.answer(question, scope or Scope(), k=k, authors=AUTHORS)

    def calls(self) -> list[str]:
        """The kind of every model call so far, in order."""
        kinds = {
            GRADING_PROMPT: "verdict",
            ANSWER_PROMPT: "answer",
            SURVEY_PROMPT: "survey",
            SUBJECT_PROMPT: "subject",
        }
        return [kinds.get(c[0].content, "rewrite") for c in self.model.conversations]

    def verdicts(self) -> list[tuple[str, str]]:
        """(passage, question) of every verdict asked for, in order."""
        pairs = []
        for conversation in self.model.conversations:
            if conversation[0].content == GRADING_PROMPT:
                passage, question = (
                    conversation[1].content.split(":\n", 1)[1].rsplit("\n\nQuestion: ", 1)
                )
                [chunk] = [c for c in (KANT, LAW, HOBBES, MILL) if c.text == passage]
                pairs.append((chunk.id, question))
        return pairs


def _setup(script: Script) -> Setup:
    embedder = HashingEmbedder()
    store = InMemoryVectorStore(embedder.space)
    store.upsert(
        [
            IndexRecord(chunk=chunk, vector=hash_vector(chunk.text), fingerprint="f")
            for chunk in (KANT, LAW, HOBBES, MILL)
        ]
    )
    model = ScriptedChatModel(reply=REPLY, structured_reply=script.reply)
    graph = CorrectiveRAG(
        Retriever(embedder, store), RelevanceGrader(model), Generator(model), QueryRewriter(model)
    )
    return Setup(graph, model, embedder)


def _chunks(passages: Sequence[RetrievedChunk]) -> list[str]:
    return [passage.chunk.id for passage in passages]


def _only(*kept: Chunk) -> Callable[[Chunk], bool]:
    return lambda chunk: chunk in kept


def test_a_relevant_first_search_is_answered_with_no_other_model_call() -> None:  # 25.1
    setup = _setup(Script(keep=_only(KANT)))

    outcome = setup.answer(GOOD_WILL, k=2)  # KANT, then LAW

    assert setup.calls() == ["verdict", "verdict", "answer"]
    assert outcome.answer is not None
    assert outcome.answer.text == REPLY
    assert _chunks(outcome.answer.sources) == [KANT.id]
    assert _chunks(outcome.relevant) == [KANT.id]
    assert _chunks(outcome.left_out) == [LAW.id]
    assert outcome.query is None
    assert setup.embedder.query_calls == [GOOD_WILL]  # one search


def test_nothing_relevant_is_rewritten_once_and_searched_again() -> None:  # 25.2
    setup = _setup(Script(keep=_only(HOBBES), query="nasty brutish short life"))

    outcome = setup.answer(GOOD_WILL)  # KANT, not kept

    assert setup.calls() == ["verdict", "rewrite", "verdict", "answer"]
    rewrite = setup.model.conversations[1]
    assert all(author in rewrite[0].content for author in AUTHORS)
    assert rewrite[1].content == f"Question: {GOOD_WILL}"
    assert setup.embedder.query_calls == [GOOD_WILL, "nasty brutish short life"]
    # graded and answered against the reader's question, not the search query
    assert setup.verdicts() == [(KANT.id, GOOD_WILL), (HOBBES.id, GOOD_WILL)]
    assert setup.model.conversations[-1][1].content.endswith(f"Question: {GOOD_WILL}")
    assert outcome.query == "nasty brutish short life"
    assert outcome.answer is not None
    assert _chunks(outcome.answer.sources) == [HOBBES.id]
    assert _chunks(outcome.left_out) == [KANT.id]


def test_the_second_search_keeps_the_author_filter() -> None:  # 25.2
    # Unfiltered, this query finds HOBBES first; among Kant's works, LAW.
    setup = _setup(
        Script(keep=_only(LAW), query="solitary poore nasty brutish universal law maxim")
    )

    outcome = setup.answer(GOOD_WILL, Scope(author="Immanuel Kant"))

    assert outcome.answer is not None
    assert _chunks(outcome.answer.sources) == [LAW.id]
    assert _chunks(outcome.graded) == [KANT.id, LAW.id]


def test_a_passage_graded_in_the_first_search_is_not_graded_again() -> None:  # 25.3
    setup = _setup(Script(keep=_only(HOBBES), query="universal law maxim nasty brutish"))

    outcome = setup.answer("Can a good will become a universal law?", k=2)  # KANT, LAW

    # the second search finds LAW again and HOBBES: only HOBBES is new
    assert setup.verdicts() == [
        (KANT.id, "Can a good will become a universal law?"),
        (LAW.id, "Can a good will become a universal law?"),
        (HOBBES.id, "Can a good will become a universal law?"),
    ]
    assert _chunks(outcome.graded) == [KANT.id, LAW.id, HOBBES.id]


def test_nothing_relevant_after_the_rewrite_ends_without_an_answer() -> None:  # 25.4
    setup = _setup(Script(keep=lambda chunk: False))

    outcome = setup.answer(GOOD_WILL)

    assert setup.calls() == ["verdict", "rewrite", "verdict"]
    assert len(setup.embedder.query_calls) == 2  # no third search
    assert outcome.answer is None
    assert outcome.relevant == ()
    assert _chunks(outcome.left_out) == [KANT.id, HOBBES.id]


@pytest.mark.parametrize("anything_kept", [True, False])
def test_a_survey_of_all_authors_is_never_rewritten(anything_kept: bool) -> None:  # 25.5
    setup = _setup(Script(keep=_only(KANT) if anything_kept else (lambda chunk: False)))

    outcome = setup.answer(GOOD_WILL, Scope(survey=AUTHORS))

    assert setup.calls() == ["verdict"] * 3 + (["survey"] if anything_kept else [])
    assert sorted(_chunks(outcome.graded)) == sorted([KANT.id, MILL.id, HOBBES.id])
    assert outcome.query is None
    if anything_kept:
        assert outcome.answer is not None
        assert outcome.answer.text == "Immanuel Kant [1]\nParagraph 1."
    else:
        assert outcome.answer is None


def test_a_comparison_asks_about_each_author_on_their_own() -> None:  # 25.6
    setup = _setup(Script(subject="the good"))
    compared = ("Immanuel Kant", "Thomas Hobbes")
    question = "How do Kant and Hobbes differ on the good?"

    outcome = setup.answer(question, Scope(survey=compared, named=compared))

    assert setup.calls() == ["subject", "verdict", "verdict", "survey"]
    kant_question = "What does Immanuel Kant say about the good?"
    hobbes_question = "What does Thomas Hobbes say about the good?"
    assert outcome.author_questions == (kant_question, hobbes_question)
    # Unfiltered, Hobbes's question finds KANT first: each search keeps to its author.
    assert setup.embedder.query_calls == [kant_question, hobbes_question]
    assert setup.verdicts() == [(KANT.id, kant_question), (HOBBES.id, hobbes_question)]
    survey = setup.model.conversations[-1][1].content
    assert survey.endswith(f"Question: {question}")
    assert outcome.answer is not None
    assert (
        outcome.answer.text == "Immanuel Kant [1]\nParagraph 1.\n\nThomas Hobbes [2]\nParagraph 2."
    )


def _clock(monkeypatch: pytest.MonkeyPatch, *times: float) -> None:
    ticks: Iterator[float] = iter(times)
    monkeypatch.setattr("aporia_rag.graph.corrective.perf_counter", lambda: next(ticks))


def test_the_outcome_records_how_long_each_stage_took(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # 25.7
    # retrieve, grade, rewrite, retrieve, grade, generate: a start and an end each
    _clock(monkeypatch, 0.0, 0.5, 0.5, 2.0, 2.0, 3.5, 3.5, 3.75, 3.75, 5.0, 5.0, 9.0)
    setup = _setup(Script(keep=_only(HOBBES)))

    rewritten = setup.answer(GOOD_WILL)

    assert rewritten.seconds == {
        "retrieval": 0.75,
        "grading": 2.75,
        "rewriting": 1.5,
        "generation": 4.0,
    }

    _clock(monkeypatch, 0.0, 0.5, 0.5, 2.0, 2.0, 6.0)
    direct = _setup(Script(keep=_only(KANT))).answer(GOOD_WILL)

    assert direct.seconds == {"retrieval": 0.5, "grading": 1.5, "generation": 4.0}
