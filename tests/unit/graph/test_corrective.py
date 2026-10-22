"""Tests for aporia_rag.graph.corrective, the corrective loop (test plan: 25.1-25.7, 26.4-26.8, 27,
28.7-28.9).

The graph runs the real retriever, grader, generator and rewriter over
in-memory fakes: the hashing embedder ranks passages by the words they share
with the query, one `ScriptedChatModel` plays every model role, and a
`ScriptedWebSearch` finds fixed web pages. `Script` decides the model's
replies by the kind of request, so each test states which passages the
grader keeps and what the rewriter writes, and the recorded conversations
show every model call the graph made.
"""

import json
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import pytest

from aporia_rag.domain.errors import ProviderError
from aporia_rag.domain.models import Chunk, RetrievedChunk, web_passage
from aporia_rag.domain.ports import IndexRecord
from aporia_rag.generation.generator import ANSWER_PROMPT, SURVEY_PROMPT, Generator
from aporia_rag.grading.relevance import SYSTEM_PROMPT as GRADING_PROMPT
from aporia_rag.grading.relevance import RelevanceGrader
from aporia_rag.graph.corrective import CorrectiveRAG, Outcome, Scope
from aporia_rag.retrieval.retriever import Retriever
from aporia_rag.retrieval.rewriter import SUBJECT_PROMPT, QueryRewriter
from support.builders import make_chunk
from support.fakes import (
    HashingEmbedder,
    InMemoryVectorStore,
    ScriptedChatModel,
    ScriptedWebSearch,
    hash_vector,
)

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
PAGE = web_passage(
    url="https://en.wikipedia.org/wiki/Original_position",
    title="Original position",
    text="Behind a veil of ignorance, no one knows their place in society.",
)
PAGES = (
    PAGE,
    web_passage(
        url="https://plato.stanford.edu/entries/original-position/",
        title="Original position",
        text="Rawls asks which principles people would choose behind the veil.",
    ),
    web_passage(
        url="https://www.britannica.com/topic/veil-of-ignorance",
        title="Veil of ignorance",
        text="The veil of ignorance is a device for thinking about justice.",
    ),
)
CHUNKS = (KANT, LAW, HOBBES, MILL, *PAGES)
AUTHORS = ("Immanuel Kant", "John Stuart Mill", "Thomas Hobbes")
GOOD_WILL = "What can be called good without qualification?"  # finds KANT first
REPLY = "The scripted answer [1]."


@dataclass
class Script:
    """The scripted model's replies: which passages it keeps, and what it rewrites."""

    keep: Callable[[Chunk], bool] = lambda chunk: True
    query: str = "nasty brutish short life"  # finds HOBBES first
    author: str = "Thomas Hobbes"  # the author the rewriter names: in the corpus
    subject: str = "the good"
    draft: str = REPLY  # the first answer
    revised: str = "The revised answer [1]."  # the answer written again, when asked
    paragraph: str = "Paragraph {n}."  # a survey's paragraph n

    def answer(self, prompt: str) -> str:
        return self.revised if prompt.startswith("Your answer has these problems") else self.draft

    def reply(self, prompt: str) -> str:
        if prompt.startswith("Passage from "):
            passage = prompt.split(":\n", 1)[1].rsplit("\n\nQuestion: ", 1)[0]
            [chunk] = [chunk for chunk in CHUNKS if chunk.text == passage]
            return f'{{"relevant": {str(self.keep(chunk)).lower()}}}'
        if prompt.startswith("[1] "):  # a survey: one paragraph per numbered passage
            count = len(re.findall(r"^\[\d+\] ", prompt, re.MULTILINE))
            paragraphs = ", ".join(
                json.dumps(self.paragraph.format(n=n)) for n in range(1, count + 1)
            )
            return f'{{"paragraphs": [{paragraphs}]}}'
        if "\nAuthors compared: " in prompt:
            return f'{{"subject": "{self.subject}"}}'
        return f'{{"query": "{self.query}", "author": "{self.author}"}}'


@dataclass
class Setup:
    graph: CorrectiveRAG
    model: ScriptedChatModel
    embedder: HashingEmbedder
    web: ScriptedWebSearch
    retriever: Retriever

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
        return [
            "revision" if len(c) > 2 else kinds.get(c[0].content, "rewrite")
            for c in self.model.conversations
        ]

    def verdicts(self) -> list[tuple[str, str]]:
        """(passage, question) of every verdict asked for, in order."""
        pairs = []
        for conversation in self.model.conversations:
            if conversation[0].content == GRADING_PROMPT:
                passage, question = (
                    conversation[1].content.split(":\n", 1)[1].rsplit("\n\nQuestion: ", 1)
                )
                [chunk] = [c for c in CHUNKS if c.text == passage]
                pairs.append((chunk.id, question))
        return pairs


def _setup(script: Script, web: ScriptedWebSearch | None = None) -> Setup:
    web = web or ScriptedWebSearch()
    embedder = HashingEmbedder()
    store = InMemoryVectorStore(embedder.space)
    store.upsert(
        [
            IndexRecord(chunk=chunk, vector=hash_vector(chunk.text), fingerprint="f")
            for chunk in (KANT, LAW, HOBBES, MILL)
        ]
    )
    model = ScriptedChatModel(reply=script.answer, structured_reply=script.reply)
    retriever = Retriever(embedder, store)
    graph = CorrectiveRAG(
        retriever, RelevanceGrader(model), Generator(model), QueryRewriter(model), web
    )
    return Setup(graph, model, embedder, web, retriever)


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
    assert setup.web.calls == []
    assert not outcome.web


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
    assert outcome.outside_author is None
    assert outcome.answer is not None
    assert _chunks(outcome.answer.sources) == [HOBBES.id]
    assert _chunks(outcome.left_out) == [KANT.id]
    assert setup.web.calls == []


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
    assert len(setup.embedder.query_calls) == 2  # no third corpus search: the web instead (26.5)
    assert setup.web.calls == [(GOOD_WILL, 1)]
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
    assert setup.web.calls == []
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
    assert setup.web.calls == []
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


OUTSIDE = "John Rawls"  # an author the rewriter may name who is not in the corpus


def _ids(chunks: Sequence[Chunk]) -> list[str]:
    return [chunk.id for chunk in chunks]


def test_an_author_outside_the_corpus_sends_the_question_to_the_web() -> None:  # 26.4
    web = ScriptedWebSearch([PAGE])
    setup = _setup(Script(keep=_only(PAGE), author=OUTSIDE), web)

    outcome = setup.answer(GOOD_WILL)  # KANT, not kept

    assert setup.calls() == ["verdict", "rewrite", "verdict", "answer"]
    assert setup.embedder.query_calls == [GOOD_WILL]  # the corpus is not searched again
    assert web.calls == [(GOOD_WILL, 1)]  # the reader's question, for k pages
    assert setup.verdicts() == [(KANT.id, GOOD_WILL), (PAGE.id, GOOD_WILL)]
    assert setup.model.conversations[-1][1].content.endswith(f"Question: {GOOD_WILL}")
    assert outcome.outside_author == OUTSIDE
    assert outcome.query is None  # nothing was searched again in the corpus
    assert outcome.web
    assert outcome.web_error is None
    assert outcome.answer is not None
    assert _chunks(outcome.answer.sources) == [PAGE.id]
    assert _chunks(outcome.left_out) == [KANT.id]


def test_a_corpus_author_is_searched_again_and_the_web_only_after_that() -> None:  # 26.5
    web = ScriptedWebSearch([PAGE])
    setup = _setup(Script(keep=_only(PAGE), author="Hobbes"), web)  # a surname names him

    outcome = setup.answer(GOOD_WILL)

    assert setup.calls() == ["verdict", "rewrite", "verdict", "verdict", "answer"]
    assert setup.embedder.query_calls == [GOOD_WILL, "nasty brutish short life"]
    assert web.calls == [(GOOD_WILL, 1)]
    assert outcome.query == "nasty brutish short life"
    assert outcome.outside_author is None
    assert outcome.web
    assert outcome.answer is not None
    assert _chunks(outcome.answer.sources) == [PAGE.id]
    assert _chunks(outcome.left_out) == [KANT.id, HOBBES.id]


@pytest.mark.parametrize("pages", [[PAGE], []], ids=["nothing-relevant", "nothing-found"])
def test_nothing_relevant_on_the_web_ends_without_an_answer(pages: list[Chunk]) -> None:  # 26.6
    web = ScriptedWebSearch(pages)
    setup = _setup(Script(keep=lambda chunk: False, author=OUTSIDE), web)

    outcome = setup.answer(GOOD_WILL)

    assert setup.calls() == ["verdict", "rewrite"] + ["verdict"] * len(pages)
    assert setup.embedder.query_calls == [GOOD_WILL]
    assert len(web.calls) == 1  # no other search of any kind
    assert outcome.answer is None
    assert outcome.web
    assert _chunks(outcome.left_out) == [KANT.id, *_ids(pages)]


def test_a_failed_web_search_ends_without_an_answer_and_says_why() -> None:  # 26.7
    web = ScriptedWebSearch(failure=ProviderError("ddgs: 202 Ratelimit"))
    setup = _setup(Script(keep=lambda chunk: False, author=OUTSIDE), web)

    outcome = setup.answer(GOOD_WILL)

    assert setup.calls() == ["verdict", "rewrite"]
    assert outcome.answer is None
    assert outcome.web
    assert outcome.web_error == "ddgs: 202 Ratelimit"
    assert _chunks(outcome.left_out) == [KANT.id]  # what was graded before is kept


def test_the_web_search_is_timed_as_a_stage_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # 26.8
    # retrieve, grade, rewrite, web search, grade, generate: a start and an end each
    _clock(monkeypatch, 0.0, 0.5, 0.5, 2.0, 2.0, 3.5, 3.5, 4.75, 4.75, 6.25, 6.25, 10.25)
    setup = _setup(Script(keep=_only(PAGE), author=OUTSIDE), ScriptedWebSearch([PAGE]))

    outcome = setup.answer(GOOD_WILL)

    assert outcome.seconds == {
        "retrieval": 0.5,
        "grading": 3.0,
        "rewriting": 1.5,
        "web search": 1.25,
        "generation": 4.0,
    }


def test_grading_stops_once_two_passages_are_kept() -> None:  # 27.1
    setup = _setup(Script())  # keeps every passage it grades

    outcome = setup.answer(GOOD_WILL, k=3)

    assert setup.calls() == ["verdict", "verdict", "answer"]
    found = setup.retriever.search(GOOD_WILL, k=3)
    assert _chunks(outcome.graded) == _chunks(outcome.relevant) == _chunks(found[:2])
    assert _chunks(outcome.skipped) == _chunks(found[2:])
    assert outcome.answer is not None
    assert _chunks(outcome.answer.sources) == _chunks(found[:2])
    assert outcome.left_out == ()


def test_the_early_stop_holds_for_web_pages() -> None:  # 27.2
    setup = _setup(
        Script(keep=lambda chunk: chunk in PAGES, author=OUTSIDE), ScriptedWebSearch(PAGES)
    )

    outcome = setup.answer(GOOD_WILL, k=3)  # 3 corpus passages, none kept: 3 web pages

    assert setup.calls() == ["verdict"] * 3 + ["rewrite"] + ["verdict"] * 2 + ["answer"]
    assert _chunks(outcome.relevant) == _ids(PAGES[:2])
    assert _chunks(outcome.skipped) == _ids(PAGES[2:])


def test_a_survey_grades_every_authors_passage() -> None:  # 27.3
    setup = _setup(Script())  # keeps every passage

    outcome = setup.answer(GOOD_WILL, Scope(survey=AUTHORS))

    assert setup.calls() == ["verdict"] * 3 + ["survey"]
    assert len(outcome.relevant) == 3
    assert outcome.skipped == ()


def test_an_answer_without_problems_is_not_revised() -> None:  # 28.7
    setup = _setup(Script(keep=_only(KANT)))

    outcome = setup.answer(GOOD_WILL)

    assert setup.calls() == ["verdict", "answer"]
    assert outcome.draft_problems == outcome.problems == ()


def test_an_answer_with_problems_is_written_once_more_with_them_listed() -> None:  # 28.7
    setup = _setup(Script(keep=_only(KANT), draft="A good will alone is good [2]."))

    outcome = setup.answer(GOOD_WILL)  # one passage: KANT

    assert setup.calls() == ["verdict", "answer", "revision"]
    assert [problem.kind for problem in outcome.draft_problems] == ["citation"]
    request = setup.model.conversations[-1][-1].content
    assert outcome.draft_problems[0].note in request
    assert outcome.answer is not None
    assert outcome.answer.text == "The revised answer [1]."
    assert outcome.problems == ()  # the revised answer passes


def test_an_answer_is_revised_once_however_it_comes_back() -> None:  # 28.7
    still = "Hobbes calls it good [1]."  # names Hobbes for Kant's passage, again
    setup = _setup(Script(keep=_only(KANT), draft=still, revised=still))

    outcome = setup.answer(GOOD_WILL)

    assert setup.calls() == ["verdict", "answer", "revision"]
    assert [problem.kind for problem in outcome.problems] == ["author"]
    assert outcome.problems == outcome.draft_problems


def test_a_survey_is_not_checked() -> None:  # 28.8
    # A survey's paragraphs cite no numbers: a quotation checked would be "not in a passage".
    setup = _setup(Script(keep=_only(KANT), paragraph='Kant: "words no passage has" {n}.'))

    outcome = setup.answer(GOOD_WILL, Scope(survey=AUTHORS))

    assert "revision" not in setup.calls()
    assert outcome.draft_problems == ()


def test_the_revision_is_timed_as_a_stage_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # 28.9
    # retrieve, grade, generate, revise: a start and an end each
    _clock(monkeypatch, 0.0, 0.5, 0.5, 2.0, 2.0, 6.0, 6.0, 9.5)
    setup = _setup(Script(keep=_only(KANT), draft="An answer [7]."))

    outcome = setup.answer(GOOD_WILL)

    assert outcome.seconds == {
        "retrieval": 0.5,
        "grading": 1.5,
        "generation": 4.0,
        "revision": 3.5,
    }
