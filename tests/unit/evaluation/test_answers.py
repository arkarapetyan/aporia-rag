"""Tests for aporia_rag.evaluation.answers: systems' answers to the gold set (test plan: 29.1-29.6).

The results are built by hand, so each test states what every system
answered, from which passages, by which route.
"""

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

from aporia_rag.domain.models import Answer, RetrievedChunk, web_passage
from aporia_rag.evaluation.answers import (
    Result,
    Summary,
    System,
    evaluate_answers,
    summarize,
    write_answers,
)
from aporia_rag.evaluation.gold import Evidence, GoldQuestion, GoldSet
from support.builders import make_chunk

GOOD_WILL = Evidence(document="kant/first-section", quote="except a good will")
ANSWER = RetrievedChunk(
    chunk=make_chunk("kant/first-section#0", "Nothing can be called good, except a good will."),
    score=0.9,
)
NEARBY = RetrievedChunk(
    chunk=make_chunk("kant/first-section#1", "Intelligence, wit, judgement.", index=1), score=0.8
)
OTHER = RetrievedChunk(
    chunk=make_chunk(
        "mill/chapter-ii#0",
        "Actions are right in proportion as they tend to promote happiness.",
        author="John Stuart Mill",
        title="Utilitarianism",
        section="Chapter II",
    ),
    score=0.7,
)
PAGE = RetrievedChunk(
    chunk=web_passage(
        url="https://en.wikipedia.org/wiki/Language_game",
        title="Language game",
        text="A language-game is a concept developed by Ludwig Wittgenstein.",
    ),
    score=0.0,
)


@dataclass(frozen=True, slots=True)
class _Problem:
    kind: str


@dataclass(frozen=True, slots=True)
class ScriptedRun:
    """A `Run` as a pipeline's `Outcome` reports it, with the problems given by kind."""

    answer: Answer | None
    route: str
    web: bool = False
    kinds: tuple[str, ...] = ()  # of the problems left in the answer
    revised: bool = False  # the first draft had a problem and was written again

    @property
    def problems(self) -> tuple[_Problem, ...]:
        return tuple(_Problem(kind) for kind in self.kinds)

    @property
    def draft_problems(self) -> tuple[_Problem, ...]:
        return (_Problem("quote"),) if self.revised else ()


def _question(question_id: str, *evidence: Evidence) -> GoldQuestion:
    return GoldQuestion(
        id=question_id, question=f"Question {question_id}?", evidence=list(evidence)
    )


def _answer(*sources: RetrievedChunk) -> Answer:
    return Answer(text="An answer [1].", sources=sources)


def _clock(monkeypatch: pytest.MonkeyPatch, *times: float) -> None:
    ticks: Iterator[float] = iter(times)
    monkeypatch.setattr("aporia_rag.evaluation.answers.perf_counter", lambda: next(ticks))


TWO = GoldSet(questions=[_question("good-will", GOOD_WILL), _question("outside")])


def test_every_system_answers_every_question_once_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # 29.1
    calls: list[tuple[str, str]] = []

    def system(name: str) -> System:
        def reply(question: str) -> ScriptedRun:
            calls.append((name, question))
            return ScriptedRun(answer=None, route=name)

        return reply

    _clock(monkeypatch, 0.0, 1.5, 2.0, 2.5, 10.0, 14.0, 14.0, 20.0)

    report = evaluate_answers({"plain": system("plain"), "corrective": system("corrective")}, TWO)

    assert calls == [
        ("plain", "Question good-will?"),
        ("plain", "Question outside?"),
        ("corrective", "Question good-will?"),
        ("corrective", "Question outside?"),
    ]
    assert list(report) == ["plain", "corrective"]
    assert [(r.question.id, r.run.route, r.seconds) for r in report["plain"]] == [
        ("good-will", "plain", 1.5),
        ("outside", "plain", 0.5),
    ]
    assert [r.seconds for r in report["corrective"]] == [4.0, 6.0]


def test_a_results_sources_are_marked_gold_when_they_hold_the_evidence() -> None:  # 29.2
    question = _question("good-will", GOOD_WILL)

    answered = Result(
        question, ScriptedRun(answer=_answer(NEARBY, ANSWER, OTHER), route="plain"), 1.0
    )
    unanswered = Result(question, ScriptedRun(answer=None, route="normal"), 1.0)

    assert answered.gold == (False, True, False)
    assert answered.sources == (NEARBY, ANSWER, OTHER)
    assert unanswered.gold == ()
    assert unanswered.sources == ()


# Six questions, for every way an answer can meet the gold set:
# - good-will: answered from the evidence and a passage that is not;
# - wrong-passages: answered, but from no gold passage;
# - unanswered: has evidence, no answer;
# - outside-corpus, outside-web, outside-none: no evidence; answered from the corpus, from a web
#   page, not at all.
FIVE = GoldSet(
    questions=[
        _question("good-will", GOOD_WILL),
        _question("wrong-passages", GOOD_WILL),
        _question("unanswered", GOOD_WILL),
        _question("outside-corpus"),
        _question("outside-web"),
        _question("outside-none"),
    ]
)
REPLIES = {
    "good-will": ScriptedRun(
        answer=_answer(ANSWER, NEARBY), route="normal", kinds=("quote", "quote"), revised=True
    ),
    "wrong-passages": ScriptedRun(answer=_answer(OTHER), route="loop", kinds=("citation",)),
    "unanswered": ScriptedRun(answer=None, route="loop"),
    "outside-corpus": ScriptedRun(answer=_answer(OTHER, NEARBY), route="normal", kinds=("author",)),
    "outside-web": ScriptedRun(answer=_answer(PAGE), route="web", web=True),
    "outside-none": ScriptedRun(answer=None, route="web", web=True),
}


def _results(seconds: Mapping[str, float] | None = None) -> list[Result]:
    times = seconds or {}
    return [
        Result(question, REPLIES[question.id], times.get(question.id, 1.0))
        for question in FIVE.questions
    ]


def test_summary_counts_answers_routes_and_gold_passages() -> None:  # 29.3
    summary = summarize(_results())

    assert summary.questions == 6
    assert summary.answered == 4
    assert list(summary.routes.items()) == [("normal", 2), ("loop", 2), ("web", 2)]  # first seen
    assert summary.in_corpus == 3
    assert summary.grounded == 1  # good-will only: wrong-passages had no gold passage
    assert summary.passages == 3  # given for the questions with evidence: 2 + 1
    assert summary.gold_passages == 1


def test_summary_counts_where_the_answers_to_outside_questions_came_from() -> None:  # 29.4
    summary = summarize(_results())

    assert summary.outside == 3
    assert summary.outside_from_corpus == 1
    assert summary.outside_from_web == 1
    assert summary.outside_unanswered == 1


def test_summary_counts_problems_second_drafts_and_the_median_time() -> None:  # 29.5
    seconds = {"good-will": 9.0, "wrong-passages": 3.0, "outside-web": 20.0}

    summary = summarize(_results(seconds))

    assert summary.flagged == 3  # good-will, wrong-passages, outside-corpus
    assert summary.problems == {"quote": 2, "citation": 1, "author": 1}
    assert summary.revised == 1
    assert summary.median_seconds == 2.0  # 1, 1, 1, 3, 9, 20


def test_a_summary_of_nothing_answered_has_no_division_by_zero() -> None:  # 29.3
    summary = summarize(
        [Result(_question("outside"), ScriptedRun(answer=None, route="plain"), 2.0)]
    )

    assert summary == Summary(
        questions=1,
        answered=0,
        routes={"plain": 1},
        in_corpus=0,
        grounded=0,
        passages=0,
        gold_passages=0,
        outside=1,
        outside_from_corpus=0,
        outside_from_web=0,
        outside_unanswered=1,
        flagged=0,
        problems={},
        revised=0,
        median_seconds=2.0,
    )


def test_write_answers_saves_every_answer_with_its_passages(tmp_path: Path) -> None:  # 29.6
    plain = [
        Result(FIVE.questions[0], ScriptedRun(answer=_answer(NEARBY, ANSWER), route="plain"), 2.5),
        Result(FIVE.questions[2], ScriptedRun(answer=None, route="plain"), 0.5),
    ]
    corrective = [Result(FIVE.questions[0], REPLIES["good-will"], 9.0)]
    path = tmp_path / "eval" / "answers.jsonl"  # the folder is created

    write_answers(path, {"plain": plain, "corrective": corrective})

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [(r["system"], r["question_id"]) for r in records] == [
        ("plain", "good-will"),
        ("plain", "unanswered"),
        ("corrective", "good-will"),
    ]
    first = records[0]
    assert first == {
        "system": "plain",
        "question_id": "good-will",
        "question": "Question good-will?",
        "route": "plain",
        "seconds": 2.5,
        "answer": "An answer [1].",
        "revised": False,
        "sources": [
            {
                "id": NEARBY.chunk.id,
                "citation": NEARBY.chunk.metadata.citation,
                "text": NEARBY.chunk.text,
                "gold": False,
            },
            {
                "id": ANSWER.chunk.id,
                "citation": ANSWER.chunk.metadata.citation,
                "text": ANSWER.chunk.text,
                "gold": True,
            },
        ],
        "problems": [],
    }
    assert records[1]["answer"] is None
    assert records[1]["sources"] == []
    assert records[2]["revised"] is True
    assert records[2]["problems"] == ["quote", "quote"]
