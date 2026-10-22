"""Measuring answers: what the corrective graph's model calls buy over plain RAG.

Plain RAG answers from the top k passages, whatever they are; the graph
grades them, searches again or on the web when none answers, and writes the
answer again when the citation checks find problems. Both answer the same
gold questions, and plain code compares what they did:

- Evidence, for the questions the corpus answers: the answers given at least
  one gold passage (a passage hit), and the share of the passages given that
  are gold. The labels are complete only for what the pooled model retrieved
  for the question as asked, so a passage found by a rewritten search can
  answer and still count as not gold.
- Outside questions (no evidence): the answers from the corpus, which cannot
  answer them, from the web, and the questions left without an answer.
- Checks: the answers with a citation-check problem in the answer given, the
  problems by kind, and the answers written again.
- Time: the median seconds per question.

What plain code cannot see (a claim its passage does not make) is found by
reading the answers, which `write_answers` saves with their passages.

A system is anything that turns a question into a `Run`, a protocol the
pipelines' `Outcome` satisfies, as `Retriever` satisfies `Searcher`; so this
module imports neither the pipelines nor the generator.
"""

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Protocol

from aporia_rag.domain.models import Answer, RetrievedChunk
from aporia_rag.evaluation.gold import GoldQuestion, GoldSet


class _Problem(Protocol):
    @property
    def kind(self) -> str: ...


class Run(Protocol):
    """What a system reports for one question."""

    @property
    def answer(self) -> Answer | None: ...  # None: it found nothing to answer from

    @property
    def route(self) -> str: ...  # how it got there: "plain", "normal", "loop", "web"

    @property
    def web(self) -> bool: ...  # the answer's sources, if any, are web pages

    @property
    def problems(self) -> Sequence[_Problem]: ...  # the citation checks' findings, in the answer

    @property
    def draft_problems(self) -> Sequence[_Problem]: ...  # in a first draft that was rewritten


type System = Callable[[str], Run]


@dataclass(frozen=True, slots=True)
class Result:
    """One system's run on one gold question, and how long it took."""

    question: GoldQuestion
    run: Run
    seconds: float

    @property
    def sources(self) -> tuple[RetrievedChunk, ...]:
        """The passages the answer was written from, in the order it cites them."""
        return () if self.run.answer is None else tuple(self.run.answer.sources)

    @property
    def problems(self) -> tuple[str, ...]:
        """The kind of each problem the checks found in the answer given."""
        return tuple(problem.kind for problem in self.run.problems)

    @property
    def revised(self) -> bool:
        """The answer was written again."""
        return bool(self.run.draft_problems)

    @property
    def gold(self) -> tuple[bool, ...]:
        """For each source, whether it holds the question's evidence."""
        return tuple(self.question.is_passage_hit(source.chunk) for source in self.sources)


@dataclass(frozen=True, slots=True)
class Summary:
    """One system's results, counted."""

    questions: int
    answered: int
    routes: Mapping[str, int]  # in the order first seen
    in_corpus: int  # questions with evidence
    grounded: int  # of those, answers given at least one gold passage
    passages: int  # passages given for the questions with evidence
    gold_passages: int  # of those, the gold ones
    outside: int  # questions without evidence
    outside_from_corpus: int
    outside_from_web: int
    outside_unanswered: int
    flagged: int  # answers with a problem
    problems: Mapping[str, int]  # by kind, in the order first seen
    revised: int  # answers written again
    median_seconds: float


def evaluate_answers(systems: Mapping[str, System], gold: GoldSet) -> dict[str, tuple[Result, ...]]:
    """Ask every system about every gold question, system after system, in order."""
    report: dict[str, tuple[Result, ...]] = {}
    for name, system in systems.items():
        results: list[Result] = []
        for question in gold.questions:
            started = perf_counter()
            run = system(question.question)
            results.append(Result(question, run, perf_counter() - started))
        report[name] = tuple(results)
    return report


def summarize(results: Sequence[Result]) -> Summary:
    """Count one system's results."""
    in_corpus = [result for result in results if result.question.has_evidence]
    outside = [result for result in results if not result.question.has_evidence]
    answered = [result for result in results if result.run.answer is not None]
    return Summary(
        questions=len(results),
        answered=len(answered),
        routes=dict(Counter(result.run.route for result in results)),
        in_corpus=len(in_corpus),
        grounded=sum(1 for result in in_corpus if any(result.gold)),
        passages=sum(len(result.sources) for result in in_corpus),
        gold_passages=sum(sum(result.gold) for result in in_corpus),
        outside=len(outside),
        outside_from_corpus=sum(
            1 for result in outside if result.run.answer is not None and not result.run.web
        ),
        outside_from_web=sum(
            1 for result in outside if result.run.answer is not None and result.run.web
        ),
        outside_unanswered=sum(1 for result in outside if result.run.answer is None),
        flagged=sum(1 for result in results if result.problems),
        problems=dict(Counter(kind for result in results for kind in result.problems)),
        revised=sum(1 for result in results if result.revised),
        median_seconds=median(result.seconds for result in results),
    )


def write_answers(path: Path, report: Mapping[str, Sequence[Result]]) -> None:
    """Save every system's answer to every question, one JSON object per line, for reading."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for system, results in report.items():
            for result in results:
                answer = result.run.answer
                record = {
                    "system": system,
                    "question_id": result.question.id,
                    "question": result.question.question,
                    "route": result.run.route,
                    "seconds": result.seconds,
                    "answer": None if answer is None else answer.text,
                    "revised": result.revised,
                    "sources": [
                        {
                            "id": source.chunk.id,
                            "citation": source.chunk.metadata.citation,
                            "text": source.chunk.text,
                            "gold": gold,
                        }
                        for source, gold in zip(result.sources, result.gold, strict=True)
                    ],
                    "problems": list(result.problems),
                }
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
