"""Measuring the relevance grader against the gold set.

The corrective loop starts from the grader's verdicts, so its two errors
have different costs: a false "not relevant" can start a needless loop (a
query rewrite or a web search, while the corpus had the answer), and a false
"relevant" can keep the loop from starting and give the generator a passage
beside the point.

The gold set supplies the labels: a retrieved chunk is *evidence* when it is
a passage hit. That label is complete only where the evidence was pooled
(every chunk the pooled models retrieved was judged), and the gold set's
rule is stricter than the grader's prompt, so every disagreement is worth
reading, not only counting.

Every chunk of a retrieval report is judged once; the metrics then take the
top k of each question, so one run measures several values of k:

- Agreement: precision (the share of kept chunks that are evidence) and
  recall (the share of the evidence that is kept).
- Loop starts: the questions with no relevant chunk in the top k, by the
  gold set and by the grader.
- Time: the mean grading time per question.

The evaluation needs only something that judges one passage, the `Judge`
protocol below, which `RelevanceGrader` satisfies.
"""

from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean
from time import perf_counter
from typing import Protocol

from aporia_rag.domain.models import RetrievedChunk
from aporia_rag.evaluation.gold import GoldSet
from aporia_rag.evaluation.retrieval import RetrievalReport


class Judge(Protocol):
    def is_relevant(self, question: str, passage: RetrievedChunk) -> bool: ...


@dataclass(frozen=True, slots=True)
class Verdict:
    """What the grader said about one retrieved chunk, and what the gold set says."""

    rank: int
    chunk_id: str
    evidence: bool  # the gold set counts the chunk as evidence
    relevant: bool  # the grader kept it
    seconds: float


@dataclass(frozen=True, slots=True)
class QuestionVerdicts:
    question_id: str
    verdicts: tuple[Verdict, ...]  # in rank order


@dataclass(frozen=True, slots=True)
class GradingReport:
    questions: tuple[QuestionVerdicts, ...]  # in gold-set order


@dataclass(frozen=True, slots=True)
class Agreement:
    """How the grader's verdicts meet the gold set's, counted over chunks."""

    kept_evidence: int
    kept_only: int
    evidence_only: int
    neither: int

    @property
    def total(self) -> int:
        return self.kept_evidence + self.kept_only + self.evidence_only + self.neither

    @property
    def kept(self) -> int:
        return self.kept_evidence + self.kept_only

    @property
    def evidence(self) -> int:
        return self.kept_evidence + self.evidence_only

    @property
    def precision(self) -> float | None:
        """The share of the kept chunks that are evidence; None when nothing was kept."""
        return _share(self.kept_evidence, self.kept)

    @property
    def recall(self) -> float | None:
        """The share of the evidence that was kept; None when there is no evidence."""
        return _share(self.kept_evidence, self.evidence)


@dataclass(frozen=True, slots=True)
class LoopStarts:
    """The questions with no relevant chunk in their top k, by question id in gold-set order."""

    needed: tuple[str, ...]  # by the gold set and by the grader: the loop starts, rightly
    needless: tuple[str, ...]  # by the grader only: it dropped the evidence
    missed: tuple[str, ...]  # by the gold set only: the grader kept a chunk that does not answer


def evaluate_grading(retrieval: RetrievalReport, gold: GoldSet, judge: Judge) -> GradingReport:
    """Judge every retrieved chunk of every question, and label it with the gold set."""
    questions = {question.id: question for question in gold.questions}
    graded: list[QuestionVerdicts] = []
    for result in retrieval.results:
        question = questions[result.question_id]
        verdicts: list[Verdict] = []
        for rank, passage in enumerate(result.retrieved, start=1):
            started = perf_counter()
            relevant = judge.is_relevant(question.question, passage)
            verdicts.append(
                Verdict(
                    rank=rank,
                    chunk_id=passage.chunk.id,
                    evidence=question.is_passage_hit(passage.chunk),
                    relevant=relevant,
                    seconds=perf_counter() - started,
                )
            )
        graded.append(QuestionVerdicts(question.id, tuple(verdicts)))
    return GradingReport(tuple(graded))


def agreement(report: GradingReport, k: int) -> Agreement:
    """Count the verdicts in the top k of every question by how they meet the gold set."""
    verdicts = [verdict for question in report.questions for verdict in question.verdicts[:k]]

    def count(evidence: bool, relevant: bool) -> int:
        return sum(1 for v in verdicts if (v.evidence, v.relevant) == (evidence, relevant))

    return Agreement(
        kept_evidence=count(True, True),
        kept_only=count(False, True),
        evidence_only=count(True, False),
        neither=count(False, False),
    )


def loop_starts(report: GradingReport, k: int) -> LoopStarts:
    """The questions with no relevant chunk in their top k, by the gold set and by the grader."""

    def none_in_top_k(question: QuestionVerdicts, is_relevant: Callable[[Verdict], bool]) -> bool:
        return not any(is_relevant(verdict) for verdict in question.verdicts[:k])

    def ids(by_gold: bool, by_grader: bool) -> tuple[str, ...]:
        return tuple(
            question.question_id
            for question in report.questions
            if none_in_top_k(question, lambda verdict: verdict.evidence) == by_gold
            and none_in_top_k(question, lambda verdict: verdict.relevant) == by_grader
        )

    return LoopStarts(
        needed=ids(by_gold=True, by_grader=True),
        needless=ids(by_gold=False, by_grader=True),
        missed=ids(by_gold=True, by_grader=False),
    )


def seconds_per_question(report: GradingReport, k: int) -> float:
    """The mean time spent on a question's top k verdicts."""
    return fmean(
        sum(verdict.seconds for verdict in question.verdicts[:k]) for question in report.questions
    )


def _share(part: int, whole: int) -> float | None:
    return part / whole if whole else None
