"""Tests for aporia_rag.evaluation.grading: the grader against the gold set (test plan: 23.1-23.6).

Retrieval is a `ScriptedSearcher` and the grader a `ScriptedJudge`, so each
test states which chunks each question retrieves and which the grader keeps.
"""

from collections.abc import Iterator

import pytest

from aporia_rag.domain.models import RetrievedChunk
from aporia_rag.evaluation.gold import Evidence, GoldQuestion, GoldSet
from aporia_rag.evaluation.grading import (
    Agreement,
    GradingReport,
    LoopStarts,
    Verdict,
    agreement,
    evaluate_grading,
    loop_starts,
    seconds_per_question,
)
from aporia_rag.evaluation.retrieval import evaluate_retrieval
from support.builders import make_chunk
from support.fakes import ScriptedSearcher

GOOD_WILL = Evidence(document="kant/first-section", quote="except a good will")
ANSWER = make_chunk("kant/first-section#0", "Nothing can be called good, except a good will.")
NEARBY = make_chunk("kant/first-section#1", "Intelligence, wit, judgement.", index=1)
REPEAT = make_chunk("kant/first-section#2", "Once more: except a good will.", index=2)
OTHER = make_chunk(
    "mill/chapter-ii#0",
    "Actions are right in proportion as they tend to promote happiness.",
    author="John Stuart Mill",
    title="Utilitarianism",
    section="Chapter II",
)


class ScriptedJudge:
    """A `Judge` that keeps the given (question, chunk id) pairs and records every call."""

    def __init__(self, keep: set[tuple[str, str]] | None = None) -> None:
        self.keep = keep or set()
        self.calls: list[tuple[str, str]] = []

    def is_relevant(self, question: str, passage: RetrievedChunk) -> bool:
        self.calls.append((question, passage.chunk.id))
        return (question, passage.chunk.id) in self.keep


def _question(question_id: str, *evidence: Evidence) -> GoldQuestion:
    return GoldQuestion(
        id=question_id, question=f"Question {question_id}?", evidence=list(evidence)
    )


# Four questions, one for each way the grader and the gold set can meet:
# - good-will: evidence at ranks 2 and 4, kept, along with one chunk that is not evidence;
# - dropped: evidence at rank 1, and the grader keeps nothing;
# - unanswered: no evidence retrieved, and the grader keeps nothing;
# - outside: no evidence at all (the corpus does not answer it), and the grader keeps a chunk.
GOLD = GoldSet(
    questions=[
        _question("good-will", GOOD_WILL),
        _question("dropped", GOOD_WILL),
        _question("unanswered", GOOD_WILL),
        _question("outside"),
    ]
)
RETRIEVED = {
    "Question good-will?": [NEARBY, ANSWER, OTHER, REPEAT],
    "Question dropped?": [ANSWER, NEARBY],
    "Question unanswered?": [OTHER, NEARBY],
    "Question outside?": [OTHER, NEARBY],
}
KEPT = {
    ("Question good-will?", ANSWER.id),
    ("Question good-will?", OTHER.id),
    ("Question good-will?", REPEAT.id),
    ("Question outside?", OTHER.id),
}


def _grade(judge: ScriptedJudge, gold: GoldSet = GOLD) -> GradingReport:
    retrieval = evaluate_retrieval(ScriptedSearcher(RETRIEVED), gold)
    return evaluate_grading(retrieval, gold, judge)


def _clock(monkeypatch: pytest.MonkeyPatch, *times: float) -> None:
    ticks: Iterator[float] = iter(times)
    monkeypatch.setattr("aporia_rag.evaluation.grading.perf_counter", lambda: next(ticks))


def test_every_retrieved_chunk_is_judged_once_in_order() -> None:  # 23.1
    judge = ScriptedJudge(KEPT)

    report = _grade(judge)

    assert judge.calls == [
        (question, chunk.id) for question, chunks in RETRIEVED.items() for chunk in chunks
    ]
    assert [question.question_id for question in report.questions] == [
        "good-will",
        "dropped",
        "unanswered",
        "outside",
    ]
    good_will = report.questions[0].verdicts
    assert [(v.rank, v.chunk_id, v.evidence, v.relevant) for v in good_will] == [
        (1, NEARBY.id, False, False),
        (2, ANSWER.id, True, True),
        (3, OTHER.id, False, True),
        (4, REPEAT.id, True, True),
    ]


def test_each_verdict_records_how_long_the_judge_took(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # 23.2
    _clock(monkeypatch, 10.0, 12.5, 12.5, 13.0)
    gold = GoldSet(questions=[_question("dropped", GOOD_WILL)])

    report = _grade(ScriptedJudge(), gold)

    assert report.questions[0].verdicts == (
        Verdict(rank=1, chunk_id=ANSWER.id, evidence=True, relevant=False, seconds=2.5),
        Verdict(rank=2, chunk_id=NEARBY.id, evidence=False, relevant=False, seconds=0.5),
    )


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (2, Agreement(kept_evidence=1, kept_only=1, evidence_only=1, neither=5)),
        (4, Agreement(kept_evidence=2, kept_only=2, evidence_only=1, neither=5)),
    ],
)
def test_agreement_counts_the_top_k_of_every_question(k: int, expected: Agreement) -> None:  # 23.3
    assert agreement(_grade(ScriptedJudge(KEPT)), k) == expected


def test_precision_and_recall() -> None:  # 23.3
    found = Agreement(kept_evidence=2, kept_only=2, evidence_only=1, neither=5)

    assert found.precision == pytest.approx(2 / 4)
    assert found.recall == pytest.approx(2 / 3)


def test_precision_and_recall_without_a_denominator_are_none() -> None:  # 23.4
    nothing_kept = Agreement(kept_evidence=0, kept_only=0, evidence_only=2, neither=1)
    no_evidence = Agreement(kept_evidence=0, kept_only=1, evidence_only=0, neither=1)

    assert nothing_kept.precision is None
    assert nothing_kept.recall == 0.0
    assert no_evidence.precision == 0.0
    assert no_evidence.recall is None


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (2, LoopStarts(needed=("unanswered",), needless=("dropped",), missed=("outside",))),
        # good-will's rank 1 is neither evidence nor kept: within the top 1, the loop is needed
        (
            1,
            LoopStarts(
                needed=("good-will", "unanswered"), needless=("dropped",), missed=("outside",)
            ),
        ),
    ],
)
def test_loop_starts_compare_the_grader_with_the_gold_set(
    k: int, expected: LoopStarts
) -> None:  # 23.5
    assert loop_starts(_grade(ScriptedJudge(KEPT)), k) == expected


def test_seconds_per_question_is_the_mean_time_of_the_top_k_verdicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # 23.6
    # dropped: 2.0 s and 1.0 s; unanswered: 3.0 s and 0.0 s
    _clock(monkeypatch, 0.0, 2.0, 2.0, 3.0, 3.0, 6.0, 6.0, 6.0)
    gold = GoldSet(questions=[_question("dropped", GOOD_WILL), _question("unanswered", GOOD_WILL)])

    report = _grade(ScriptedJudge(), gold)

    assert seconds_per_question(report, 1) == pytest.approx((2.0 + 3.0) / 2)
    assert seconds_per_question(report, 2) == pytest.approx((3.0 + 3.0) / 2)
