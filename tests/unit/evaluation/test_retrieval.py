"""Tests for aporia_rag.evaluation.retrieval: relevance, metrics, the runner (test plan: 16.4-16.7).

The runner is tested with a scripted `Searcher`, so each test states exactly
which chunks come back, in which order, for which question.
"""

import pytest

from aporia_rag.domain.models import Chunk, RetrievedChunk
from aporia_rag.evaluation.gold import Evidence, GoldQuestion, GoldSet
from aporia_rag.evaluation.retrieval import (
    QuestionResult,
    evaluate_retrieval,
    hit_rate,
    mean_reciprocal_rank,
)
from support.builders import make_chunk

GOOD_WILL = Evidence(document="kant/first-section", quote="except a good will")
ANSWER = make_chunk("kant/first-section#0", "Nothing can be called good, except a good will.")
SAME_SECTION = make_chunk("kant/first-section#1", "Intelligence, wit, judgement.", index=1)
ELSEWHERE = make_chunk("mill/chapter-ii#0", "Mill quoting Kant: except a good will.")


class TestRelevance:
    def test_passage_hit_needs_the_document_and_the_quote(self) -> None:  # 16.4
        rewrapped = make_chunk("kant/first-section#0", "good,\n\nexcept  a\ngood will.")

        assert GOOD_WILL.is_passage_hit(ANSWER)
        assert GOOD_WILL.is_passage_hit(rewrapped)  # whitespace differences do not matter
        assert not GOOD_WILL.is_passage_hit(SAME_SECTION)  # right section, wrong paragraph
        assert not GOOD_WILL.is_passage_hit(ELSEWHERE)  # the quote, but in another work

    def test_section_hit_needs_only_the_document(self) -> None:  # 16.4
        assert GOOD_WILL.is_section_hit(ANSWER)
        assert GOOD_WILL.is_section_hit(SAME_SECTION)
        assert not GOOD_WILL.is_section_hit(ELSEWHERE)

    def test_any_piece_of_evidence_counts(self) -> None:  # 16.5
        question = GoldQuestion(
            id="good-will",
            question="Is anything good without qualification?",
            evidence=[Evidence(document="mill/chapter-ii", quote="Kant"), GOOD_WILL],
        )

        assert question.is_passage_hit(ANSWER)  # the second piece
        assert question.is_passage_hit(ELSEWHERE)  # the first piece
        assert not question.is_passage_hit(SAME_SECTION)
        assert question.is_section_hit(SAME_SECTION)


class TestMetrics:
    RANKS = (1, 3, None, 6)

    @pytest.mark.parametrize(("k", "expected"), [(1, 0.25), (5, 0.5), (10, 0.75)])
    def test_hit_rate_is_the_share_of_ranks_within_k(self, k: int, expected: float) -> None:  # 16.6
        assert hit_rate(self.RANKS, k) == expected

    def test_mean_reciprocal_rank_counts_misses_as_zero(self) -> None:  # 16.6
        assert mean_reciprocal_rank(self.RANKS) == pytest.approx((1 + 1 / 3 + 0 + 1 / 6) / 4)


class ScriptedSearcher:
    """A `Searcher` that returns fixed chunks per question and records every call."""

    def __init__(self, results: dict[str, list[Chunk]]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, question: str, *, k: int) -> list[RetrievedChunk]:
        self.calls.append((question, k))
        chunks = self.results.get(question, [])[:k]
        return [RetrievedChunk(chunk=chunk, score=1.0) for chunk in chunks]


def _question(question_id: str, evidence: Evidence = GOOD_WILL) -> GoldQuestion:
    return GoldQuestion(id=question_id, question=f"Question {question_id}?", evidence=[evidence])


def test_reports_the_first_passage_and_section_hit_of_every_question() -> None:  # 16.7
    gold = GoldSet(questions=[_question("found"), _question("missed"), _question("near")])
    searcher = ScriptedSearcher(
        {
            "Question found?": [SAME_SECTION, ANSWER],
            "Question missed?": [ELSEWHERE],
            "Question near?": [ELSEWHERE, ELSEWHERE, SAME_SECTION],
        }
    )

    report = evaluate_retrieval(searcher, gold)

    assert report.results == (
        QuestionResult(question_id="found", passage_rank=2, section_rank=1),
        QuestionResult(question_id="missed", passage_rank=None, section_rank=None),
        QuestionResult(question_id="near", passage_rank=None, section_rank=3),
    )
    assert searcher.calls == [
        ("Question found?", 10),
        ("Question missed?", 10),
        ("Question near?", 10),
    ]
    assert report.passage_ranks == (2, None, None)
    assert report.section_ranks == (1, None, 3)
