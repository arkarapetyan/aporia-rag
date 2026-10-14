"""Measuring retrieval on the gold set: where the first hit lands, Hit@k and MRR.

Every question is searched once for the top `DEPTH` chunks. For each, the
report keeps the 1-based rank of the first *passage hit* (a chunk holding
the evidence quote) and of the first *section hit* (a chunk from the
evidence section), or None when there is none in the top `DEPTH`. Every
metric is then computed from those ranks:

- Hit@k: the share of questions whose first hit is within the top k.
- MRR: the mean of 1/rank over all questions, a miss counting 0. It
  rewards ranking the answer first over merely including it.

The evaluation needs only something that can search, the `Searcher`
protocol below, which `Retriever` satisfies; later, the corrective graph
can be measured the same way.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from aporia_rag.domain.models import Chunk, RetrievedChunk
from aporia_rag.evaluation.gold import GoldSet

DEPTH: Final = 10

type Rank = int | None


class Searcher(Protocol):
    def search(self, question: str, *, k: int) -> list[RetrievedChunk]: ...


@dataclass(frozen=True, slots=True)
class QuestionResult:
    question_id: str
    passage_rank: Rank
    section_rank: Rank


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    results: tuple[QuestionResult, ...]

    @property
    def passage_ranks(self) -> tuple[Rank, ...]:
        return tuple(result.passage_rank for result in self.results)

    @property
    def section_ranks(self) -> tuple[Rank, ...]:
        return tuple(result.section_rank for result in self.results)


def evaluate_retrieval(searcher: Searcher, gold: GoldSet) -> RetrievalReport:
    """Search every gold question and report where its first hits are."""
    results: list[QuestionResult] = []
    for question in gold.questions:
        chunks = [result.chunk for result in searcher.search(question.question, k=DEPTH)]
        results.append(
            QuestionResult(
                question_id=question.id,
                passage_rank=_first_rank(chunks, question.is_passage_hit),
                section_rank=_first_rank(chunks, question.is_section_hit),
            )
        )
    return RetrievalReport(tuple(results))


def hit_rate(ranks: Sequence[Rank], k: int) -> float:
    """The share of questions whose first hit is within the top k."""
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[Rank]) -> float:
    """The mean of 1/rank, a miss counting 0."""
    return sum(1 / rank for rank in ranks if rank is not None) / len(ranks)


def _first_rank(chunks: Sequence[Chunk], is_hit: Callable[[Chunk], bool]) -> Rank:
    return next((rank for rank, chunk in enumerate(chunks, start=1) if is_hit(chunk)), None)
