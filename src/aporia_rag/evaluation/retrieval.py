"""Measuring retrieval on the gold set: where the first hit lands, Hit@k and MRR.

Every question is searched once for the top `DEPTH` chunks. For each, the
report keeps the 1-based rank of the first *passage hit* (a chunk holding
the evidence quote) and of the first *section hit* (a chunk from the
evidence section), or None when there is none in the top `DEPTH`. Every
metric is then computed from those ranks:

- Hit@k: the share of questions whose first hit is within the top k.
- MRR: the mean of 1/rank over all questions, a miss counting 0. It
  rewards ranking the answer first over merely including it.

A question without evidence (the corpus does not answer it) is searched like
any other, but it has no rank to find, so the metrics leave it out.

The report also keeps what was retrieved, and `write_retrievals` saves it
for reading: a retrieved chunk that answers the question but is not in the
gold set counts as a miss, and reading the misses is how such passages are
found and added (pooling).

The evaluation needs only something that can search, the `Searcher`
protocol below, which `Retriever` satisfies; later, the corrective graph
can be measured the same way.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
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
    has_evidence: bool
    passage_rank: Rank
    section_rank: Rank
    retrieved: tuple[RetrievedChunk, ...]


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    results: tuple[QuestionResult, ...]

    @property
    def passage_ranks(self) -> tuple[Rank, ...]:
        """The rank of each scored question's first passage hit."""
        return tuple(result.passage_rank for result in self._scored)

    @property
    def section_ranks(self) -> tuple[Rank, ...]:
        """The rank of each scored question's first section hit."""
        return tuple(result.section_rank for result in self._scored)

    @property
    def _scored(self) -> list[QuestionResult]:
        return [result for result in self.results if result.has_evidence]


def evaluate_retrieval(searcher: Searcher, gold: GoldSet) -> RetrievalReport:
    """Search every gold question and report where its first hits are."""
    results: list[QuestionResult] = []
    for question in gold.questions:
        retrieved = tuple(searcher.search(question.question, k=DEPTH))
        chunks = [result.chunk for result in retrieved]
        results.append(
            QuestionResult(
                question_id=question.id,
                has_evidence=question.has_evidence,
                passage_rank=_first_rank(chunks, question.is_passage_hit),
                section_rank=_first_rank(chunks, question.is_section_hit),
                retrieved=retrieved,
            )
        )
    return RetrievalReport(tuple(results))


def write_retrievals(path: Path, reports: Mapping[str, RetrievalReport], gold: GoldSet) -> None:
    """Save every retrieved chunk, one JSON object per line, with its verdicts.

    `reports` maps each model (or any other name for a setup) to its report
    on `gold`.
    """
    questions = {question.id: question for question in gold.questions}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for model, report in reports.items():
            for result in report.results:
                question = questions[result.question_id]
                for rank, retrieved in enumerate(result.retrieved, start=1):
                    chunk = retrieved.chunk
                    record = {
                        "model": model,
                        "question_id": question.id,
                        "question": question.question,
                        "rank": rank,
                        "score": retrieved.score,
                        "chunk_id": chunk.id,
                        "citation": chunk.metadata.citation,
                        "passage_hit": question.is_passage_hit(chunk),
                        "section_hit": question.is_section_hit(chunk),
                        "text": chunk.text,
                    }
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")


def hit_rate(ranks: Sequence[Rank], k: int) -> float:
    """The share of questions whose first hit is within the top k."""
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[Rank]) -> float:
    """The mean of 1/rank, a miss counting 0."""
    return sum(1 / rank for rank in ranks if rank is not None) / len(ranks)


def _first_rank(chunks: Sequence[Chunk], is_hit: Callable[[Chunk], bool]) -> Rank:
    return next((rank for rank, chunk in enumerate(chunks, start=1) if is_hit(chunk)), None)
