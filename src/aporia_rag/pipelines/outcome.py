"""What a pipeline did for one question, and the steps it reports as it runs."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from aporia_rag.domain.models import Answer, RetrievedChunk
from aporia_rag.grading.citations import Problem

# How the answer was reached: "plain", retrieve then answer (`PlainRAG`); the corrective
# pipeline's "normal" (the first search answered), "loop" (a second corpus search, with the
# rewritten query) and "web" (the web was searched).
type Route = Literal["plain", "normal", "loop", "web"]


@dataclass(frozen=True, slots=True)
class Outcome:
    """What a pipeline did for one question, and what it found."""

    answer: Answer | None  # None: no passage found answers the question
    route: Route
    graded: tuple[RetrievedChunk, ...] = ()  # every passage graded, in the order found
    relevant: tuple[RetrievedChunk, ...] = ()  # those the answer was written from
    skipped: tuple[RetrievedChunk, ...] = ()  # found, not graded: enough relevant came first
    query: str | None = None  # the rewritten query, when the corpus was searched again with it
    outside_author: str | None = None  # what the rewriter named, when no author in the corpus
    web: bool = False  # the web was searched: the answer's sources, if any, are web pages
    web_error: str | None = None  # why the web search failed, if it did
    author_questions: tuple[str, ...] = ()  # a comparison: the question asked of each author
    draft_problems: tuple[Problem, ...] = ()  # found in the first answer: it was written again
    problems: tuple[Problem, ...] = ()  # found in the answer given, the second if there was one
    others: tuple[RetrievedChunk, ...] = ()  # other authors' relevant passages, when asked for
    seconds: Mapping[str, float] = field(default_factory=dict)  # by stage: retrieval, grading...

    @property
    def left_out(self) -> tuple[RetrievedChunk, ...]:
        """The passages graded and judged not relevant, in the order found."""
        return tuple(passage for passage in self.graded if passage not in self.relevant)


@dataclass(frozen=True, slots=True)
class Step:
    """One finished step of a pipeline, as it is reported while the pipeline runs."""

    node: str  # which step: "retrieve", "grade", "rewrite", "web_search", "generate"...
    summary: str  # what it did, in one sentence: "Graded 3 passages: 2 relevant"
    seconds: float
