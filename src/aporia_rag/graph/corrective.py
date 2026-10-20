"""The corrective loop: grade what retrieval finds, and search again when nothing answers.

Corrective RAG: retrieval returns the passages nearest to the question, the
grader keeps the ones that answer it, and the answer is written from those
alone. When the grader keeps nothing, the graph does not answer from noise:
it rewrites the question once into the corpus's own vocabulary, searches
again in the same scope, and grades what is new. If that finds nothing
either, the outcome says so (the web fallback will be the next node).

    START -> [ask_each_author] -> retrieve -> grade -> generate -> END
                                     ^           |
                                     +- rewrite -+ (nothing kept, once)

Where the model calls go (measured on 2026-10-20, see `retrieval.rewriter`):

- Normal path, a passage is kept: one verdict per passage and one answer.
  Nothing else: the loop costs nothing when it is not needed.
- Loop path, nothing kept: one rewrite, one more search with the same `k`
  and author filter, a verdict for each passage not graded yet, then an
  answer or none. At most one rewrite: one was enough in the measurements.
- Comparison path, two or more authors named: one call for the subject of the
  comparison, then each author's best passage, found and graded with that
  author's own question. A survey of all authors is never rewritten.

LangGraph states the loop as nodes and conditional edges, so the web
fallback adds one node and one edge. It is the only package that uses it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Final, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from aporia_rag.domain.models import Answer, RetrievedChunk
from aporia_rag.generation.generator import Generator
from aporia_rag.grading.relevance import RelevanceGrader
from aporia_rag.retrieval.retriever import Retriever
from aporia_rag.retrieval.rewriter import QueryRewriter

_ASK_EACH_AUTHOR: Final = "ask_each_author"
_RETRIEVE: Final = "retrieve"
_GRADE: Final = "grade"
_REWRITE: Final = "rewrite"
_GENERATE: Final = "generate"


@dataclass(frozen=True, slots=True)
class Scope:
    """Whose passages answer the question, and how many of each."""

    author: str | None = None  # the k best passages of this author only
    survey: tuple[str, ...] = ()  # instead, the best passage of each of these authors
    named: tuple[str, ...] = ()  # the authors named in the question, when they set the scope

    @property
    def unrestricted(self) -> bool:
        """The k best passages of any author: the one scope that can miss other authors."""
        return self.author is None and not self.survey

    @property
    def compares(self) -> bool:
        """Two or more authors named in the question: each is asked about on their own."""
        return len(self.named) > 1


def search(retriever: Retriever, question: str, scope: Scope, *, k: int) -> list[RetrievedChunk]:
    """The passages for `question` in its scope, best first: as `search` shows them."""
    if scope.survey:
        return retriever.best_per_author(question, scope.survey, limit=len(scope.survey))
    filters = None if scope.author is None else {"author": scope.author}
    return retriever.search(question, k=k, filters=filters)


@dataclass(frozen=True, slots=True)
class Outcome:
    """What the graph did for one question, and what it found."""

    answer: Answer | None  # None: no passage found answers the question
    graded: tuple[RetrievedChunk, ...]  # every passage graded, in the order found
    relevant: tuple[RetrievedChunk, ...]  # those judged relevant: the answer's sources
    query: str | None = None  # the rewritten search query, when the first search found nothing
    author_questions: tuple[str, ...] = ()  # a comparison: the question asked of each author
    seconds: Mapping[str, float] = field(default_factory=dict)  # by stage: retrieval, grading...

    @property
    def left_out(self) -> tuple[RetrievedChunk, ...]:
        """The passages graded and judged not relevant, in the order found."""
        return tuple(passage for passage in self.graded if passage not in self.relevant)


class _State(TypedDict, total=False):
    question: str
    scope: Scope
    k: int
    authors: tuple[str, ...]  # every author in the corpus, for the rewriter
    author_questions: dict[str, str]  # a comparison: author -> the question asked of them
    query: str | None
    found: list[RetrievedChunk]  # the latest search's passages
    graded: list[RetrievedChunk]
    relevant: list[RetrievedChunk]
    answer: Answer | None
    seconds: dict[str, float]


class CorrectiveRAG:
    """Answers a question from the passages that answer it, searching again when none does."""

    def __init__(
        self,
        retriever: Retriever,
        grader: RelevanceGrader,
        generator: Generator,
        rewriter: QueryRewriter,
    ) -> None:
        self._retriever = retriever
        self._grader = grader
        self._generator = generator
        self._rewriter = rewriter
        self._graph = self._build()

    def answer(self, question: str, scope: Scope, *, k: int, authors: Sequence[str]) -> Outcome:
        """Run the loop for `question`; `authors` are the corpus's, for the rewriter."""
        start: _State = {
            "question": question,
            "scope": scope,
            "k": k,
            "authors": tuple(authors),
            "author_questions": {},
            "query": None,
            "graded": [],
            "relevant": [],
            "answer": None,
            "seconds": {},
        }
        end: _State = self._graph.invoke(start)  # type: ignore[assignment]
        return Outcome(
            answer=end["answer"],
            graded=tuple(end["graded"]),
            relevant=tuple(end["relevant"]),
            query=end["query"],
            author_questions=tuple(end["author_questions"].values()),
            seconds=dict(end["seconds"]),
        )

    def _build(self) -> CompiledStateGraph[_State, None, _State, _State]:
        graph = StateGraph(_State)
        graph.add_node(_ASK_EACH_AUTHOR, self._ask_each_author)
        graph.add_node(_RETRIEVE, self._retrieve)
        graph.add_node(_GRADE, self._grade)
        graph.add_node(_REWRITE, self._rewrite)
        graph.add_node(_GENERATE, self._generate)
        graph.add_conditional_edges(START, self._first_step, [_ASK_EACH_AUTHOR, _RETRIEVE])
        graph.add_edge(_ASK_EACH_AUTHOR, _RETRIEVE)
        graph.add_edge(_RETRIEVE, _GRADE)
        graph.add_conditional_edges(_GRADE, self._after_grading, [_GENERATE, _REWRITE, END])
        graph.add_edge(_REWRITE, _RETRIEVE)
        graph.add_edge(_GENERATE, END)
        return graph.compile()

    # -- routing ------------------------------------------------------------

    @staticmethod
    def _first_step(state: _State) -> str:
        return _ASK_EACH_AUTHOR if state["scope"].compares else _RETRIEVE

    @staticmethod
    def _after_grading(state: _State) -> str:
        if state["relevant"]:
            return _GENERATE
        if state["scope"].survey or state["query"] is not None:
            return END  # a survey is not rewritten, and a question is rewritten once
        return _REWRITE

    # -- nodes --------------------------------------------------------------

    def _ask_each_author(self, state: _State) -> _State:
        started = perf_counter()
        compared = state["scope"].survey
        questions = self._rewriter.per_author(state["question"], compared)
        return {
            "author_questions": dict(zip(compared, questions, strict=True)),
            "seconds": _add(state, "rewriting", started),
        }

    def _retrieve(self, state: _State) -> _State:
        started = perf_counter()
        if state["author_questions"]:
            found = [
                results[0]
                for author, question in state["author_questions"].items()
                if (results := self._retriever.search(question, k=1, filters={"author": author}))
            ]
        else:
            query = state["query"] or state["question"]
            found = search(self._retriever, query, state["scope"], k=state["k"])
        return {"found": found, "seconds": _add(state, "retrieval", started)}

    def _grade(self, state: _State) -> _State:
        started = perf_counter()
        seen = {passage.chunk.id for passage in state["graded"]}
        new = [passage for passage in state["found"] if passage.chunk.id not in seen]
        asked = state["author_questions"]
        kept = [
            passage
            for passage in new
            if self._grader.is_relevant(
                asked.get(passage.chunk.metadata.author, state["question"]), passage
            )
        ]
        return {
            "graded": [*state["graded"], *new],
            "relevant": [*state["relevant"], *kept],
            "seconds": _add(state, "grading", started),
        }

    def _rewrite(self, state: _State) -> _State:
        started = perf_counter()
        query = self._rewriter.rewrite(state["question"], authors=state["authors"])
        return {"query": query, "seconds": _add(state, "rewriting", started)}

    def _generate(self, state: _State) -> _State:
        started = perf_counter()
        write = self._generator.survey if state["scope"].survey else self._generator.answer
        answer = write(state["question"], state["relevant"])
        return {"answer": answer, "seconds": _add(state, "generation", started)}


def _add(state: _State, stage: str, started: float) -> dict[str, float]:
    """`state`'s times with the time since `started` added to `stage`."""
    seconds = dict(state["seconds"])
    seconds[stage] = seconds.get(stage, 0.0) + perf_counter() - started
    return seconds
