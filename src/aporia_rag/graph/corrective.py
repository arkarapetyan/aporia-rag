"""The corrective loop: grade what retrieval finds, and search again when nothing answers.

Corrective RAG: retrieval returns the passages nearest to the question, the
grader keeps the ones that answer it, and the answer is written from those
alone. When the grader keeps nothing, the graph does not answer from noise:
it rewrites the question once into the corpus's own vocabulary, searches
again in the same scope, and grades what is new. If that finds nothing
either, it searches the web, the last resort, and grades the pages the same
way. When even they do not answer, the outcome says so.

    START -> [ask_each_author] -> retrieve -> grade -> generate -> check -> END
                                                                  |   ^
                                    (problems, the first time)    +-> revise

    after grade, when nothing is kept:
      not rewritten yet -> rewrite -> retrieve     the author it names is in the corpus
                                   -> web_search   ...or is not
      rewritten         -> web_search -> grade
      web searched      -> END, with no answer

Where the model calls go (measured on 2026-10-20, see `retrieval.rewriter`):

- Normal path, a passage is kept: a verdict per passage, in the order found,
  until ENOUGH are kept, then one answer. Nothing else: the loop costs
  nothing when it is not needed.
- Loop path, nothing kept: one rewrite, which also names the author most
  likely to answer. A corpus author: one more search with the same `k` and
  author filter, a verdict for each passage not graded yet, and, if nothing
  is kept again, the web. An author outside the corpus: the web at once,
  with no second corpus search. The web gets the reader's question, for `k`
  pages, one verdict each, then an answer or none.
- Comparison path, two or more authors named: one call for the subject of the
  comparison, then each author's best passage, found and graded with that
  author's own question. A survey (all authors, or a comparison) is never
  rewritten and never goes to the web: its form is one section per author.

Grading stops once ENOUGH passages are kept (2026-10-21): about half of a
verdict is the model reading its prompt, and grading is most of the wait. On
the saved verdicts of the 40 retrieval questions, stopping at 2 makes 35%
fewer verdicts and leaves no question unanswered; answers draw on fewer
passages. The stop never changes whether the loop starts, since it comes
only after a kept passage. A survey grades every author's passage: each is
a section of its own.

Every answer but a survey is checked with plain code (`grading.citations`):
quotations word for word, citations with a passage behind them, the right
author named. With problems, the answer is written once more with them
listed (one more call, measured on 2026-10-21: 13 of 48 answers flagged),
and what remains is reported with it.

LangGraph states the loop as nodes and conditional edges, so the web
fallback is one node and three edges. It is the only package that uses it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Final, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from aporia_rag.domain.errors import ProviderError
from aporia_rag.domain.models import Answer, RetrievedChunk
from aporia_rag.domain.ports import WebSearch
from aporia_rag.generation.generator import Generator
from aporia_rag.grading.citations import Problem, check_citations
from aporia_rag.grading.relevance import RelevanceGrader
from aporia_rag.retrieval.names import authors_named_in
from aporia_rag.retrieval.retriever import Retriever
from aporia_rag.retrieval.rewriter import QueryRewriter

_ASK_EACH_AUTHOR: Final = "ask_each_author"
_RETRIEVE: Final = "retrieve"
_GRADE: Final = "grade"
_REWRITE: Final = "rewrite"
_WEB_SEARCH: Final = "web_search"
_GENERATE: Final = "generate"
_CHECK: Final = "check"
_REVISE: Final = "revise"

ENOUGH: Final = 2  # relevant passages after which grading stops (not in a survey)


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
    skipped: tuple[RetrievedChunk, ...] = ()  # found, not graded: ENOUGH relevant came first
    query: str | None = None  # the rewritten query, when the corpus was searched again with it
    outside_author: str | None = None  # what the rewriter named, when no author in the corpus
    web: bool = False  # the web was searched: the answer's sources, if any, are web pages
    web_error: str | None = None  # why the web search failed, if it did
    author_questions: tuple[str, ...] = ()  # a comparison: the question asked of each author
    draft_problems: tuple[Problem, ...] = ()  # found in the first answer: it was written again
    problems: tuple[Problem, ...] = ()  # found in the answer given, the second if there was one
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
    outside_author: str | None
    web: bool
    web_error: str | None
    found: list[RetrievedChunk]  # the latest search's passages
    graded: list[RetrievedChunk]
    relevant: list[RetrievedChunk]
    skipped: list[RetrievedChunk]
    answer: Answer | None
    draft_problems: list[Problem]
    problems: list[Problem]
    revised: bool
    seconds: dict[str, float]


class CorrectiveRAG:
    """Answers a question from the passages that answer it, searching again when none does."""

    def __init__(
        self,
        retriever: Retriever,
        grader: RelevanceGrader,
        generator: Generator,
        rewriter: QueryRewriter,
        web: WebSearch,
    ) -> None:
        self._retriever = retriever
        self._grader = grader
        self._generator = generator
        self._rewriter = rewriter
        self._web = web
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
            "outside_author": None,
            "web": False,
            "web_error": None,
            "graded": [],
            "relevant": [],
            "skipped": [],
            "answer": None,
            "draft_problems": [],
            "problems": [],
            "revised": False,
            "seconds": {},
        }
        end: _State = self._graph.invoke(start)  # type: ignore[assignment]
        return Outcome(
            answer=end["answer"],
            graded=tuple(end["graded"]),
            relevant=tuple(end["relevant"]),
            skipped=tuple(end["skipped"]),
            query=end["query"],
            outside_author=end["outside_author"],
            web=end["web"],
            web_error=end["web_error"],
            author_questions=tuple(end["author_questions"].values()),
            draft_problems=tuple(end["draft_problems"]),
            problems=tuple(end["problems"]),
            seconds=dict(end["seconds"]),
        )

    def _build(self) -> CompiledStateGraph[_State, None, _State, _State]:
        graph = StateGraph(_State)
        graph.add_node(_ASK_EACH_AUTHOR, self._ask_each_author)
        graph.add_node(_RETRIEVE, self._retrieve)
        graph.add_node(_GRADE, self._grade)
        graph.add_node(_REWRITE, self._rewrite)
        graph.add_node(_WEB_SEARCH, self._search_web)
        graph.add_node(_GENERATE, self._generate)
        graph.add_node(_CHECK, self._check)
        graph.add_node(_REVISE, self._revise)
        graph.add_conditional_edges(START, self._first_step, [_ASK_EACH_AUTHOR, _RETRIEVE])
        graph.add_edge(_ASK_EACH_AUTHOR, _RETRIEVE)
        graph.add_edge(_RETRIEVE, _GRADE)
        graph.add_conditional_edges(
            _GRADE, self._after_grading, [_GENERATE, _REWRITE, _WEB_SEARCH, END]
        )
        graph.add_conditional_edges(_REWRITE, self._after_rewrite, [_RETRIEVE, _WEB_SEARCH])
        graph.add_edge(_WEB_SEARCH, _GRADE)
        graph.add_edge(_GENERATE, _CHECK)
        graph.add_conditional_edges(_CHECK, self._after_check, [_REVISE, END])
        graph.add_edge(_REVISE, _CHECK)
        return graph.compile()

    # -- routing ------------------------------------------------------------

    @staticmethod
    def _first_step(state: _State) -> str:
        return _ASK_EACH_AUTHOR if state["scope"].compares else _RETRIEVE

    @staticmethod
    def _after_grading(state: _State) -> str:
        if state["relevant"]:
            return _GENERATE
        if state["scope"].survey or state["web"]:
            return END  # a survey keeps to the corpus, and the web is the last resort
        if state["query"] is None and state["outside_author"] is None:
            return _REWRITE  # once
        return _WEB_SEARCH

    @staticmethod
    def _after_rewrite(state: _State) -> str:
        return _WEB_SEARCH if state["outside_author"] is not None else _RETRIEVE

    @staticmethod
    def _after_check(state: _State) -> str:
        return _REVISE if state["problems"] and not state["revised"] else END  # once

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
        relevant = list(state["relevant"])
        graded = 0
        for passage in new:
            if len(relevant) >= ENOUGH and not state["scope"].survey:
                break
            graded += 1
            question = asked.get(passage.chunk.metadata.author, state["question"])
            if self._grader.is_relevant(question, passage):
                relevant.append(passage)
        return {
            "graded": [*state["graded"], *new[:graded]],
            "relevant": relevant,
            "skipped": [*state["skipped"], *new[graded:]],
            "seconds": _add(state, "grading", started),
        }

    def _rewrite(self, state: _State) -> _State:
        started = perf_counter()
        rewrite = self._rewriter.rewrite(state["question"], authors=state["authors"])
        if authors_named_in(rewrite.author, state["authors"]):
            route: _State = {"query": rewrite.query}  # search the corpus again with it
        else:  # the corpus has nothing of theirs to find: to the web, with no second search
            route = {"outside_author": rewrite.author}
        return {**route, "seconds": _add(state, "rewriting", started)}

    def _search_web(self, state: _State) -> _State:
        started = perf_counter()
        error = None
        try:
            pages = self._web.search(state["question"], limit=state["k"])
        except ProviderError as failure:  # offline, say: still report what was graded
            pages, error = [], str(failure)
        # A web page has no similarity to the query; its rank is its place in the list.
        found = [RetrievedChunk(chunk=page, score=0.0) for page in pages]
        return {
            "found": found,
            "web": True,
            "web_error": error,
            "seconds": _add(state, "web search", started),
        }

    def _generate(self, state: _State) -> _State:
        started = perf_counter()
        write = self._generator.survey if state["scope"].survey else self._generator.answer
        answer = write(state["question"], state["relevant"])
        return {"answer": answer, "seconds": _add(state, "generation", started)}

    def _check(self, state: _State) -> _State:
        problems = _checked(state)  # plain code, too quick to be a stage of its own
        if state["revised"]:
            return {"problems": problems}
        return {"problems": problems, "draft_problems": problems}

    def _revise(self, state: _State) -> _State:
        started = perf_counter()
        draft = state["answer"]
        assert draft is not None  # checked: there are problems only in an answer
        notes = [problem.note for problem in state["problems"]]
        answer = self._generator.revise(state["question"], draft, notes)
        return {"answer": answer, "revised": True, "seconds": _add(state, "revision", started)}


def _checked(state: _State) -> list[Problem]:
    """The plain-code problems of the state's answer; none for a survey (it cites no numbers)."""
    answer = state["answer"]
    if answer is None or state["scope"].survey:
        return []
    authors = state["authors"]
    return check_citations(answer, named_in=lambda text: authors_named_in(text, authors))


def _add(state: _State, stage: str, started: float) -> dict[str, float]:
    """`state`'s times with the time since `started` added to `stage`."""
    seconds = dict(state["seconds"])
    seconds[stage] = seconds.get(stage, 0.0) + perf_counter() - started
    return seconds
