"""The corrective loop: grade what retrieval finds, and search again when nothing answers.

Corrective RAG: retrieval returns the passages nearest to the question, the
grader keeps the ones that answer it, and the answer is written from those
alone. When the grader keeps nothing, the graph does not answer from noise:
it rewrites the question once into the corpus's own vocabulary, searches
again in the same scope, and grades what is new. If that finds nothing
either, it searches the web, the last resort, and grades the pages the same
way. When even they do not answer, the outcome says so.

    START -> [ask_each_author] -> retrieve -> grade -> generate -> check -> [other_authors] -> END
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

Asked for (`others=True`), and only beside an answer from the corpus to an
unrestricted question, the best passage of up to `OTHER_AUTHORS` authors who
had none among those graded is graded too, and the relevant ones are listed:
top-k retrieval can fill every place with one author (Aristotle has 63 chunks
on friendship, Cicero 6).

LangGraph states the loop as nodes and conditional edges, so the web
fallback is one node and three edges. This module is the only one that uses it.
"""

from collections.abc import Callable, Iterator, Sequence
from time import perf_counter
from typing import Any, Final, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from aporia_rag.domain.errors import ProviderError
from aporia_rag.domain.models import Answer, RetrievedChunk
from aporia_rag.domain.ports import WebSearch
from aporia_rag.generation.generator import Generator
from aporia_rag.grading.citations import Problem, check_citations
from aporia_rag.grading.relevance import RelevanceGrader
from aporia_rag.pipelines.outcome import Outcome, Route, Step
from aporia_rag.retrieval.names import authors_named_in
from aporia_rag.retrieval.retriever import Retriever
from aporia_rag.retrieval.rewriter import QueryRewriter
from aporia_rag.retrieval.scope import Scope, search

_ASK_EACH_AUTHOR: Final = "ask_each_author"
_RETRIEVE: Final = "retrieve"
_GRADE: Final = "grade"
_REWRITE: Final = "rewrite"
_WEB_SEARCH: Final = "web_search"
_GENERATE: Final = "generate"
_CHECK: Final = "check"
_REVISE: Final = "revise"
_OTHER_AUTHORS: Final = "other_authors"

ENOUGH: Final = 2  # relevant passages after which grading stops (not in a survey)
# How many other authors are graded when asked for: each verdict is one model call (2-4 s on an
# 8 GB laptop GPU), so this bounds the extra wait.
OTHER_AUTHORS: Final = 4


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
    want_others: bool  # the reader asked for other authors' passages
    others: list[RetrievedChunk]
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

    def answer(
        self,
        question: str,
        scope: Scope,
        *,
        k: int,
        authors: Sequence[str],
        others: bool = False,
        on_step: Callable[[Step], None] | None = None,
    ) -> Outcome:
        """Run the loop for `question`; `authors` are the corpus's, for the rewriter.

        With `others`, also list other authors who discuss the question (see
        the module's notes). `on_step` is told of every step as it finishes.
        """
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
            "want_others": others,
            "others": [],
            "seconds": {},
        }
        end = start
        # "values" is the state before and after each node, "updates" what the node changed.
        stream = self._graph.stream(start, stream_mode=["values", "updates"])
        for mode, chunk in cast(Iterator[tuple[str, Any]], stream):
            if mode == "values":
                end = cast(_State, chunk)
            elif on_step is not None:
                [(node, update)] = cast(dict[str, _State], chunk).items()
                on_step(_step(node, end, cast(_State, {**end, **update})))
        return Outcome(
            answer=end["answer"],
            route=_route(end),
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
            others=tuple(end["others"]),
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
        graph.add_node(_OTHER_AUTHORS, self._other_authors)
        graph.add_conditional_edges(START, self._first_step, [_ASK_EACH_AUTHOR, _RETRIEVE])
        graph.add_edge(_ASK_EACH_AUTHOR, _RETRIEVE)
        graph.add_edge(_RETRIEVE, _GRADE)
        graph.add_conditional_edges(
            _GRADE, self._after_grading, [_GENERATE, _REWRITE, _WEB_SEARCH, END]
        )
        graph.add_conditional_edges(_REWRITE, self._after_rewrite, [_RETRIEVE, _WEB_SEARCH])
        graph.add_edge(_WEB_SEARCH, _GRADE)
        graph.add_edge(_GENERATE, _CHECK)
        graph.add_conditional_edges(_CHECK, self._after_check, [_REVISE, _OTHER_AUTHORS, END])
        graph.add_edge(_REVISE, _CHECK)
        graph.add_edge(_OTHER_AUTHORS, END)
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
        if state["problems"] and not state["revised"]:
            return _REVISE  # once
        # Only beside an answer from the corpus: the web was searched because the corpus had none.
        wanted = state["want_others"] and state["scope"].unrestricted and not state["web"]
        return _OTHER_AUTHORS if wanted else END

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

    def _other_authors(self, state: _State) -> _State:
        started = perf_counter()
        present = {passage.chunk.metadata.author for passage in state["graded"]}
        absent = [author for author in state["authors"] if author not in present]
        candidates = self._retriever.best_per_author(state["question"], absent, limit=OTHER_AUTHORS)
        others = self._grader.relevant(state["question"], candidates)
        return {"others": others, "seconds": _add(state, "other authors", started)}

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


def _step(node: str, before: _State, after: _State) -> Step:
    """What `node` did, from the state before and after it."""
    seconds = sum(after["seconds"].values()) - sum(before["seconds"].values())
    return Step(node=node, summary=_SUMMARIES[node](before, after), seconds=seconds)


def _retrieved(before: _State, after: _State) -> str:
    found = _count(len(after["found"]), "passage")
    if after["query"] is not None:
        return f'Searched again as "{after["query"]}": found {found}'
    return f"Found {found}"


def _graded(before: _State, after: _State) -> str:
    graded = len(after["graded"]) - len(before["graded"])
    if graded == 0:
        return "Nothing to grade"
    kind = "web page" if after["web"] else "passage"
    summary = f"Graded {_count(graded, kind)}: {len(after['relevant'])} relevant"
    if skipped := len(after["skipped"]) - len(before["skipped"]):
        summary += f", {skipped} not graded"
    return summary


def _rewritten(before: _State, after: _State) -> str:
    if after["outside_author"] is not None:
        return f"The rewriter named {after['outside_author']}, who is not in the corpus"
    return f'Rewrote the question as "{after["query"]}"'


def _searched_web(before: _State, after: _State) -> str:
    if after["web_error"] is not None:
        return f"The web search failed: {after['web_error']}"
    return f"Searched the web: found {_count(len(after['found']), 'page')}"


def _checked_summary(before: _State, after: _State) -> str:
    problems = len(after["problems"])
    return f"Checked the answer: {_count(problems, 'problem') if problems else 'no problem'}"


def _others_found(before: _State, after: _State) -> str:
    match len(after["others"]):
        case 0:
            return "No other author discusses this"
        case 1:
            return "1 other author discusses this"
        case count:
            return f"{count} other authors discuss this"


_SUMMARIES: Final[dict[str, Callable[[_State, _State], str]]] = {
    _ASK_EACH_AUTHOR: lambda before, after: (
        f"Asked each of the {len(after['author_questions'])} authors on their own"
    ),
    _RETRIEVE: _retrieved,
    _GRADE: _graded,
    _REWRITE: _rewritten,
    _WEB_SEARCH: _searched_web,
    _GENERATE: lambda before, after: (
        "Wrote a section for each author" if after["scope"].survey else "Wrote the answer"
    ),
    _CHECK: _checked_summary,
    _REVISE: lambda before, after: "Wrote the answer again, with its problems listed",
    _OTHER_AUTHORS: _others_found,
}


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _route(state: _State) -> Route:
    if state["web"]:
        return "web"
    return "loop" if state["query"] is not None else "normal"


def _add(state: _State, stage: str, started: float) -> dict[str, float]:
    """`state`'s times with the time since `started` added to `stage`."""
    seconds = dict(state["seconds"])
    seconds[stage] = seconds.get(stage, 0.0) + perf_counter() - started
    return seconds
