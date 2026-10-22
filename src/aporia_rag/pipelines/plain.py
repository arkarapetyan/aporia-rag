"""Plain RAG: retrieve, then answer from whatever was retrieved.

The baseline the corrective pipeline is measured against (`evaluate
--answers`): the same retrieval, scope and answer prompt, but no grader, no
second search, no web and no second draft. The answer is checked with the
same citation checks, so the two can be compared, and is never written again.
"""

from collections.abc import Sequence
from time import perf_counter

from aporia_rag.generation.generator import Generator
from aporia_rag.grading.citations import check_citations
from aporia_rag.pipelines.outcome import Outcome
from aporia_rag.retrieval.names import authors_named_in
from aporia_rag.retrieval.retriever import Retriever
from aporia_rag.retrieval.scope import Scope, search


class PlainRAG:
    """Answers a question from the top k passages in its scope, whatever they are."""

    def __init__(self, retriever: Retriever, generator: Generator) -> None:
        self._retriever = retriever
        self._generator = generator

    def answer(self, question: str, scope: Scope, *, k: int, authors: Sequence[str]) -> Outcome:
        """Retrieve, answer from every passage found, check; `authors` are the corpus's."""
        started = perf_counter()
        found = search(self._retriever, question, scope, k=k)
        retrieved = perf_counter()
        answer = self._generator.answer(question, found)
        answered = perf_counter()
        problems = check_citations(answer, named_in=lambda text: authors_named_in(text, authors))
        return Outcome(
            answer=answer,
            route="plain",
            relevant=tuple(found),
            problems=tuple(problems),
            seconds={"retrieval": retrieved - started, "generation": answered - retrieved},
        )
