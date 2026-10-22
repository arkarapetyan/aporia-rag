"""Whose passages answer a question: the scope rules every entry point shares.

A question is searched in one of three scopes:

- unrestricted: the k best passages of any author;
- one author: the k best passages of that author, when the reader names
  them (`author`, any part of the name) or the question names exactly one
  ("What does Kant say about lying?");
- a survey: the best passage of each of several authors, for all of them
  (`all_authors`) or for those the question names, when it names two or
  more (a comparison, which the pipeline asks of each author on their own).

The CLI and the web UI turn their options into a `Scope` with `scope_for`,
so the rules live here once.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from aporia_rag.domain.errors import AporiaError
from aporia_rag.domain.models import RetrievedChunk
from aporia_rag.retrieval.names import authors_named_in
from aporia_rag.retrieval.retriever import Retriever

DEFAULT_K: Final = 5  # passages per question unless the reader asks for another number


class UnknownAuthorError(AporiaError):
    """An author's name that matches no author of the corpus, or several."""


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


def scope_for(
    question: str,
    authors: Sequence[str],
    *,
    author: str | None = None,
    all_authors: bool = False,
) -> Scope:
    """`author` or `all_authors` decide; without them, the authors `question` names.

    `authors` are the corpus's; `author` may be any part of one of their
    names, in any case.
    """
    if author is not None:
        return Scope(author=resolve_author(author, authors))
    if all_authors:
        return Scope(survey=tuple(authors))
    match authors_named_in(question, authors):
        case []:
            return Scope()
        case [named]:
            return Scope(author=named, named=(named,))
        case named:
            return Scope(survey=tuple(named), named=tuple(named))


def resolve_author(name: str, authors: Sequence[str]) -> str:
    """The one author of `authors` whose name contains `name`, in any case."""
    matches = [author for author in authors if name.casefold() in author.casefold()]
    if len(matches) != 1:
        raise UnknownAuthorError(
            f"{name!r} matches {len(matches)} authors; authors: {', '.join(authors)}"
        )
    return matches[0]


def search(retriever: Retriever, question: str, scope: Scope, *, k: int) -> list[RetrievedChunk]:
    """The passages for `question` in its scope, best first: as `search` shows them."""
    if scope.survey:
        return retriever.best_per_author(question, scope.survey, limit=len(scope.survey))
    filters = None if scope.author is None else {"author": scope.author}
    return retriever.search(question, k=k, filters=filters)
