"""Tests for aporia_rag.adapters.web, the ddgs web search (test plan: 26.2-26.3).

ddgs does not go through httpx, so the seam is the search function itself:
each test gives the adapter a function that answers as ddgs's `DDGS().text`
does (a list of dicts with "title", "href" and "body") or raises as it does.
"""

import re
from typing import Any

import pytest
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from aporia_rag.adapters.web import BACKENDS, DdgsWebSearch
from aporia_rag.domain.errors import ProviderError

QUESTION = "What did Wittgenstein mean by a language game?"
WIKIPEDIA = {
    "title": "Language game (philosophy)",
    "href": "https://en.wikipedia.org/wiki/Language_game_(philosophy)",
    "body": "A language-game is a philosophical concept developed by Ludwig Wittgenstein.",
}
SEP = {
    "title": "Ludwig Wittgenstein",
    "href": "https://plato.stanford.edu/entries/wittgenstein/",
    "body": "Considered by some to be the greatest philosopher of the 20th century.",
}


class TextSearch:
    """Answers like `DDGS().text` and records how it was called."""

    def __init__(self, results: list[dict[str, Any]] | Exception) -> None:
        self.results = results
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, query: str, **options: Any) -> list[dict[str, Any]]:
        self.calls.append((query, options))
        if isinstance(self.results, Exception):
            raise self.results
        return self.results


def test_results_become_web_passages_in_ddgs_order() -> None:  # 26.2
    search = TextSearch([WIKIPEDIA, SEP])

    pages = DdgsWebSearch(search).search(QUESTION, limit=2)

    assert search.calls == [(QUESTION, {"max_results": 2, "backend": BACKENDS})]
    assert [page.id for page in pages] == [WIKIPEDIA["href"], SEP["href"]]
    assert [page.metadata.citation for page in pages] == [
        "en.wikipedia.org, Language game (philosophy)",
        "plato.stanford.edu, Ludwig Wittgenstein",
    ]
    assert pages[0].text == WIKIPEDIA["body"]


def test_every_engine_is_asked_but_the_one_whose_pages_a_model_wrote() -> None:  # 26.2
    engines = BACKENDS.split(",")

    assert engines[0] == "wikipedia"
    assert "grokipedia" not in engines


def test_no_more_pages_than_the_limit() -> None:  # 26.2
    assert len(DdgsWebSearch(TextSearch([WIKIPEDIA, SEP])).search(QUESTION, limit=1)) == 1


@pytest.mark.parametrize(
    "unusable",
    [
        {**SEP, "body": "  "},
        {"title": SEP["title"], "href": SEP["href"]},
        {**SEP, "href": "plato.stanford.edu/entries/wittgenstein/"},
        {"title": SEP["title"], "body": SEP["body"]},
    ],
    ids=["blank-text", "no-text", "not-a-web-url", "no-url"],
)
def test_a_result_with_no_text_or_web_url_is_skipped(unusable: dict[str, Any]) -> None:  # 26.2
    pages = DdgsWebSearch(TextSearch([unusable, WIKIPEDIA])).search(QUESTION, limit=5)

    assert [page.id for page in pages] == [WIKIPEDIA["href"]]


def test_a_result_with_no_title_is_titled_by_its_url() -> None:  # 26.2
    [page] = DdgsWebSearch(TextSearch([{**SEP, "title": ""}])).search(QUESTION, limit=5)

    assert page.metadata.title == SEP["href"]


def test_nothing_found_is_no_pages() -> None:  # 26.3
    nothing = TextSearch(DDGSException("No results found."))

    assert DdgsWebSearch(nothing).search(QUESTION, limit=5) == []


@pytest.mark.parametrize(
    "failure",
    [
        RatelimitException("https://html.duckduckgo.com/html 202 Ratelimit"),
        TimeoutException("https://www.bing.com/search: timed out"),
        DDGSException("ConnectError('dns error')"),
    ],
    ids=["rate-limit", "time-out", "no-network"],
)
def test_a_failed_search_is_a_provider_error_naming_ddgs(failure: DDGSException) -> None:  # 26.3
    with pytest.raises(ProviderError, match=f"^ddgs: {re.escape(str(failure))}$") as raised:
        DdgsWebSearch(TextSearch(failure)).search(QUESTION, limit=5)

    assert raised.value.__cause__ is failure
