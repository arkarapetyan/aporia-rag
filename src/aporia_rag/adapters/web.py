"""Web search through ddgs, a free metasearch library that needs no API key.

The corrective graph searches the web only for a question the corpus cannot
answer, so this is the one part of the default setup that leaves the
machine, and only then. ddgs asks several engines, Wikipedia first, and
merges their results. Each result is a title, a URL and a short text: an
engine's snippet, or a Wikipedia article's opening. A result becomes a web
passage, which the graph grades like any other passage.

Every ddgs text engine is asked except Grokipedia, whose articles are
written by a language model: an answer cites its web pages as sources, and
a source should be what people wrote, not another model's output.

ddgs does not go through httpx, so its search function is the seam for
tests (`text_search`).
"""

from collections.abc import Callable, Mapping
from typing import Any, Final

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from aporia_rag.domain.errors import ProviderError
from aporia_rag.domain.models import Chunk, web_passage

type TextSearch = Callable[..., list[dict[str, Any]]]

# How ddgs (9.x) reports a search that found nothing: the same exception type as a failure.
_NOTHING_FOUND: Final = "No results found."
# ddgs 9.16's text engines, but Grokipedia; ddgs ranks Wikipedia first, then the rest.
BACKENDS: Final = "wikipedia,brave,duckduckgo,google,mojeek,startpage,yahoo"


class DdgsWebSearch:
    """A `WebSearch` over ddgs's text search."""

    def __init__(self, text_search: TextSearch | None = None) -> None:
        self._text_search = text_search or DDGS().text  # ddgs's own time-out: 5 s per engine

    def search(self, query: str, *, limit: int) -> list[Chunk]:
        """At most `limit` pages for `query`, in ddgs's order."""
        try:
            results = self._text_search(query, max_results=limit, backend=BACKENDS)
        except DDGSException as error:
            if str(error) == _NOTHING_FOUND:
                return []
            raise ProviderError(f"ddgs: {error}") from error
        pages = [_page(result) for result in results if _usable(result)]
        return pages[:limit]


def _usable(result: Mapping[str, Any]) -> bool:
    url = str(result.get("href", ""))
    return url.startswith(("https://", "http://")) and bool(str(result.get("body", "")).strip())


def _page(result: Mapping[str, Any]) -> Chunk:
    url = str(result["href"])
    return web_passage(url=url, title=str(result.get("title") or url), text=str(result["body"]))
