"""The ddgs web search against the real web (test plan: I.10).

Opt-in (`make test-integration`), on a machine that can reach the search
engines; the cloud workspace cannot, so it fails there, as I.1 does.
"""

import pytest

from aporia_rag.adapters.web import DdgsWebSearch

pytestmark = pytest.mark.integration


def test_finds_web_pages_for_a_question_outside_the_corpus() -> None:  # I.10
    pages = DdgsWebSearch().search("What did Wittgenstein mean by a language game?", limit=5)

    assert pages
    for page in pages:
        assert page.id.startswith(("https://", "http://"))
        assert page.text
    assert any("Wittgenstein" in f"{page.metadata.title} {page.text}" for page in pages)
