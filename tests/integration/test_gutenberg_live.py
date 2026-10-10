"""Opt-in test against the real Project Gutenberg mirror (test plan: I.1).

Run with `make test-integration` on a machine that can reach the mirror.
It checks our assumptions about real files (mirror layout, marker wording)
that the unit tests can only encode from documentation.
"""

import httpx
import pytest

from aporia_rag.ingestion.cleaning import clean_gutenberg_text
from aporia_rag.ingestion.gutenberg import GutenbergMirror

KANT_METAPHYSIC_OF_MORALS = 5682  # ~177 kB, the smallest-but-one work in the corpus


@pytest.mark.integration
def test_downloads_and_cleans_a_real_work() -> None:  # I.1
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        raw = GutenbergMirror(client).download(KANT_METAPHYSIC_OF_MORALS)

    body = clean_gutenberg_text(raw)

    assert "good will" in body.lower()
    assert "START OF THE PROJECT GUTENBERG" not in body
    assert len(body) > 100_000
