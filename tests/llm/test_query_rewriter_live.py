"""Real-model tests for the query rewriter (test plan: L.8-L.9).

Opt-in: `make test-llm`. They use the configured chat model, at temperature 0.
They check the form of what it writes; what a rewrite finds was measured on
the gold set before the rewriter was built (data/eval/spike_rewrites.py).
"""

import re

import pytest

from aporia_rag.bootstrap import Container
from aporia_rag.config import load_settings
from aporia_rag.ingestion.corpus import load_manifest

pytestmark = pytest.mark.llm

AUTHORS = sorted({work.author for work in load_manifest().works})


def test_rewrites_a_question_into_one_short_query() -> None:  # L.8
    question = "Can a dishonourable act ever really be in our interest?"

    with Container(load_settings()) as container:
        query = container.query_rewriter.rewrite(question, authors=AUTHORS)

    assert "\n" not in query
    assert 0 < len(query.split()) <= 30
    assert query != question


def test_asks_a_comparison_of_each_author() -> None:  # L.9
    compared = ["Immanuel Kant", "John Stuart Mill"]

    with Container(load_settings()) as container:
        questions = container.query_rewriter.per_author(
            "How do Kant and Mill differ on lying?", compared
        )

    assert len(questions) == 2
    for author, question in zip(compared, questions, strict=True):
        subject = question.removeprefix(f"What does {author} say about ").removesuffix("?")
        assert subject != question
        assert re.search(r"\bl(ie|ies|ying)\b", subject, re.IGNORECASE), subject
        assert "Kant" not in subject
        assert "Mill" not in subject
