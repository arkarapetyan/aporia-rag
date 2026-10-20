"""Real-model tests for the relevance grader (test plan: L.5, L.7).

Opt-in: `make test-llm`. It uses the configured chat model. The verdicts are
requested at temperature 0, so the outcome is stable from run to run.
"""

import pytest

from aporia_rag.bootstrap import Container
from aporia_rag.config import load_settings
from aporia_rag.domain.models import RetrievedChunk
from support.passages import HOBBES, KANT, KANT_ON_LYING, MILL_ON_DISSENT

pytestmark = pytest.mark.llm


def test_keeps_the_answering_passage_and_leaves_out_the_other() -> None:  # L.5
    passages = [RetrievedChunk(chunk=chunk, score=0.5) for chunk in (HOBBES, KANT)]

    with Container(load_settings()) as container:
        kept = container.relevance_grader.relevant(
            "Is there anything that is good without qualification?", passages
        )

    assert [passage.chunk for passage in kept] == [KANT]


@pytest.mark.xfail(
    strict=True,
    reason="known limitation (2026-10-20): the grader drops one author's side of a comparison; "
    "a prompt line did not fix it, so comparisons are the caller's job",
)
def test_keeps_one_side_of_a_comparison() -> None:  # L.7
    passages = [
        RetrievedChunk(chunk=chunk, score=0.5) for chunk in (KANT_ON_LYING, MILL_ON_DISSENT)
    ]

    with Container(load_settings()) as container:
        kept = container.relevance_grader.relevant(
            "How do Kant and Mill differ on lying?", passages
        )

    assert [passage.chunk for passage in kept] == [KANT_ON_LYING]
