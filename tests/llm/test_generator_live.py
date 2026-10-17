"""Real-model tests for the configured chat model (test plan: L.4, L.6).

Opt-in: `make test-llm`. Like the embedder tests, it uses the application's
settings, so it tests whatever chat model you have configured. An answer is
sampled, so L.4 checks its shape (citations that exist), not its wording. A
survey is a structured reply at temperature 0, so L.6 checks what each of its
sections says.
"""

import re

import pytest

from aporia_rag.bootstrap import Container
from aporia_rag.config import load_settings
from aporia_rag.domain.models import RetrievedChunk
from support.passages import HOBBES, KANT, MILL

pytestmark = pytest.mark.llm

PASSAGES = [RetrievedChunk(chunk=chunk, score=0.5) for chunk in (HOBBES, KANT, MILL)]


def test_answers_from_the_passages_citing_them_by_number() -> None:  # L.4
    with Container(load_settings()) as container:
        answer = container.generator.answer(
            "Is there anything that is good without qualification?", PASSAGES
        )

    cited = {int(number) for number in re.findall(r"\[(\d+)\]", answer.text)}
    assert 2 in cited, answer.text  # Kant, the passage that answers
    assert cited <= {1, 2, 3}, answer.text  # no citation of a passage it was not given
    assert "<think>" not in answer.text


def test_surveys_each_author_from_their_own_passage() -> None:  # L.6
    passages = [RetrievedChunk(chunk=chunk, score=0.5) for chunk in (KANT, MILL)]

    with Container(load_settings()) as container:
        survey = container.generator.survey("What makes something morally good?", passages)

    # The sections and their headings are the generator's; what is in them is the model's.
    kant, mill = survey.text.split("\n\nJohn Stuart Mill [2]\n")
    assert "good will" in kant.casefold(), survey.text
    assert "happiness" in mill.casefold(), survey.text
