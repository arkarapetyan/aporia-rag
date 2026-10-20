"""Tests for aporia_rag.retrieval.rewriter (test plan: 24.1-24.3).

The chat model is a `ScriptedChatModel`, so each test sees exactly what the
model was asked and decides what it replies.
"""

import pytest
from pydantic import BaseModel, ValidationError

from aporia_rag.domain.models import ChatMessage
from aporia_rag.retrieval.rewriter import (
    SUBJECT_PROMPT,
    ComparisonSubject,
    QueryRewriter,
    RewrittenQuery,
)
from support.fakes import ScriptedChatModel

AUTHORS = ("Immanuel Kant", "John Stuart Mill", "Thomas Hobbes")
QUESTION = "How can I test whether the principle behind my action is morally acceptable?"
COMPARISON = "How do Kant and Mill differ on lying?"


def test_rewrite_asks_once_for_a_query_in_the_authors_words() -> None:  # 24.1
    model = ScriptedChatModel(
        structured_reply=lambda prompt: '{"query": "  categorical imperative universal law \\n"}'
    )

    query = QueryRewriter(model).rewrite(QUESTION, authors=AUTHORS)

    assert query == "categorical imperative universal law"
    [(system, user)] = model.conversations
    assert system.role == "system"
    assert all(author in system.content for author in AUTHORS)
    assert user == ChatMessage(role="user", content=f"Question: {QUESTION}")
    assert model.schemas == [RewrittenQuery]


@pytest.mark.parametrize("subject", ["lying", " lying? ", "lying."])
def test_per_author_asks_once_for_the_subject_and_asks_each_author_about_it(
    subject: str,
) -> None:  # 24.2
    model = ScriptedChatModel(structured_reply=lambda prompt: f'{{"subject": "{subject}"}}')

    questions = QueryRewriter(model).per_author(COMPARISON, ["Immanuel Kant", "John Stuart Mill"])

    assert questions == [
        "What does Immanuel Kant say about lying?",
        "What does John Stuart Mill say about lying?",
    ]
    assert model.conversations == [
        [
            ChatMessage(role="system", content=SUBJECT_PROMPT),
            ChatMessage(
                role="user",
                content=(
                    f"Question: {COMPARISON}\nAuthors compared: Immanuel Kant, John Stuart Mill"
                ),
            ),
        ]
    ]
    assert model.schemas == [ComparisonSubject]


@pytest.mark.parametrize(
    ("schema", "field"), [(RewrittenQuery, "query"), (ComparisonSubject, "subject")]
)
def test_a_blank_reply_does_not_fit_the_schema(schema: type[BaseModel], field: str) -> None:  # 24.3
    with pytest.raises(ValidationError, match=field):
        schema.model_validate({field: "  \n"})
