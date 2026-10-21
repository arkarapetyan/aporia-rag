"""Tests for aporia_rag.retrieval.rewriter (test plan: 24.1-24.5).

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
from support.fakes import ScriptedChatModel, ungrammatical

AUTHORS = ("Immanuel Kant", "John Stuart Mill", "Thomas Hobbes")
QUESTION = "How can I test whether the principle behind my action is morally acceptable?"
COMPARISON = "How do Kant and Mill differ on lying?"


def test_rewrite_asks_once_for_a_query_in_the_authors_words() -> None:  # 24.1
    model = ScriptedChatModel(
        structured_reply=lambda prompt: (
            '{"query": "  categorical imperative universal law \\n", "author": " Immanuel Kant "}'
        )
    )

    rewrite = QueryRewriter(model).rewrite(QUESTION, authors=AUTHORS)

    assert rewrite == RewrittenQuery(
        query="categorical imperative universal law", author="Immanuel Kant"
    )
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
    ("schema", "field"),
    [(RewrittenQuery, "query"), (RewrittenQuery, "author"), (ComparisonSubject, "subject")],
)
def test_a_blank_reply_does_not_fit_the_schema(schema: type[BaseModel], field: str) -> None:  # 24.3
    reply = dict.fromkeys(schema.model_fields, "Some words") | {field: "  \n"}
    with pytest.raises(ValidationError, match=field):
        schema.model_validate(reply)
    # ...but the schema the model gets stays a plain string: Ollama refuses regex patterns (18.10)
    assert ungrammatical(schema.model_json_schema()) is None
    assert schema.model_json_schema()["properties"][field]["type"] == "string"


def test_the_author_is_asked_for_before_the_query() -> None:  # 24.4
    # The model writes the fields in the schema's order: it picks the author, then their words.
    assert list(RewrittenQuery.model_json_schema()["properties"]) == ["author", "query"]


def test_a_rewrite_and_a_subject_have_a_reply_cap() -> None:  # 24.5
    def reply(prompt: str) -> str:
        if "\nAuthors compared: " in prompt:
            return '{"subject": "lying"}'
        return '{"author": "Immanuel Kant", "query": "maxim universal law"}'

    model = ScriptedChatModel(structured_reply=reply)
    rewriter = QueryRewriter(model)

    rewriter.rewrite(QUESTION, authors=AUTHORS)
    rewriter.per_author(COMPARISON, ["Immanuel Kant", "John Stuart Mill"])

    assert model.max_tokens == [64, 32]
