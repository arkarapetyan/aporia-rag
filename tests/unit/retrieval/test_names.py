"""Tests for aporia_rag.retrieval.names (test plan: 22.1-22.3)."""

import pytest

from aporia_rag.retrieval.names import authors_named_in

AUTHORS = ["Immanuel Kant", "John Stuart Mill", "Plato", "Thomas Hobbes"]


@pytest.mark.parametrize(
    ("question", "author"),
    [
        ("what does kant say about lying?", "Immanuel Kant"),
        ("Is Kant's good will good in itself?", "Immanuel Kant"),
        ("Is Hobbes' state of nature a war?", "Thomas Hobbes"),
        ("What does Plato mean by justice?", "Plato"),
        ("How does John Stuart Mill define utility?", "John Stuart Mill"),
    ],
)
def test_a_surname_names_its_author_in_any_case_and_the_possessive(
    question: str, author: str
) -> None:  # 22.1
    assert authors_named_in(question, AUTHORS) == [author]


@pytest.mark.parametrize("question", ["Is Kantian ethics Platonic?", "What is friendship?"])
def test_only_whole_words_count(question: str) -> None:  # 22.2
    assert authors_named_in(question, AUTHORS) == []


def test_every_named_author_is_returned_in_the_order_given() -> None:  # 22.3
    named = authors_named_in("Did Mill answer Kant?", AUTHORS)

    assert named == ["Immanuel Kant", "John Stuart Mill"]
