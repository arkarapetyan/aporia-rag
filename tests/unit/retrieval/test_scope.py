"""Tests for aporia_rag.retrieval.scope, whose passages answer a question (test plan: 30.1-30.4)."""

import pytest

from aporia_rag.domain.errors import AporiaError
from aporia_rag.domain.ports import IndexRecord
from aporia_rag.retrieval.retriever import Retriever
from aporia_rag.retrieval.scope import Scope, UnknownAuthorError, scope_for, search
from support.builders import make_chunk
from support.fakes import HashingEmbedder, InMemoryVectorStore, hash_vector

AUTHORS = ("Immanuel Kant", "John Stuart Mill", "Thomas Hobbes")
KANT = make_chunk("kant/first-section#0", "Lying is wrong: a lie destroys the dignity of man.")
LAW = make_chunk(
    "kant/second-section#0",
    "Act only on that maxim whereby thou canst will that it should become a universal law.",
    section="Second Section",
)
MILL = make_chunk(
    "mill/chapter-ii#0",
    "A lie may be right when it saves a life; truthfulness is useful.",
    author="John Stuart Mill",
    title="Utilitarianism",
    section="Chapter II",
)
HOBBES = make_chunk(
    "hobbes/chapter-xiii#0",
    "And the life of man, solitary, poore, nasty, brutish, and short.",
    author="Thomas Hobbes",
    title="Leviathan",
    section="Chapter XIII",
)


@pytest.mark.parametrize("name", ["hobbes", "Thomas Hobbes", "HOB"])
def test_an_author_given_is_matched_in_part_and_in_any_case(name: str) -> None:  # 30.1
    # The author given decides, even over an author the question names.
    assert scope_for("What does Kant say?", AUTHORS, author=name) == Scope(author="Thomas Hobbes")


@pytest.mark.parametrize(("name", "count"), [("spinoza", 0), ("o", 2)])
def test_an_author_given_must_match_exactly_one_author(name: str, count: int) -> None:  # 30.1
    with pytest.raises(UnknownAuthorError) as raised:
        scope_for("Anything?", AUTHORS, author=name)

    assert isinstance(raised.value, AporiaError)  # one line from every entry point
    message = str(raised.value)
    assert f"{name!r} matches {count} authors" in message
    assert all(author in message for author in AUTHORS)


def test_all_authors_is_a_survey_of_every_author() -> None:  # 30.2
    assert scope_for("What is friendship?", AUTHORS, all_authors=True) == Scope(survey=AUTHORS)


@pytest.mark.parametrize(
    ("question", "scope"),
    [
        ("Is lying ever right?", Scope()),
        (
            "What does kant say about lying?",
            Scope(author="Immanuel Kant", named=("Immanuel Kant",)),
        ),
        (
            "How do Mill and Kant differ on lying?",
            Scope(survey=AUTHORS[:2], named=AUTHORS[:2]),  # in the corpus's order
        ),
    ],
    ids=["none", "one", "several"],
)
def test_without_options_the_authors_the_question_names_set_the_scope(
    question: str, scope: Scope
) -> None:  # 30.3
    assert scope_for(question, AUTHORS) == scope


@pytest.fixture
def retriever() -> Retriever:
    embedder = HashingEmbedder()
    store = InMemoryVectorStore(embedder.space)
    store.upsert(
        [
            IndexRecord(chunk=chunk, vector=hash_vector(chunk.text), fingerprint="f")
            for chunk in (KANT, LAW, MILL, HOBBES)
        ]
    )
    return Retriever(embedder, store)


def test_search_takes_the_k_best_passages_of_the_scope(retriever: Retriever) -> None:  # 30.4
    question = "Is a lie ever right?"

    anyone = search(retriever, question, Scope(), k=2)
    kant = search(retriever, question, Scope(author="Immanuel Kant"), k=2)
    survey = search(retriever, question, Scope(survey=("Thomas Hobbes", "Immanuel Kant")), k=2)

    assert len(anyone) == 2
    assert {passage.chunk.metadata.author for passage in anyone} != {"Immanuel Kant"}
    assert [passage.chunk.id for passage in kant] == [KANT.id, LAW.id]
    assert sorted(passage.chunk.metadata.author for passage in survey) == [
        "Immanuel Kant",
        "Thomas Hobbes",
    ]  # one passage each
