"""Tests for aporia_rag.grading.citations, plain-code checks of an answer (test plan: 28.1-28.5).

The passages are cut from the real ones behind the errors read in live answers
(Cicero, De Officiis III#24; Plato, Republic IX#59; Hobbes, Leviathan XVII#0).
"""

from collections.abc import Sequence

import pytest

from aporia_rag.domain.models import Answer, RetrievedChunk
from aporia_rag.grading.citations import Problem, check_citations
from aporia_rag.retrieval.names import authors_named_in
from support.builders import make_chunk

CICERO = RetrievedChunk(
    chunk=make_chunk(
        "cicero/book-iii#24",
        "he is mistaken in thinking that any ills affecting either his person or his\n"
        "property are more serious than those affecting his soul.",
        author="Marcus Tullius Cicero",
        title="De Officiis",
        section="Book III",
    ),
    score=0.5,
)
PLATO = RetrievedChunk(
    chunk=make_chunk(
        "plato/book-ix#59",
        "the Foole's argument aside, the soul is more honourable than the body. "
        "Certainly, he said.",
        author="Plato",
        title="The Republic",
        section="Book IX",
    ),
    score=0.5,
)
HOBBES = RetrievedChunk(
    chunk=make_chunk(
        "hobbes/chapter-xvii#0",
        "the foresight of their own preservation, and of a more contented life thereby; that is\n"
        "to say, of getting themselves out from that miserable condition of Warre, which is\n"
        "necessarily consequent to the naturall Passions of men",
        author="Thomas Hobbes",
        title="Leviathan",
        section="Chapter XVII",
    ),
    score=0.5,
)
AUTHORS = ("Marcus Tullius Cicero", "Plato", "Thomas Hobbes")


def _check(text: str, *sources: RetrievedChunk) -> list[Problem]:
    return check_citations(
        Answer(text=text, sources=sources), named_in=lambda part: authors_named_in(part, AUTHORS)
    )


def _details(problems: Sequence[Problem]) -> list[tuple[str, str]]:
    return [(problem.kind, problem.detail) for problem in problems]


@pytest.mark.parametrize(
    "text",
    [
        'Men act from "the foresight of their own preservation" [1].',
        "Men act from \u201cThe foresight of their own   preservation,\u201d says Hobbes [1].",
        'They seek "a more contented life thereby;" [1].',
    ],
    ids=["exact", "case-space-curly-end-punctuation", "end-punctuation"],
)
def test_a_quote_found_word_for_word_in_a_cited_passage_passes(text: str) -> None:  # 28.1
    assert _check(text, HOBBES) == []


def test_a_curly_apostrophe_matches_a_straight_one() -> None:  # 28.1
    assert _check("Set \u201cthe Foole\u2019s argument\u201d aside [1].", PLATO) == []


@pytest.mark.parametrize(
    "quote",
    ["to get themselves out from that miserable condition of Warre", "natural Passions"],
)
def test_a_quote_changed_from_the_passage_is_a_problem(quote: str) -> None:  # 28.1
    text = f'Peace is wanted, as men want "{quote}" [1]. Hobbes says so.'

    problems = _check(text, HOBBES)

    assert problems == [
        Problem(kind="quote", sentence=f'Peace is wanted, as men want "{quote}" [1].', detail=quote)
    ]


def test_a_quote_may_hold_a_full_stop() -> None:  # 28.1
    text = 'Plato: "the soul is more honourable than the body. Certainly, he said." [1]. Next.'

    assert _check(text, PLATO) == []
    assert _details(_check('Plato: "the body. Surely not" [1].', PLATO)) == [
        ("quote", "the body. Surely not")
    ]


def test_a_citation_after_a_closing_quote_belongs_to_the_quote() -> None:  # 28.5
    # In Cicero's passage only, cited as Plato's [2]: a problem, unless [2] came loose.
    text = '"those affecting his soul." [2]. Next.'

    assert _details(_check(text, CICERO, PLATO)) == [("quote", "those affecting his soul.")]


def test_a_quote_must_be_in_a_passage_its_sentence_cites() -> None:  # 28.1
    # In Plato's passage, but the sentence cites Cicero's.
    assert _details(_check('Cicero: "the soul is more honourable" [1].', CICERO, PLATO)) == [
        ("quote", "the soul is more honourable")
    ]


def test_a_quote_in_a_sentence_without_citations_may_be_in_any_passage() -> None:  # 28.2
    assert _check('Plato holds "the soul is more honourable than the body".', CICERO, PLATO) == []
    assert _details(_check('Plato holds "the body is a prison".', CICERO, PLATO)) == [
        ("quote", "the body is a prison")
    ]


@pytest.mark.parametrize("ellipsis", ["...", "\u2026", " ... ", " [...] "])
def test_a_quote_with_an_ellipsis_passes_when_every_part_is_in_the_passage(
    ellipsis: str,
) -> None:  # 28.3
    quote = f"he is mistaken{ellipsis}more serious than those affecting his soul"

    assert _check(f'Cicero: "{quote}" [1].', CICERO) == []
    assert _details(_check(f'Cicero: "he is right{ellipsis}his soul" [1].', CICERO)) == [
        ("quote", f"he is right{ellipsis}his soul")
    ]


def test_an_editors_change_in_brackets_is_not_the_authors_words() -> None:  # 28.3
    # Kant: "...must necessarily destroy itself"; quoted as in a live answer of 2026-10-17
    kant = RetrievedChunk(
        chunk=make_chunk("kant/first-section#29", "a law would necessarily destroy itself."),
        score=0.5,
    )

    assert _check('The maxim "necessarily destroy[s] itself" [1].', kant) == []
    assert _details(_check('The maxim "necessarily destroy[s] the law" [1].', kant)) == [
        ("quote", "necessarily destroy[s] the law")
    ]


def test_an_author_named_for_another_authors_passage_is_a_problem() -> None:  # 28.4
    text = (
        "Plato argues that no ills of the body are worse than those of the soul [1]. "
        "Plato also ranks the soul above the body [2]."
    )

    assert _check(text, CICERO, PLATO) == [
        Problem(
            kind="author",
            sentence="Plato argues that no ills of the body are worse than those of the soul [1].",
            detail="names Plato; [1] is by Marcus Tullius Cicero",
        )
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Cicero answers the Foole that Hobbes describes [1].",  # one of the named authors cited
        "Both Plato and Cicero rank the soul above the body [1][2].",
        "The soul matters more than the body [1].",  # no author named: nothing to check
    ],
)
def test_naming_an_author_of_a_cited_passage_passes(text: str) -> None:  # 28.4
    assert _check(text, CICERO, PLATO) == []


@pytest.mark.parametrize("citations", ["[1][2]", "[1, 2]", "[2] [1]"])
def test_citations_are_read_in_every_form(citations: str) -> None:  # 28.5
    text = f'Plato: "the soul is more honourable than the body" {citations}.'

    assert _check(text, CICERO, PLATO) == []


def test_a_citation_with_no_passage_is_a_problem() -> None:  # 28.5
    assert _check("The soul comes first [3].", CICERO, PLATO) == [
        Problem(kind="citation", sentence="The soul comes first [3].", detail="[3]")
    ]
