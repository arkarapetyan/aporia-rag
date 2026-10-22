"""Tests for aporia_rag.generation.generator (test plan: 19.1-19.6).

The chat model is a `ScriptedChatModel`, so each test sees exactly what the
model was sent and decides what it answers.
"""

import pytest

from aporia_rag.domain.models import Answer, ChatMessage, RetrievedChunk
from aporia_rag.generation.generator import ANSWER_PROMPT, SURVEY_PROMPT, Generator
from support.builders import make_chunk
from support.fakes import ScriptedChatModel

KANT = RetrievedChunk(
    chunk=make_chunk(
        "kant/first-section#0",
        "Nothing can be called good without qualification,\nexcept a good will.",
    ),
    score=0.62,
)
HOBBES = RetrievedChunk(
    chunk=make_chunk(
        "hobbes/chapter-xiii#3",
        "And the life of man, solitary, poore, nasty, brutish, and short.",
        author="Thomas Hobbes",
        title="Leviathan",
        section="Chapter XIII",
    ),
    score=0.48,
)
QUESTION = "Is anything good without qualification?"


def test_the_model_gets_the_numbered_cited_passages_and_then_the_question() -> None:  # 19.1
    model = ScriptedChatModel()

    # Hobbes first, despite the lower score: the numbers follow the order given.
    Generator(model).answer(QUESTION, [HOBBES, KANT])

    [[system, user]] = model.conversations
    assert system == ChatMessage(role="system", content=ANSWER_PROMPT)
    assert user == ChatMessage(
        role="user",
        content=(
            "[1] Thomas Hobbes, Leviathan, Chapter XIII\n"
            "And the life of man, solitary, poore, nasty, brutish, and short.\n\n"
            "[2] Immanuel Kant, Groundwork of the Metaphysic of Morals, First Section\n"
            "Nothing can be called good without qualification,\nexcept a good will.\n\n"
            "There are 2 passages: cite them as [1] and [2].\n\n"
            f"Question: {QUESTION}"
        ),
    )


def test_the_answer_is_the_reply_with_the_passages_as_its_numbered_sources() -> None:  # 19.2
    model = ScriptedChatModel(reply="Only a good will [2].")

    answer = Generator(model).answer(QUESTION, [HOBBES, KANT])

    assert answer == Answer(text="Only a good will [2].", sources=(HOBBES, KANT))
    assert answer.sources[2 - 1] == KANT  # the passage the answer cites as [2]


def test_a_survey_has_one_section_per_passage_headed_by_its_author_and_number() -> None:
    # 19.3
    model = ScriptedChatModel(
        structured_reply=lambda prompt: '{"paragraphs": ["War of all.", " A good will.\\n"]}'
    )
    generator = Generator(model)

    survey = generator.survey(QUESTION, [HOBBES, KANT])
    generator.answer(QUESTION, [HOBBES, KANT])

    [surveyed, answered] = model.conversations
    assert surveyed[0] == ChatMessage(role="system", content=SURVEY_PROMPT)
    # the same passages and question, without the numbers to cite: a survey cites none
    passages, numbers, question = answered[1].content.rsplit("\n\n", 2)
    assert numbers.startswith("There are 2 passages")
    assert surveyed[1].content == f"{passages}\n\n{question}"
    [schema] = model.schemas
    paragraphs = schema.model_json_schema()["properties"]["paragraphs"]
    assert (paragraphs["minItems"], paragraphs["maxItems"]) == (2, 2)  # none skipped or merged
    assert survey == Answer(
        text="Thomas Hobbes [1]\nWar of all.\n\nImmanuel Kant [2]\nA good will.",
        sources=(HOBBES, KANT),
    )


def test_an_answer_and_a_survey_have_a_reply_cap() -> None:  # 19.4
    model = ScriptedChatModel(structured_reply=lambda prompt: '{"paragraphs": ["One.", "Two."]}')
    generator = Generator(model)

    generator.answer(QUESTION, [KANT, HOBBES])
    generator.survey(QUESTION, [KANT, HOBBES])

    assert model.max_tokens == [1024, 2 * 256]


def test_revise_sends_the_draft_and_what_to_fix_and_returns_the_new_answer() -> None:  # 19.5
    model = ScriptedChatModel(reply="Only a good will is good without qualification [2].")
    generator = Generator(model)
    draft = Answer(text='Only "a good will alone" [3].', sources=(HOBBES, KANT))
    notes = ['"Only ... [3]." cites [3], a number no passage has.', "Quote the exact words."]

    revised = generator.revise(QUESTION, draft, notes)

    [conversation] = model.conversations
    first = ScriptedChatModel()  # the messages a first answer is written from
    Generator(first).answer(QUESTION, [HOBBES, KANT])
    assert conversation[:2] == first.conversations[0]
    assert conversation[2] == ChatMessage(role="assistant", content=draft.text)
    assert conversation[3].role == "user"
    assert conversation[3].content == (
        "Your answer has these problems:\n"
        f"- {notes[0]}\n- {notes[1]}\n\n"
        "Write the whole answer again with them fixed, keeping to the same rules."
    )
    assert revised == Answer(
        text="Only a good will is good without qualification [2].", sources=(HOBBES, KANT)
    )
    assert model.max_tokens == [1024]


@pytest.mark.parametrize(
    ("count", "line"),
    [
        (1, "There is one passage: cite it as [1]."),
        (2, "There are 2 passages: cite them as [1] and [2]."),
        (5, "There are 5 passages: cite them as [1] to [5]."),
    ],
)
def test_the_numbers_to_cite_are_named(count: int, line: str) -> None:  # 19.6
    model = ScriptedChatModel()

    Generator(model).answer(QUESTION, [KANT] * count)

    [[system, user]] = model.conversations
    assert user.content.endswith(f"\n\n{line}\n\nQuestion: {QUESTION}")
    assert "[2]" not in system.content  # no example number for the model to copy
