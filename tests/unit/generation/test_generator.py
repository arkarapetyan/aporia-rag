"""Tests for aporia_rag.generation.generator (test plan: 19.1-19.2).

The chat model is a `ScriptedChatModel`, so each test sees exactly what the
model was sent and decides what it answers.
"""

from aporia_rag.domain.models import Answer, ChatMessage, RetrievedChunk
from aporia_rag.generation.generator import SYSTEM_PROMPT, Generator
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
    assert system == ChatMessage(role="system", content=SYSTEM_PROMPT)
    assert user == ChatMessage(
        role="user",
        content=(
            "[1] Thomas Hobbes, Leviathan, Chapter XIII\n"
            "And the life of man, solitary, poore, nasty, brutish, and short.\n\n"
            "[2] Immanuel Kant, Groundwork of the Metaphysic of Morals, First Section\n"
            "Nothing can be called good without qualification,\nexcept a good will.\n\n"
            f"Question: {QUESTION}"
        ),
    )


def test_the_answer_is_the_reply_with_the_passages_as_its_numbered_sources() -> None:  # 19.2
    model = ScriptedChatModel(reply="Only a good will [2].")

    answer = Generator(model).answer(QUESTION, [HOBBES, KANT])

    assert answer == Answer(text="Only a good will [2].", sources=(HOBBES, KANT))
    assert answer.sources[2 - 1] == KANT  # the passage the answer cites as [2]
