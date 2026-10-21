"""Tests for aporia_rag.grading.relevance (test plan: 21.1-21.4).

The chat model is a `ScriptedChatModel`, so each test sees exactly what the
model was asked and decides each verdict.
"""

from aporia_rag.domain.models import ChatMessage, RetrievedChunk
from aporia_rag.grading.relevance import SYSTEM_PROMPT, RelevanceGrader, RelevanceVerdict
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
        "hobbes/chapter-x#3",
        "Honourable is whatsoever possession, action, or quality, is an argument of Power.",
        author="Thomas Hobbes",
        title="Leviathan",
        section="Chapter X",
    ),
    score=0.58,
)
QUESTION = "Is anything good without qualification?"


def test_each_passage_is_graded_in_its_own_call() -> None:  # 21.1
    model = ScriptedChatModel()

    RelevanceGrader(model).relevant(QUESTION, [KANT, HOBBES])

    first, second = model.conversations
    assert first == [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=(
                "Passage from Immanuel Kant, Groundwork of the Metaphysic of Morals, "
                "First Section:\n"
                "Nothing can be called good without qualification,\nexcept a good will.\n\n"
                f"Question: {QUESTION}"
            ),
        ),
    ]
    assert second[1].content.startswith("Passage from Thomas Hobbes, Leviathan, Chapter X:\n")
    assert model.schemas == [RelevanceVerdict, RelevanceVerdict]


def test_relevant_keeps_the_passages_judged_relevant_in_order() -> None:  # 21.2
    hobbes_only = ScriptedChatModel(
        structured_reply=lambda prompt: f'{{"relevant": {str("Hobbes" in prompt).lower()}}}'
    )
    everything = ScriptedChatModel()

    assert RelevanceGrader(hobbes_only).relevant(QUESTION, [KANT, HOBBES]) == [HOBBES]
    assert RelevanceGrader(everything).relevant(QUESTION, [HOBBES, KANT]) == [HOBBES, KANT]


def test_is_relevant_grades_one_passage_in_one_call() -> None:  # 21.3
    hobbes_only = ScriptedChatModel(
        structured_reply=lambda prompt: f'{{"relevant": {str("Hobbes" in prompt).lower()}}}'
    )
    grader = RelevanceGrader(hobbes_only)

    assert grader.is_relevant(QUESTION, HOBBES) is True
    assert grader.is_relevant(QUESTION, KANT) is False
    assert len(hobbes_only.conversations) == 2


def test_a_verdict_has_a_reply_cap() -> None:  # 21.4
    model = ScriptedChatModel()

    RelevanceGrader(model).is_relevant(QUESTION, KANT)

    assert model.max_tokens == [16]
