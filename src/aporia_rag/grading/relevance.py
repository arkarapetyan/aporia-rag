"""Deciding which retrieved passages help to answer a question.

Retrieval returns the passages *nearest* to a question, which is not the
same as the passages that answer it: a Hobbes chapter on honour is near a
question about the dishonourable. Left in, such a passage gets its own
paragraph in the answer, often misread. The grader asks the chat model
about each passage and keeps the ones it judges relevant.

One call per passage, not one call for all of them: each prompt stays
short (well inside the model's context), and each verdict depends on its
passage alone, not on what else happened to be retrieved.

Measured against the gold set (`evaluate --grade`, 2026-10-20): the one
systematic error was dropping a passage that answers one part of a question
with several parts, or gives one side of a comparison. Kant on truthfulness
was kept for "What does Kant say about lying?" and dropped for "How do Kant
and Mill differ on lying?". A prompt line naming that case was measured
the same day and reverted: it kept 25 more chunks (8 of them evidence, 17
not), recovered 2 of the 7 dropped passages, and still dropped Kant's for
the comparison. A prompt rule shifts the threshold; it does not fix this
case. Comparisons are therefore the caller's job (live test L.7 records the
limitation).
"""

from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel

from aporia_rag.domain.models import ChatMessage, RetrievedChunk
from aporia_rag.domain.ports import ChatModel

SYSTEM_PROMPT: Final = """\
You decide whether a passage from a philosophical text helps to answer a question.

- Reply true if the passage states, explains or argues something that the question asks \
about, even in part.
- Reply false if the passage only uses some of the same words, or is about another subject."""


MAX_TOKENS: Final = 16  # a verdict, `{"relevant": false}`, is about 6


class RelevanceVerdict(BaseModel):
    """The structured reply the grader asks for."""

    relevant: bool


class RelevanceGrader:
    """Keeps the passages that help to answer a question."""

    def __init__(self, model: ChatModel) -> None:
        self._model = model

    def relevant(self, question: str, passages: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """The passages judged relevant to `question`, in their original order."""
        return [passage for passage in passages if self.is_relevant(question, passage)]

    def is_relevant(self, question: str, passage: RetrievedChunk) -> bool:
        """The model's verdict on one passage, from one call."""
        chunk = passage.chunk
        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=(
                    f"Passage from {chunk.metadata.citation}:\n{chunk.text}\n\nQuestion: {question}"
                ),
            ),
        ]
        verdict = self._model.chat_structured(messages, RelevanceVerdict, max_tokens=MAX_TOKENS)
        return verdict.relevant
