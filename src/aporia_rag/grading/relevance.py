"""Deciding which retrieved passages help to answer a question.

Retrieval returns the passages *nearest* to a question, which is not the
same as the passages that answer it: a Hobbes chapter on honour is near a
question about the dishonourable. Left in, such a passage gets its own
paragraph in the answer, often misread. The grader asks the chat model
about each passage and keeps the ones it judges relevant.

One call per passage, not one call for all of them: each prompt stays
short (well inside the model's context), and each verdict depends on its
passage alone, not on what else happened to be retrieved.
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


class RelevanceVerdict(BaseModel):
    """The structured reply the grader asks for."""

    relevant: bool


class RelevanceGrader:
    """Keeps the passages that help to answer a question."""

    def __init__(self, model: ChatModel) -> None:
        self._model = model

    def relevant(self, question: str, passages: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """The passages judged relevant to `question`, in their original order."""
        return [passage for passage in passages if self._is_relevant(question, passage)]

    def _is_relevant(self, question: str, passage: RetrievedChunk) -> bool:
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
        return self._model.chat_structured(messages, RelevanceVerdict).relevant
