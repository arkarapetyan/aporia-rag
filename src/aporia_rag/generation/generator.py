"""Answering a question from retrieved passages: the "G" in RAG.

The model gets the passages numbered from 1, each headed by its citation,
then the question, and is told to answer only from them, citing the passage
behind each claim by its number. The choices behind that layout:

- Numbers in the answer, not citations: "[2]" is short and unambiguous, and
  `Answer.sources` keeps the passages in the same order, so every citation
  leads back to an exact chunk. A model asked to write "Kant, Groundwork,
  First Section" paraphrases it.
- The question last, after the material it must be answered from: the usual
  advice for long prompts.
- The passages in the order given (the retriever's: best first). Numbering
  is the caller's choice; the generator never reorders.

Whether the model really keeps to the passages is not checked here: that is
the job of the corrective graph's graders.
"""

from collections.abc import Sequence
from typing import Final

from aporia_rag.domain.models import Answer, ChatMessage, RetrievedChunk
from aporia_rag.domain.ports import ChatModel

SYSTEM_PROMPT: Final = """\
You answer questions about philosophy from numbered passages of primary texts.

- Use only the passages that bear on the question, and leave the others out. If none \
of them answers it, say so instead of answering from memory.
- Philosophers often state a view in order to reject it. Attribute a view to an author \
only if the author holds it, and say so when they report or argue against someone else's.
- Cite the passage behind each claim by its number in square brackets, like [2]. Quote \
only words that appear in the passages.
- Keep different authors' views apart.
- Write clear, modern English in a few short paragraphs."""


class Generator:
    """Writes the answer to a question from the passages it is given."""

    def __init__(self, model: ChatModel) -> None:
        self._model = model

    def answer(self, question: str, passages: Sequence[RetrievedChunk]) -> Answer:
        """The model's answer, citing `passages` by their numbers in the order given."""
        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=_prompt(question, passages)),
        ]
        return Answer(text=self._model.chat(messages), sources=tuple(passages))


def _prompt(question: str, passages: Sequence[RetrievedChunk]) -> str:
    numbered = (
        f"[{number}] {passage.chunk.metadata.citation}\n{passage.chunk.text}"
        for number, passage in enumerate(passages, start=1)
    )
    return "\n\n".join([*numbered, f"Question: {question}"])
