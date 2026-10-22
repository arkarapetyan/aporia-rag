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
- Two kinds of reply from the same message: an answer, in a few paragraphs,
  and a survey of passages that come one per author. A survey's form is the
  code's: the model returns exactly one paragraph per passage, and each
  becomes a section headed by its author and number. Asked for in the
  prompt, the numbers were a tendency: a two-author survey came back with
  none. The rules that keep a reply faithful to the passages are shared.

Whether the model really keeps to the passages is not checked here: that is
the job of the corrective graph's graders.
"""

from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel, Field, create_model

from aporia_rag.domain.models import Answer, ChatMessage, RetrievedChunk
from aporia_rag.domain.ports import ChatModel

_ATTRIBUTION: Final = """\
- Philosophers often state a view in order to reject it. Attribute a view to an author \
only if the author holds it, and say so when they report or argue against someone else's."""
_QUOTES: Final = "Quote only words that appear in the passages."

ANSWER_PROMPT: Final = f"""\
You answer questions about philosophy from numbered passages of primary texts.

- Use only the passages that bear on the question, and leave the others out. If none \
of them answers it, say so instead of answering from memory.
{_ATTRIBUTION}
- Cite the passage behind each claim by its number in square brackets. {_QUOTES}
- Keep different authors' views apart.
- Write clear, modern English in a few short paragraphs."""

SURVEY_PROMPT: Final = f"""\
You compare what philosophers say on a question, from numbered passages of primary texts, \
one passage per author.

- Write one short paragraph per passage, in the order of the passages, on what its author \
says about the question. Say only what the passage says, not what you know of the author \
from elsewhere.
{_ATTRIBUTION}
- {_QUOTES}
- Write clear, modern English."""


# Reply caps: the longest answer so far took about 550 tokens; a survey's paragraphs are short.
ANSWER_MAX_TOKENS: Final = 1024
SURVEY_MAX_TOKENS_PER_PASSAGE: Final = 256


class SurveyReply(BaseModel):
    """A survey as the model writes it: one paragraph per passage, in their order."""

    paragraphs: list[str]


class Generator:
    """Writes the answer to a question from the passages it is given."""

    def __init__(self, model: ChatModel) -> None:
        self._model = model

    def answer(self, question: str, passages: Sequence[RetrievedChunk]) -> Answer:
        """The model's answer, citing `passages` by their numbers in the order given."""
        reply = self._model.chat(
            _messages(ANSWER_PROMPT, question, passages, cite=True),
            max_tokens=ANSWER_MAX_TOKENS,
        )
        return Answer(text=reply, sources=tuple(passages))

    def revise(self, question: str, draft: Answer, notes: Sequence[str]) -> Answer:
        """The answer written again from the draft's passages, told by `notes` what to fix.

        The model sees the conversation it answered in, its own draft, and the
        notes; the new answer cites the same passages by the same numbers.
        """
        listed = "\n".join(f"- {note}" for note in notes)
        messages = [
            *_messages(ANSWER_PROMPT, question, draft.sources, cite=True),
            ChatMessage(role="assistant", content=draft.text),
            ChatMessage(
                role="user",
                content=f"Your answer has these problems:\n{listed}\n\n"
                "Write the whole answer again with them fixed, keeping to the same rules.",
            ),
        ]
        reply = self._model.chat(messages, max_tokens=ANSWER_MAX_TOKENS)
        return Answer(text=reply, sources=draft.sources)

    def survey(self, question: str, passages: Sequence[RetrievedChunk]) -> Answer:
        """One section per passage, in the order given: its author and number, then a paragraph."""
        reply = self._model.chat_structured(
            _messages(SURVEY_PROMPT, question, passages),
            _survey_reply(len(passages)),
            max_tokens=SURVEY_MAX_TOKENS_PER_PASSAGE * len(passages),
        )
        sections = (
            f"{passage.chunk.metadata.author} [{number}]\n{paragraph.strip()}"
            for number, (passage, paragraph) in enumerate(
                zip(passages, reply.paragraphs, strict=True), start=1
            )
        )
        return Answer(text="\n\n".join(sections), sources=tuple(passages))


def _messages(
    instructions: str, question: str, passages: Sequence[RetrievedChunk], *, cite: bool = False
) -> list[ChatMessage]:
    """The instructions, then the numbered passages, the numbers to cite if `cite`, the question."""
    numbered = [
        f"[{number}] {passage.chunk.metadata.citation}\n{passage.chunk.text}"
        for number, passage in enumerate(passages, start=1)
    ]
    if cite:
        numbered.append(_numbers_to_cite(len(passages)))
    return [
        ChatMessage(role="system", content=instructions),
        ChatMessage(role="user", content="\n\n".join([*numbered, f"Question: {question}"])),
    ]


def _numbers_to_cite(count: int) -> str:
    """The citations an answer may use, named: no number for the model to invent or copy.

    With an example number in the prompt ("like [2]"), 4 of the 5 answers
    that made up citations on 2026-10-21 cited [2] from a single passage.
    """
    match count:
        case 1:
            return "There is one passage: cite it as [1]."
        case 2:
            return "There are 2 passages: cite them as [1] and [2]."
        case _:
            return f"There are {count} passages: cite them as [1] to [{count}]."


def _survey_reply(count: int) -> type[SurveyReply]:
    """`SurveyReply` with exactly `count` paragraphs, so none is skipped or two merged.

    The count is in the JSON schema (minItems, maxItems) for the provider to
    constrain generation, and validation checks it either way.
    """
    return create_model(
        "SurveyReply",
        __base__=SurveyReply,
        paragraphs=(list[str], Field(min_length=count, max_length=count)),
    )
