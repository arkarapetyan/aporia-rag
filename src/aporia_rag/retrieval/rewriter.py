"""Rewriting a question for retrieval, when its own words find nothing that answers it.

Readers ask in modern English; the corpus is old translations ("Soveraign",
"Common-wealth", "maxim"). Dense retrieval bridges most of that gap, but not
all: for 2 of the 40 retrieval gold questions, the grader keeps no passage in
the top 5 although the corpus answers them. Measured before this was built
(2026-10-20, data/eval/spike_rewrites.py): the chat model's rewrites, in the
vocabulary of the author most likely to answer, put gold evidence in the top
5 for all 3 retrieval questions that fail, in 1.5-1.9 s per rewrite. The
prompt below is the one that was measured.

A comparison fails differently. Asked "How do Kant and Mill differ on
lying?", the grader leaves out every passage, even those on lying; asked
"What does Immanuel Kant say about lying?" and the same of Mill, it keeps each
side (3 of 3 in the spike, against 1 of 3 for two plain-code framings of the
original question). Finding the subject takes the model, once per comparison;
asking it of each author is plain code.

The query is for the search only: verdicts and answers are always about the
reader's question.
"""

from collections.abc import Sequence
from typing import Annotated, Final

from pydantic import AfterValidator, BaseModel

from aporia_rag.domain.models import ChatMessage
from aporia_rag.domain.ports import ChatModel


def _not_blank(text: str) -> str:
    if not text.strip():
        raise ValueError("must not be blank")
    return text


# Checked when the reply is read, not written into the JSON schema: Ollama cannot turn an
# unanchored regex `pattern` into a grammar and refuses the request (found live, 2026-10-20).
_NonBlank = Annotated[str, AfterValidator(_not_blank)]

SUBJECT_PROMPT: Final = """\
You find the subject of a question that compares philosophers.

- Reply with the subject in a few words, so that it completes the question \
"What does this author say about ...?".
- Leave out the authors' names and the comparison itself."""


class RewrittenQuery(BaseModel):
    """The structured reply to a rewrite."""

    query: _NonBlank


class ComparisonSubject(BaseModel):
    """The structured reply naming what a comparison is about."""

    subject: _NonBlank


class QueryRewriter:
    """Writes search queries the corpus's own words can match."""

    def __init__(self, model: ChatModel) -> None:
        self._model = model

    def rewrite(self, question: str, *, authors: Sequence[str]) -> str:
        """One search query for `question`, in the words of whichever of `authors` would answer."""
        messages = [
            ChatMessage(role="system", content=_rewrite_prompt(authors)),
            ChatMessage(role="user", content=f"Question: {question}"),
        ]
        return self._model.chat_structured(messages, RewrittenQuery).query.strip()

    def per_author(self, question: str, authors: Sequence[str]) -> list[str]:
        """The comparison `question` asked of each of `authors` on their own, in their order."""
        messages = [
            ChatMessage(role="system", content=SUBJECT_PROMPT),
            ChatMessage(
                role="user",
                content=f"Question: {question}\nAuthors compared: {', '.join(authors)}",
            ),
        ]
        reply = self._model.chat_structured(messages, ComparisonSubject)
        subject = reply.subject.strip().rstrip("?.").strip()
        return [f"What does {author} say about {subject}?" for author in authors]


def _rewrite_prompt(authors: Sequence[str]) -> str:
    return (
        "You turn a question into a search query for a library of old English translations of "
        "these authors: " + ", ".join(authors) + ".\n\n"
        "- Use the words that the author most likely to answer the question would use.\n"
        "- Name that author.\n"
        "- Reply with one short query."
    )
