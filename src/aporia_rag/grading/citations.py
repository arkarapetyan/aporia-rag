"""Checking an answer against its passages with plain code: quotations and named authors.

The relevance grader decides which passages the model gets; it cannot see
what the model then writes. Read claim by claim, live answers had the right
passages and still went wrong in ways plain code can find, at no cost:

- A quotation not word for word: "to get themselves out from that miserable
  condition of Warre" for Hobbes's "of getting themselves out...", "natural
  Passions" for "naturall Passions" (2026-10-21). The answer prompt asks for
  words that appear in the passages; this checks it.
- A view given to the wrong author: "Plato argues that ... [1]", where [1]
  is Cicero's (2026-10-20). A sentence that cites passages and names corpus
  authors must name the author of one of them; naming another author too is
  fine ("Cicero answers Hobbes [1]").

What it cannot see: a claim the passage does not make, or a view the
passage reports in order to reject it (component 28's layer 2, if measured
worth a model call).

Sentences are split at a full stop, question or exclamation mark followed by
a space, but never inside a quotation, and citations are read per sentence
("[1]", "[1][3]", "[1, 3]", "[5-6]"). A quotation is checked against the passages its
sentence cites, or against all the answer's passages when it cites none.
Case, spacing, curly quotes and apostrophes, Gutenberg's _italics_ and
punctuation at a quotation's ends do not count; an ellipsis or an editor's
change in brackets ("destroy[s] itself") splits a quotation into parts that
must each occur.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from aporia_rag.domain.models import Answer, RetrievedChunk

# The corpus authors a text names: `retrieval.names.authors_named_in` with the
# corpus's authors (passed in, since the grading package does not import retrieval).
type NamedIn = Callable[[str], Sequence[str]]

# A full stop, question or exclamation mark (or one closing a quotation) and a space, unless
# a citation follows: in 'said." [1].' the citation still belongs to the quotation.
_SENTENCE_END: Final = re.compile(r"(?<=[.!?])\s+(?!\[)|(?<=[.!?][\"\u201d])\s+(?!\[)")
_CITATION: Final = re.compile(r"\[(\d+(?:\s*[,\-\u2013]\s*\d+)*)\]")  # [1], [1, 3], [5-6]
_RANGE: Final = re.compile(r"(\d+)\s*[\-\u2013]\s*(\d+)")
_QUOTATION: Final = re.compile(r"\"([^\"]+)\"|\u201c([^\u201d]+)\u201d")
# Gaps in a quotation: an ellipsis, or an editor's change in brackets ("destroy[s] itself").
_GAP: Final = re.compile(r"\.\.\.|\u2026|\[[^\]]*\]")
_END_PUNCTUATION: Final = " \t\n.,;:!?"
_APOSTROPHES: Final = str.maketrans({"\u2019": "'", "\u2018": "'", "_": None})


@dataclass(frozen=True, slots=True)
class Problem:
    """Something in one sentence of an answer that its passages do not bear out."""

    kind: Literal["citation", "quote", "author"]
    sentence: str
    detail: str  # the citation with no passage, the quotation as written, or who is named

    @property
    def note(self) -> str:
        """What the model must fix, for a request to write the answer again."""
        if self.kind == "citation":
            return (
                f'"{self.sentence}" cites {self.detail}, a number no passage has: '
                "cite only the numbers of the passages."
            )
        if self.kind == "quote":
            return (
                f'The quotation "{self.detail}" is not word for word in a passage its sentence '
                "cites: quote the exact words of the passage you cite, or do not use quotation "
                "marks."
            )
        return f'"{self.sentence}" {self.detail}: name the author whose passage you cite.'


def check_citations(answer: Answer, *, named_in: NamedIn) -> list[Problem]:
    """Every problem plain code finds in `answer`, sentence by sentence, in order."""
    problems: list[Problem] = []
    for sentence in _sentences(answer.text):
        numbers = [n for group in _CITATION.findall(sentence) for n in _numbers(group)]
        cited = [answer.sources[n - 1] for n in numbers if 1 <= n <= len(answer.sources)]
        problems += [
            Problem("citation", sentence, f"[{n}]")
            for n in numbers
            if not 1 <= n <= len(answer.sources)
        ]
        searched = cited or list(answer.sources)
        problems += [
            Problem("quote", sentence, quotation)
            for quotation in _quotations(sentence)
            if not any(_occurs(quotation, passage) for passage in searched)
        ]
        if cited and (named := named_in(sentence)):
            authors = {passage.chunk.metadata.author for passage in cited}
            if not authors & set(named):
                by = ", ".join(
                    f"[{n}] is by {answer.sources[n - 1].chunk.metadata.author}"
                    for n in dict.fromkeys(numbers)
                    if 1 <= n <= len(answer.sources)
                )
                problems.append(Problem("author", sentence, f"names {', '.join(named)}; {by}"))
    return problems


def _numbers(group: str) -> list[int]:
    """The numbers of one citation: "1, 3" gives 1 and 3, "5-6" gives 5 and 6."""
    numbers: list[int] = []
    for part in group.split(","):
        if span := _RANGE.fullmatch(part.strip()):
            numbers += range(int(span[1]), int(span[2]) + 1)
        else:
            numbers.append(int(part))
    return numbers


def _sentences(text: str) -> list[str]:
    sentences = []
    for paragraph in text.splitlines():
        quoted = [match.span() for match in _QUOTATION.finditer(paragraph)]
        start = 0
        for end in _SENTENCE_END.finditer(paragraph):
            if not any(left < end.start() < right for left, right in quoted):
                sentences.append(paragraph[start : end.start()])
                start = end.end()
        sentences.append(paragraph[start:])
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _quotations(sentence: str) -> list[str]:
    return [straight or curly for straight, curly in _QUOTATION.findall(sentence)]


def _occurs(quotation: str, passage: RetrievedChunk) -> bool:
    text = _normal(passage.chunk.text)
    parts = [_normal(part.strip(_END_PUNCTUATION)) for part in _GAP.split(quotation)]
    return all(part in text for part in parts if part)


def _normal(text: str) -> str:
    return " ".join(text.translate(_APOSTROPHES).casefold().split())
