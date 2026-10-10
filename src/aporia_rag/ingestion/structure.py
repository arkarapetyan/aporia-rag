"""Cutting each edition down to the author's own text and splitting it into sections.

Everything edition-specific (where the author's text starts and ends, how its
sections are headed) comes from the work's `EditionLayout`. The clean-up
rules here are shared: they cover the footnote and apparatus conventions
found across the Gutenberg editions in the corpus.

Input is text already cleaned by `clean_gutenberg_text`: one paragraph per
line, paragraphs separated by a blank line.
"""

import re
import unicodedata
from collections import Counter
from typing import Final, NamedTuple

from aporia_rag.domain.errors import CorpusError
from aporia_rag.ingestion.corpus import EditionLayout


class StructureError(CorpusError, ValueError):
    """An edition does not match its layout."""


class Section(NamedTuple):
    label: str
    text: str


# Whole paragraphs that are never the author's prose.
_APPARATUS: Final = re.compile(
    r"""
      \[(?:\d+|[A-Z]{1,2})\].*   # footnote text: [3] ..., [A] ..., [AB] ...
    | \[Footnote.*               # [Footnote A: ...]
    | \*\s.*                     # * footnote text (Kant)
    | FOOTNOTES?:?               # heading of a block of footnotes
    | \#.*\#                     # margin summary (Cicero)
    | \[Illustration.*\]
    | SEC_\d+                    # stray HTML anchor (Kant)
    | \{\d+\}                    # page number on its own (Seneca)
    """,
    re.VERBOSE | re.DOTALL,
)

# Marks inside a paragraph, removed (or unwrapped) before anything else looks at it.
_INLINE: Final = (
    (re.compile(r"\[(?:\d+|[A-Z]{1,2})\]"), ""),  # word[3], word[A], word[AB]
    (re.compile(r"(?<=[^\s(])\(\d{1,3}\)"), ""),  # word(3); a spaced "(1)" enumeration is kept
    (re.compile(r"\{\d+\}"), ""),  # {12} page numbers
    (re.compile(r"\*\d+\*"), ""),  # *5* section numbers
    (re.compile(r"#[^#]{1,200}#"), ""),  # margin note merged into a paragraph (Cicero)
    (re.compile(r"(?<=\s)\*(?=\s|$)"), ""),  # a lone * footnote reference
    (re.compile(r"\[Greek: ([^\]]*)\]"), r"\1"),  # [Greek: arete] -> arete
)

_HAS_LOWERCASE: Final = re.compile(r"[a-z]")

# Function words for telling English from Latin paragraphs. Words that exist in
# both languages ("in", "a", "his", "me", "do", "at", "an") are left out.
_ENGLISH_WORDS: Final = frozenset(
    """the and of to that it is which be as not for with he this by are but have we from or us
    our you your they them their was were has had will shall would should can may must if so
    what who whom whose when there then than thou thy thee ye my him her its been being does
    did no nor all any one on upon into these those such let""".split()
)
_LATIN_WORDS: Final = frozenset(
    """et est non ad quod ut cum sed quae qui esse enim autem quam nec neque atque etiam sunt
    ab ex de tamen igitur vel nam quidem quid aut sit si ne dum haec hoc id eius ille illud
    illa ipse iam nunc quoque ita sic inquit nihil nos vos mihi tibi se sibi eo ea eum quo qua
    quem quibus ac erat esset fuit eorum earum iste ista""".split()
)

_ROMAN_NUMERAL: Final = re.compile(r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})")
_MINOR_WORDS: Final = frozenset(
    "a an and as at but by for from in into nor of on or the to upon with".split()
)
_WORD: Final = re.compile(r"[A-Za-z][A-Za-z'\u2019]*")
_CLAUSE_START: Final = re.compile(r"(?:^|[.:]\s+\W*)$")


def structure_work(text: str, layout: EditionLayout) -> list[Section]:
    """Split one cleaned edition into labelled sections of the author's own text."""
    paragraphs = text.split("\n\n")
    first, stop = _boundaries(paragraphs, layout)
    headings = [re.compile(pattern) for pattern in layout.sections]
    if _heading_level(_remove_inline_marks(paragraphs[first]), headings) is None:
        raise StructureError(f"start paragraph {paragraphs[first]!r} is not a section heading")

    labels = layout.section_labels
    sections: list[tuple[str, list[str]]] = []
    path: list[str] = []
    level1_headings = 0
    previous_is_english = True
    for raw_paragraph in paragraphs[first:stop]:
        if _APPARATUS.fullmatch(raw_paragraph):
            continue
        paragraph = _remove_inline_marks(raw_paragraph)
        level = _heading_level(paragraph, headings)
        if level is not None:
            label = format_label(paragraph)
            if level == 0:
                if labels is not None and level1_headings < len(labels):
                    label = labels[level1_headings]
                level1_headings += 1
            path = [*path[:level], label]
            sections.append((", ".join(path), []))
            continue
        if not _HAS_LOWERCASE.search(paragraph):
            continue
        if layout.english_only:
            previous_is_english = _is_english(paragraph, previous_is_english)
            if not previous_is_english:
                continue
        sections[-1][1].append(paragraph)

    if labels is not None and len(labels) != level1_headings:
        raise StructureError(
            f"{len(labels)} section labels given for {level1_headings} level-1 headings"
        )
    return [Section(label, "\n\n".join(body)) for label, body in sections if body]


def format_label(heading: str) -> str:
    """Turn a heading into a citation label.

    `CHAPTER XXVI. OF CIVILL LAWES` becomes `Chapter XXVI. Of Civill Lawes`.

    Only all-capital words are re-cased. Valid Roman numerals stay upper case
    (but `CIVILL`, which merely uses numeral letters, does not), and minor
    words are lowercase except at the start of a clause.
    """
    text = heading.strip().rstrip(".")

    def recase(match: re.Match[str]) -> str:
        word = match.group(0)
        if not word.isupper() or _ROMAN_NUMERAL.fullmatch(word):
            return word
        if word.lower() in _MINOR_WORDS and not _CLAUSE_START.search(text, 0, match.start()):
            return word.lower()
        return word[0] + word[1:].lower()

    return _WORD.sub(recase, text)


def _boundaries(paragraphs: list[str], layout: EditionLayout) -> tuple[int, int]:
    first = _find(paragraphs, layout.start, 0)
    if first is None:
        raise StructureError(f"start paragraph {layout.start!r} not found")
    if layout.end is None:
        return first, len(paragraphs)
    stop = _find(paragraphs, layout.end, first + 1)
    if stop is None:
        raise StructureError(f"end paragraph {layout.end!r} not found after the start")
    return first, stop


def _find(paragraphs: list[str], pattern: str, begin: int) -> int | None:
    compiled = re.compile(pattern)
    return next(
        (i for i in range(begin, len(paragraphs)) if compiled.fullmatch(paragraphs[i])), None
    )


def _heading_level(paragraph: str, headings: list[re.Pattern[str]]) -> int | None:
    return next((level for level, h in enumerate(headings) if h.fullmatch(paragraph)), None)


def _remove_inline_marks(paragraph: str) -> str:
    for pattern, replacement in _INLINE:
        paragraph = pattern.sub(replacement, paragraph)
    return " ".join(paragraph.split())


def _is_english(paragraph: str, previous_is_english: bool) -> bool:
    """Majority vote of function words; on a tie (often a short verse line with
    none of either), assume the language of the previous paragraph, since the
    bilingual editions alternate in page-sized blocks of one language."""
    words = Counter(re.findall(r"[a-z]+", _strip_accents(paragraph.lower())))
    english = sum(words[word] for word in _ENGLISH_WORDS)
    latin = sum(words[word] for word in _LATIN_WORDS)
    return previous_is_english if english == latin else english > latin


def _strip_accents(text: str) -> str:
    """`\u00e9st` -> `est`: the Loeb text marks Latin verse stress with accents."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))
