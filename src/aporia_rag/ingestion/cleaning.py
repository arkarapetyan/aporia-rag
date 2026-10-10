"""Turning a raw Project Gutenberg plain-text file into clean prose.

Two independent steps:

* `extract_body` cuts away Project Gutenberg's header and license, which
  would otherwise be indexed and retrieved as if Plato had written them.
* `normalize_text` undoes plain-text typesetting: lines hard-wrapped at ~70
  characters are joined, so each paragraph becomes one line, and
  ``_underscore_`` italics markers are dropped.

Edition-specific clean-up (translator introductions, footnotes, section
headings) needs the real files and is handled separately.
"""

import re
from typing import Final

from aporia_rag.domain.errors import CorpusError

_START: Final = re.compile(
    r"^\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG E-?BOOK\b.*$",
    re.IGNORECASE | re.MULTILINE,
)
_END: Final = re.compile(
    r"^\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG E-?BOOK\b.*$",
    re.IGNORECASE | re.MULTILINE,
)
# Older files repeat the end in prose just before the marker.
_OLD_END_LINE: Final = re.compile(
    r"^End of (?:the )?Project Gutenberg.*$", re.IGNORECASE | re.MULTILINE
)
_PARAGRAPH_BREAK: Final = re.compile(r"\n\s*\n")
# `_text_` with no underscores inside and no whitespace just inside the
# markers, not glued to a word on either side (so `snake_case` survives).
_ITALICS: Final = re.compile(r"(?<!\w)_([^_\s](?:[^_]*[^_\s])?)_(?!\w)")


class GutenbergFormatError(CorpusError, ValueError):
    """The text lacks the markers every Project Gutenberg file has."""


def clean_gutenberg_text(raw: str) -> str:
    return normalize_text(extract_body(raw))


def extract_body(raw: str) -> str:
    """Return the text between the START and END markers, without the boilerplate."""
    start = _START.search(raw)
    if start is None:
        raise GutenbergFormatError("no '*** START OF THE PROJECT GUTENBERG EBOOK' marker found")
    end = _END.search(raw, start.end())
    if end is None:
        raise GutenbergFormatError(
            "no '*** END OF THE PROJECT GUTENBERG EBOOK' marker after the start"
        )
    body = raw[start.end() : end.start()]
    return _OLD_END_LINE.sub("", body).strip()


def normalize_text(text: str) -> str:
    """One line per paragraph, paragraphs separated by one blank line."""
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = (" ".join(block.split()) for block in _PARAGRAPH_BREAK.split(text))
    return "\n\n".join(_remove_italics(paragraph) for paragraph in paragraphs if paragraph)


def _remove_italics(paragraph: str) -> str:
    # Repeat until nothing changes: removing an inner pair can turn an outer
    # pair into a valid one (`_a _b_ c_`), and the result must be idempotent.
    while (unwrapped := _ITALICS.sub(r"\1", paragraph)) != paragraph:
        paragraph = unwrapped
    return paragraph
