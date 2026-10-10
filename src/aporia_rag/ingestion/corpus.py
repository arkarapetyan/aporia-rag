"""The corpus manifest: which works to ingest and where they come from.

The list itself lives in ``corpus.toml`` next to this module, so changing the
corpus is a data change rather than a code change, and it is validated
every time it is loaded.
"""

import re
import tomllib
from importlib.resources import files
from pathlib import Path
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_SLUG: Final = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"


class EditionLayout(BaseModel):
    """Where the author's own text starts and ends in one edition, and how its
    sections are headed. Every pattern must match a *whole* paragraph."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: str = Field(
        description="First paragraph of the author's text; must be a section heading."
    )
    end: str | None = Field(
        default=None, description="First paragraph after the author's text (notes, index...)."
    )
    sections: list[str] = Field(
        min_length=1, max_length=2, description="Heading patterns, outermost level first."
    )
    section_labels: list[str] | None = Field(
        default=None,
        description="Labels for the level-1 headings, in order (when headings read badly).",
    )
    english_only: bool = Field(
        default=False, description="Drop non-English paragraphs (for bilingual editions)."
    )

    @model_validator(mode="after")
    def _patterns_compile(self) -> Self:
        for pattern in (self.start, self.end, *self.sections):
            if pattern is not None:
                _require_regex(pattern)
        return self


class WorkSpec(BaseModel):
    """One work to ingest, identified by its Project Gutenberg ebook number."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=_SLUG, description="Stable slug, the prefix of every Document id.")
    author: str = Field(min_length=1)
    title: str = Field(min_length=1)
    translator: str | None = None
    gutenberg_id: int = Field(gt=0)
    layout: EditionLayout


class CorpusManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    works: list[WorkSpec]

    @model_validator(mode="after")
    def _ids_are_unique(self) -> Self:
        _reject_duplicates("work id", [work.id for work in self.works])
        _reject_duplicates("gutenberg_id", [work.gutenberg_id for work in self.works])
        return self


def load_manifest(path: Path | None = None) -> CorpusManifest:
    """Load and validate a manifest; defaults to the packaged ``corpus.toml``."""
    if path is None:
        raw = files("aporia_rag.ingestion").joinpath("corpus.toml").read_text(encoding="utf-8")
    else:
        raw = path.read_text(encoding="utf-8")
    return CorpusManifest.model_validate(tomllib.loads(raw))


def _require_regex(pattern: str) -> None:
    try:
        re.compile(pattern)
    except re.error as error:
        raise ValueError(f"invalid regular expression {pattern!r}: {error}") from error


def _reject_duplicates(label: str, values: list[str] | list[int]) -> None:
    seen: set[str | int] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {label}: {value}")
        seen.add(value)
