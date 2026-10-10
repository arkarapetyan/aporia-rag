"""Turning the cached raw texts into Documents, one section at a time."""

import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path

from aporia_rag.domain.errors import CorpusError
from aporia_rag.domain.models import Document, SourceMetadata
from aporia_rag.ingestion.cleaning import clean_gutenberg_text
from aporia_rag.ingestion.corpus import WorkSpec
from aporia_rag.ingestion.gutenberg import cache_path
from aporia_rag.ingestion.structure import StructureError, structure_work


def iter_documents(works: Iterable[WorkSpec], cache_dir: Path) -> Iterator[Document]:
    """Yield one Document per section of each work, reading one file at a time.

    This is a generator, so nothing is read until the consumer asks for it and
    memory is bounded by the largest single work, never the whole corpus.
    """
    for work in works:
        yield from _work_documents(work, cache_dir)


def _work_documents(work: WorkSpec, cache_dir: Path) -> Iterator[Document]:
    path = cache_path(cache_dir, work.gutenberg_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"{work.id}: {path} not found; run `aporia-rag fetch` to download the corpus"
        )
    raw = path.read_bytes().decode("utf-8")
    try:
        sections = structure_work(clean_gutenberg_text(raw), work.layout)
    except CorpusError as error:
        raise CorpusError(f"{work.id}: {path} is not the expected edition: {error}") from error
    seen: set[str] = set()
    for section in sections:
        document_id = f"{work.id}/{_slug(section.label)}"
        if document_id in seen:
            raise StructureError(f"duplicate document id {document_id!r} in {work.id}")
        seen.add(document_id)
        yield Document(
            id=document_id,
            text=section.text,
            metadata=SourceMetadata(
                author=work.author,
                title=work.title,
                translator=work.translator,
                source_id=f"gutenberg:{work.gutenberg_id}",
                section=section.label,
            ),
        )


def _slug(label: str) -> str:
    ascii_label = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_label.lower()).strip("-")
