"""Bringing the vector index in line with the corpus, embedding as little as possible.

Embedding is the slow part of indexing (minutes for the whole corpus on a
laptop), so a run only embeds chunks that are new or whose content changed:

1. Read the fingerprint of every chunk already stored.
2. Chunk every document. A chunk whose id is stored with the same
   fingerprint is left alone; the others are embedded and upserted in
   groups of `flush_size`.
3. Delete stored chunks that no longer exist (a section was cut, a
   document got shorter).

So a second run on an unchanged corpus embeds nothing, editing one
paragraph re-embeds about one chunk, and a run interrupted halfway resumes
where it stopped: what was flushed is stored and will count as unchanged.

The index mirrors exactly the documents given to `run`: anything stored
that is not among them is deleted. Always pass the whole corpus.
"""

import hashlib
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import groupby

from aporia_rag.domain.models import Chunk, Document, Passage
from aporia_rag.domain.ports import (
    Chunker,
    Embedder,
    IndexRecord,
    VectorStore,
    check_same_space,
)


@dataclass(frozen=True, slots=True)
class IndexReport:
    added: int
    updated: int
    unchanged: int
    deleted: int
    seconds: float


@dataclass(frozen=True, slots=True)
class WorkProgress:
    """Sent to `on_progress` when one work has been indexed."""

    source_id: str
    author: str
    title: str
    documents: int
    chunks: int


def citation_passage(chunk: Chunk) -> Passage:
    """What gets embedded for a chunk: its text, titled with its citation.

    The title gives the embedding the context the text itself may lack (a
    paragraph from the middle of Book IV rarely says it is Plato on
    justice). The stored chunk keeps only the source text, so citations and
    character offsets stay exact.
    """
    return Passage(title=chunk.metadata.citation, text=chunk.text)


def fingerprint(passage: Passage) -> str:
    """A short hash of exactly what is embedded: changes iff the title or text does.

    The separator keeps ("ab", "c") and ("a", "bc") apart.
    """
    content = f"{passage.title or ''}\x00{passage.text}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class _RunState:
    stored: dict[str, str]
    seen: set[str] = field(default_factory=set)
    pending: list[tuple[Chunk, Passage, str]] = field(default_factory=list)
    added: int = 0
    updated: int = 0
    unchanged: int = 0


def _work_key(document: Document) -> tuple[str, str, str]:
    metadata = document.metadata
    return metadata.source_id, metadata.author, metadata.title


class Indexer:
    """Chunks documents, embeds what changed, and keeps the store in step."""

    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
        *,
        flush_size: int = 256,
    ) -> None:
        check_same_space(embedder, store)
        if flush_size <= 0:
            raise ValueError(f"flush_size must be positive, got {flush_size}")
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._flush_size = flush_size

    def run(
        self,
        documents: Iterable[Document],
        *,
        on_progress: Callable[[WorkProgress], None] | None = None,
    ) -> IndexReport:
        """Index `documents` (the whole corpus) and report what changed.

        Documents are consumed lazily, one work at a time.
        """
        started = time.perf_counter()
        state = _RunState(stored=self._store.fingerprints())
        for (source_id, author, title), work in groupby(documents, key=_work_key):
            document_count = chunk_count = 0
            for document in work:
                document_count += 1
                chunk_count += self._add(document, state)
            self._flush(state)  # a reported work is a stored work
            if on_progress is not None:
                on_progress(WorkProgress(source_id, author, title, document_count, chunk_count))
        stale = state.stored.keys() - state.seen
        self._store.delete(stale)
        return IndexReport(
            added=state.added,
            updated=state.updated,
            unchanged=state.unchanged,
            deleted=len(stale),
            seconds=time.perf_counter() - started,
        )

    def _add(self, document: Document, state: _RunState) -> int:
        chunks = self._chunker.chunk(document)
        for chunk in chunks:
            if chunk.id in state.seen:
                raise ValueError(f"two chunks have the id {chunk.id!r}")
            state.seen.add(chunk.id)
            passage = citation_passage(chunk)
            current = fingerprint(passage)
            previous = state.stored.get(chunk.id)
            if previous == current:
                state.unchanged += 1
                continue
            if previous is None:
                state.added += 1
            else:
                state.updated += 1
            state.pending.append((chunk, passage, current))
            if len(state.pending) >= self._flush_size:
                self._flush(state)
        return len(chunks)

    def _flush(self, state: _RunState) -> None:
        if not state.pending:
            return
        vectors = self._embedder.embed_documents([passage for _, passage, _ in state.pending])
        self._store.upsert(
            [
                IndexRecord(chunk=chunk, vector=vector, fingerprint=print_)
                for (chunk, _, print_), vector in zip(state.pending, vectors, strict=True)
            ]
        )
        state.pending.clear()
