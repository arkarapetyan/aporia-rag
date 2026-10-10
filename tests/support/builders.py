"""Small factories for test data, so each test states only what it cares about."""

from aporia_rag.domain.models import Chunk, SourceMetadata
from aporia_rag.domain.ports import IndexRecord, Vector


def make_chunk(
    chunk_id: str,
    text: str = "Some text.",
    *,
    author: str = "Immanuel Kant",
    title: str = "Groundwork of the Metaphysic of Morals",
    section: str | None = "First Section",
    translator: str | None = None,
    index: int = 0,
    start_char: int = 0,
) -> Chunk:
    document_id = chunk_id.partition("#")[0]
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        index=index,
        text=text,
        start_char=start_char,
        end_char=start_char + len(text),
        metadata=SourceMetadata(
            author=author,
            title=title,
            source_id=f"test:{author.rsplit(maxsplit=1)[-1].lower()}",
            section=section,
            translator=translator,
        ),
    )


def make_record(
    chunk_id: str,
    vector: Vector,
    *,
    fingerprint: str = "fingerprint",
    text: str = "Some text.",
    author: str = "Immanuel Kant",
    section: str | None = "First Section",
) -> IndexRecord:
    return IndexRecord(
        chunk=make_chunk(chunk_id, text, author=author, section=section),
        vector=vector,
        fingerprint=fingerprint,
    )
