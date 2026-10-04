"""Tests for aporia_rag.domain.models (test plan: Component 1).

Only our own constraints are tested here (provenance completeness, span
consistency, immutability), not pydantic's general behaviour.
"""

import pytest
from pydantic import ValidationError

from aporia_rag.domain.models import Chunk, Document, SourceMetadata

METADATA = SourceMetadata(author="Plato", title="The Republic", source_id="gutenberg:1497")


def _chunk(**overrides: object) -> Chunk:
    fields: dict[str, object] = {
        "id": "republic#0",
        "document_id": "republic",
        "index": 0,
        "text": "Justice",
        "start_char": 10,
        "end_char": 17,
        "metadata": METADATA,
    }
    return Chunk.model_validate(fields | overrides)


class TestSourceMetadata:
    def test_section_is_optional(self) -> None:  # 1.1
        assert METADATA.section is None

    def test_rejects_unknown_fields(self) -> None:  # 1.2
        with pytest.raises(ValidationError, match="athor"):
            SourceMetadata.model_validate(
                {"athor": "Plato", "author": "Plato", "title": "Republic", "source_id": "g:1"}
            )

    @pytest.mark.parametrize("blank_field", ["author", "title"])
    def test_rejects_blank_author_or_title(self, blank_field: str) -> None:  # 1.3
        fields = {"author": "Plato", "title": "Republic", "source_id": "g:1", blank_field: ""}
        with pytest.raises(ValidationError, match=blank_field):
            SourceMetadata.model_validate(fields)


class TestDocument:
    def test_rejects_empty_id(self) -> None:  # 1.4
        with pytest.raises(ValidationError, match="id"):
            Document(id="", text="Some text.", metadata=METADATA)

    def test_is_immutable(self) -> None:  # 1.5
        document = Document(id="republic", text="Some text.", metadata=METADATA)
        with pytest.raises(ValidationError, match="frozen"):
            document.text = "Tampered."  # type: ignore[misc]


class TestChunk:
    def test_accepts_span_matching_text_length(self) -> None:  # 1.6
        chunk = _chunk()
        assert chunk.end_char - chunk.start_char == len(chunk.text)

    def test_rejects_span_that_disagrees_with_text_length(self) -> None:  # 1.7
        with pytest.raises(ValidationError, match="span"):
            _chunk(end_char=99)

    def test_rejects_empty_text(self) -> None:  # 1.8
        with pytest.raises(ValidationError, match="text"):
            _chunk(text="", start_char=0, end_char=0)

    @pytest.mark.parametrize(
        "overrides",
        [{"index": -1}, {"start_char": -1, "end_char": 6}],
        ids=["negative-index", "negative-start"],
    )
    def test_rejects_negative_positions(self, overrides: dict[str, int]) -> None:  # 1.9
        with pytest.raises(ValidationError):
            _chunk(**overrides)
