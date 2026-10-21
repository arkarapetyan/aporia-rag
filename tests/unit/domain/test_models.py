"""Tests for aporia_rag.domain.models (test plan: Component 1, and 26.1).

Only our own constraints are tested here (provenance completeness, span
consistency, immutability), not pydantic's general behaviour.
"""

import pytest
from pydantic import ValidationError

from aporia_rag.domain.models import (
    ChatMessage,
    Chunk,
    Document,
    EmbeddingSpace,
    Passage,
    SourceMetadata,
    web_passage,
)

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

    def test_translator_is_optional(self) -> None:  # 1.10
        assert METADATA.translator is None
        translated = METADATA.model_copy(update={"translator": "Benjamin Jowett"})
        assert translated.translator == "Benjamin Jowett"

    def test_citation_names_author_title_and_section(self) -> None:  # 1.16
        assert METADATA.citation == "Plato, The Republic"
        with_section = METADATA.model_copy(update={"section": "Book I"})
        assert with_section.citation == "Plato, The Republic, Book I"

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


class TestPassage:
    def test_title_is_optional(self) -> None:  # 1.11
        assert Passage(text="Know thyself.").title is None

    @pytest.mark.parametrize(
        "fields", [{"text": ""}, {"text": "Know thyself.", "title": ""}], ids=["text", "title"]
    )
    def test_rejects_empty_text_or_title(self, fields: dict[str, str]) -> None:  # 1.11
        with pytest.raises(ValidationError):
            Passage.model_validate(fields)


class TestEmbeddingSpace:
    def test_id_names_every_part_of_the_space(self) -> None:  # 1.12
        space = EmbeddingSpace(
            provider="ollama", model="qwen3-embedding:0.6b", dimensions=512, prompt_format="qwen"
        )

        assert space.id == "ollama/qwen3-embedding:0.6b/512/qwen"

    def test_id_says_native_when_the_size_is_the_models_own(self) -> None:  # 1.12
        space = EmbeddingSpace(provider="p", model="m", dimensions=None, prompt_format="f")

        assert space.id == "p/m/native/f"

    def test_rejects_non_positive_dimensions(self) -> None:  # 1.12
        with pytest.raises(ValidationError, match="dimensions"):
            EmbeddingSpace(provider="p", model="m", dimensions=0, prompt_format="f")


class TestChatMessage:
    @pytest.mark.parametrize("role", ["system", "user"])
    def test_roles_are_system_and_user(self, role: str) -> None:  # 1.18
        assert ChatMessage.model_validate({"role": role, "content": "Hello."}).role == role

    @pytest.mark.parametrize(
        "fields",
        [{"role": "assistant", "content": "Hello."}, {"role": "user", "content": ""}],
        ids=["unknown-role", "empty-content"],
    )
    def test_rejects_other_roles_and_empty_content(self, fields: dict[str, str]) -> None:  # 1.18
        with pytest.raises(ValidationError):
            ChatMessage.model_validate(fields)


class TestWebPassage:
    URL = "https://www.britannica.com/topic/language-game"

    def test_a_web_page_is_cited_by_site_and_title_and_identified_by_its_url(self) -> None:  # 26.1
        page = web_passage(url=self.URL, title="Language game", text="  A language game is...\n")

        assert page.metadata.citation == "britannica.com, Language game"
        assert page.metadata.source_id == self.URL
        assert page.id == page.document_id == self.URL
        assert page.text == "A language game is..."
        assert (page.index, page.start_char, page.end_char) == (0, 0, len(page.text))

    def test_a_long_page_is_cut_at_a_word_before_the_limit(self) -> None:  # 26.1
        words = "word " * 400  # 2,000 characters
        text = f"{words[:1998]}xyz more"  # character 2,000 falls inside "worxyz"

        page = web_passage(url=self.URL, title="Long", text=text)

        assert page.text == words[:1994]  # "worxyz" is dropped whole, with the space before it
        assert text.startswith(page.text)
