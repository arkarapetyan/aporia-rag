"""Tests for aporia_rag.adapters.prompt_formats (test plan: Component 10.1-10.7).

The expected strings are copied from each model's own documentation. They
are part of how the model was trained, so they are tested literally: a
missing space here silently costs retrieval quality, and nothing else would
ever notice.
"""

import pytest

from aporia_rag.adapters.prompt_formats import (
    GemmaTaskFormat,
    NomicPrefixFormat,
    PromptFormat,
    QwenInstructFormat,
    prompt_format_for,
)
from aporia_rag.domain.errors import ConfigurationError
from aporia_rag.domain.models import Passage

QWEN_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
PHILOSOPHY_INSTRUCTION = (
    "Given a question about philosophy, retrieve passages from primary texts that answer it"
)
KANT = Passage(
    title="Immanuel Kant, Groundwork, First Section",
    text="Nothing can be called good without qualification, except a good will.",
)
UNTITLED = Passage(text="Every art and every inquiry aims at some good.")


class TestQwenInstructFormat:
    def test_query_is_prefixed_with_the_documented_default_instruction(self) -> None:  # 10.1
        query_format = QwenInstructFormat()

        assert query_format.query("What is justice?") == (
            f"Instruct: {QWEN_DEFAULT_INSTRUCTION}\nQuery:What is justice?"
        )
        assert query_format.name == "qwen-instruct"

    def test_custom_instruction_is_used_and_changes_the_name(self) -> None:  # 10.2
        custom = QwenInstructFormat(PHILOSOPHY_INSTRUCTION)

        assert custom.query("What is justice?") == (
            f"Instruct: {PHILOSOPHY_INSTRUCTION}\nQuery:What is justice?"
        )
        # The name is part of the embedding space, so a different instruction
        # must give a different, but stable, name.
        assert custom.name.startswith("qwen-instruct-")
        assert custom.name != QwenInstructFormat().name
        assert custom.name == QwenInstructFormat(PHILOSOPHY_INSTRUCTION).name
        assert custom.name != QwenInstructFormat(PHILOSOPHY_INSTRUCTION + ".").name

    def test_blank_instruction_is_rejected(self) -> None:  # 10.2
        with pytest.raises(ConfigurationError, match="instruction"):
            QwenInstructFormat("  ")

    def test_passage_is_its_title_then_its_text(self) -> None:  # 10.3
        query_format = QwenInstructFormat()

        assert query_format.document(KANT) == (
            "Immanuel Kant, Groundwork, First Section\n\n"
            "Nothing can be called good without qualification, except a good will."
        )
        assert query_format.document(UNTITLED) == UNTITLED.text


def test_embeddinggemma_uses_task_and_title_slots() -> None:  # 10.4
    gemma = GemmaTaskFormat()

    assert gemma.query("What is virtue?") == "task: search result | query: What is virtue?"
    assert gemma.document(KANT) == (
        "title: Immanuel Kant, Groundwork, First Section | "
        "text: Nothing can be called good without qualification, except a good will."
    )
    assert gemma.document(UNTITLED) == f"title: none | text: {UNTITLED.text}"
    assert gemma.name == "gemma-task"


def test_embeddinggemma_accepts_another_documented_task() -> None:  # 10.4
    gemma = GemmaTaskFormat("question answering")

    assert gemma.query("What is virtue?") == "task: question answering | query: What is virtue?"
    assert gemma.name not in {"gemma-task", ""}


class TestNomicPrefixFormat:
    def test_uses_the_fixed_task_prefixes(self) -> None:  # 10.5
        nomic = NomicPrefixFormat()

        assert nomic.query("What is virtue?") == "search_query: What is virtue?"
        assert nomic.document(KANT) == (
            "search_document: Immanuel Kant, Groundwork, First Section\n\n"
            "Nothing can be called good without qualification, except a good will."
        )
        assert nomic.document(UNTITLED) == f"search_document: {UNTITLED.text}"
        assert nomic.name == "nomic-prefix"

    def test_custom_instruction_is_rejected(self) -> None:  # 10.5
        with pytest.raises(ConfigurationError, match="nomic-embed-text"):
            prompt_format_for("nomic-embed-text:v1.5", instruction=PHILOSOPHY_INSTRUCTION)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("qwen3-embedding", QwenInstructFormat),
        ("qwen3-embedding:0.6b", QwenInstructFormat),
        ("qwen3-embedding:8b", QwenInstructFormat),
        ("dengcao/Qwen3-Embedding-0.6B:Q8_0", QwenInstructFormat),  # a community upload
        ("embeddinggemma:300m", GemmaTaskFormat),
        ("embeddinggemma-2:440m", GemmaTaskFormat),
        ("nomic-embed-text:v1.5", NomicPrefixFormat),
    ],
)
def test_format_is_chosen_by_model_family(model: str, expected: type[PromptFormat]) -> None:  # 10.6
    assert type(prompt_format_for(model)) is expected


def test_instruction_is_passed_to_the_chosen_format() -> None:  # 10.6
    chosen = prompt_format_for("qwen3-embedding:4b", instruction=PHILOSOPHY_INSTRUCTION)

    assert chosen.query("q") == f"Instruct: {PHILOSOPHY_INSTRUCTION}\nQuery:q"


def test_unknown_model_family_lists_the_known_ones() -> None:  # 10.7
    with pytest.raises(ConfigurationError, match="all-minilm") as raised:
        prompt_format_for("all-minilm:22m")

    for family in ("qwen3-embedding", "embeddinggemma", "nomic-embed-text"):
        assert family in str(raised.value)
