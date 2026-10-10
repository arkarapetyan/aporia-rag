"""How each embedding model family wants queries and passages written.

Retrieval embedding models are trained with specific text templates, and
using the wrong one (or none) silently costs retrieval quality. The template
belongs to the model family, not to the provider serving it: EmbeddingGemma
uses the same template whether it runs in Ollama or behind Google's API. So
formats are Strategy objects that any provider adapter can use.

Sources for each template are the model cards:
- Qwen3-Embedding: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- EmbeddingGemma: https://ai.google.dev/gemma/docs/embeddinggemma/model_card
- Nomic Embed: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5

The format's `name` is part of the EmbeddingSpace: change the template and
existing vectors no longer match new queries, so the index must be rebuilt.
"""

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Final, override

from aporia_rag.domain.errors import ConfigurationError
from aporia_rag.domain.models import Passage


class PromptFormat(ABC):
    """Writes a query, or a titled passage, the way a model family was trained on."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifies the format, including any custom instruction."""

    @abstractmethod
    def query(self, text: str) -> str:
        """The text to embed for a search query."""

    @abstractmethod
    def document(self, passage: Passage) -> str:
        """The text to embed for a passage."""


def _variant(base: str, custom: str | None) -> str:
    """`base` for the default instruction; `base-<hash>` for a custom one."""
    if custom is None:
        return base
    return f"{base}-{hashlib.sha256(custom.encode()).hexdigest()[:8]}"


def _title_then_text(passage: Passage) -> str:
    """For models without a title slot: the title as a first paragraph, if there is one."""
    return passage.text if passage.title is None else f"{passage.title}\n\n{passage.text}"


def _require_text(instruction: str | None, kind: str) -> str | None:
    if instruction is not None and not instruction.strip():
        raise ConfigurationError(f"the query {kind} must not be blank")
    return instruction


class QwenInstructFormat(PromptFormat):
    """Qwen3-Embedding: an instruction before the query; passages as they are.

    The model card's own example writes `Query:` with no space after it,
    and that is what the model saw in training.
    """

    DEFAULT_INSTRUCTION: Final = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    def __init__(self, instruction: str | None = None) -> None:
        self._custom = _require_text(instruction, "instruction")
        self._instruction = self._custom or self.DEFAULT_INSTRUCTION

    @property
    @override
    def name(self) -> str:
        return _variant("qwen-instruct", self._custom)

    @override
    def query(self, text: str) -> str:
        return f"Instruct: {self._instruction}\nQuery:{text}"

    @override
    def document(self, passage: Passage) -> str:
        return _title_then_text(passage)


class GemmaTaskFormat(PromptFormat):
    """EmbeddingGemma (and Google's gemini-embedding-2): task and title slots.

    The task names one of the model card's documented tasks: "search
    result" (the default for retrieval), "question answering", "fact
    checking"...
    """

    DEFAULT_TASK: Final = "search result"

    def __init__(self, task: str | None = None) -> None:
        self._custom = _require_text(task, "task")
        self._task = self._custom or self.DEFAULT_TASK

    @property
    @override
    def name(self) -> str:
        return _variant("gemma-task", self._custom)

    @override
    def query(self, text: str) -> str:
        return f"task: {self._task} | query: {text}"

    @override
    def document(self, passage: Passage) -> str:
        return f"title: {passage.title or 'none'} | text: {passage.text}"


class NomicPrefixFormat(PromptFormat):
    """Nomic Embed: fixed `search_query: ` / `search_document: ` prefixes (required)."""

    def __init__(self, instruction: str | None = None) -> None:
        if instruction is not None:
            raise ConfigurationError(
                "nomic-embed-text uses fixed task prefixes and takes no custom query instruction"
            )

    @property
    @override
    def name(self) -> str:
        return "nomic-prefix"

    @override
    def query(self, text: str) -> str:
        return f"search_query: {text}"

    @override
    def document(self, passage: Passage) -> str:
        return f"search_document: {_title_then_text(passage)}"


# Model family (the start of the model name) -> its format, built from the
# optional custom instruction. Supporting a new family is one entry here.
_FAMILIES: Final[dict[str, Callable[[str | None], PromptFormat]]] = {
    "qwen3-embedding": QwenInstructFormat,
    "embeddinggemma": GemmaTaskFormat,
    "nomic-embed-text": NomicPrefixFormat,
}


def prompt_format_for(model: str, *, instruction: str | None = None) -> PromptFormat:
    """The prompt format for a model, found from its name.

    Works with Ollama names (`qwen3-embedding:8b`), new versions of a family
    (`embeddinggemma-2:440m`) and namespaced uploads
    (`dengcao/Qwen3-Embedding-0.6B:Q8_0`).
    """
    family = model.rsplit("/", 1)[-1].split(":", 1)[0].lower()
    for prefix, build in _FAMILIES.items():
        if family.startswith(prefix):
            return build(instruction)
    raise ConfigurationError(
        f"no prompt format known for embedding model {model!r}; "
        f"known model families: {', '.join(sorted(_FAMILIES))}"
    )
