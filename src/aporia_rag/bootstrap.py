"""The composition root: the one place that turns settings into concrete objects.

Everything else depends on the ports in `domain.ports`; only this module
knows that "ollama" means `OllamaEmbedder` or `OllamaChatModel` and "chroma"
means `ChromaVectorStore`. Entry points (the CLI now; a UI later) get their
components from a `Container` instead of wiring them themselves.

Providers are looked up by name in registries, so adding one (open/closed
principle: extend, don't modify) is:

1. Write the adapter, e.g. `adapters/google.py` with a `GoogleEmbedder`
   that extends `BatchingEmbedder`.
2. Register a factory for it at the bottom of this module:

       def _google_embedder(settings, transport):
           return GoogleEmbedder(...settings.embedding...)

       EMBEDDERS.register("google", _google_embedder)

3. Set `APORIA_EMBEDDING__PROVIDER=google`.

No builder, no caller and no test of the indexer or retriever changes.
"""

from collections.abc import Callable
from contextlib import ExitStack
from functools import cached_property
from typing import Protocol, Self, runtime_checkable

import httpx
import ollama

from aporia_rag.adapters.chroma import ChromaVectorStore
from aporia_rag.adapters.ollama import OllamaChatModel, OllamaEmbedder
from aporia_rag.adapters.prompt_formats import prompt_format_for
from aporia_rag.config import Settings
from aporia_rag.domain.errors import ConfigurationError
from aporia_rag.domain.models import EmbeddingSpace
from aporia_rag.domain.ports import ChatModel, Chunker, Embedder, VectorStore
from aporia_rag.generation.generator import Generator
from aporia_rag.grading.relevance import RelevanceGrader
from aporia_rag.graph.corrective import CorrectiveRAG
from aporia_rag.indexing.indexer import Indexer
from aporia_rag.ingestion.chunker import RecursiveChunker
from aporia_rag.retrieval.retriever import Retriever
from aporia_rag.retrieval.rewriter import QueryRewriter

# A factory receives the settings, plus the HTTP transport to use (None: the
# real network). Every HTTP-based provider SDK can take one, which is how
# the tests run whole commands without a network.
type EmbedderFactory = Callable[[Settings, httpx.BaseTransport | None], Embedder]
type ChatModelFactory = Callable[[Settings, httpx.BaseTransport | None], ChatModel]
type VectorStoreFactory = Callable[[Settings, EmbeddingSpace], VectorStore]


class ProviderRegistry[F]:
    """Named factories of one kind ("embedding", "vector store"...)."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._factories: dict[str, F] = {}

    def register(self, name: str, factory: F) -> None:
        if name in self._factories:
            raise ValueError(f"{self._kind} provider {name!r} is already registered")
        self._factories[name] = factory

    def get(self, name: str) -> F:
        try:
            return self._factories[name]
        except KeyError:
            raise ConfigurationError(
                f"unknown {self._kind} provider {name!r}; available: {', '.join(self.names())}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._factories)


EMBEDDERS = ProviderRegistry[EmbedderFactory]("embedding")
CHAT_MODELS = ProviderRegistry[ChatModelFactory]("chat")
VECTOR_STORES = ProviderRegistry[VectorStoreFactory]("vector store")


def build_embedder(
    settings: Settings,
    *,
    transport: httpx.BaseTransport | None = None,
    registry: ProviderRegistry[EmbedderFactory] = EMBEDDERS,
) -> Embedder:
    return registry.get(settings.embedding.provider)(settings, transport)


def build_chat_model(
    settings: Settings, *, transport: httpx.BaseTransport | None = None
) -> ChatModel:
    return CHAT_MODELS.get(settings.chat.provider)(settings, transport)


def build_vector_store(settings: Settings, space: EmbeddingSpace) -> VectorStore:
    return VECTOR_STORES.get(settings.storage.provider)(settings, space)


def build_chunker(settings: Settings) -> Chunker:
    return RecursiveChunker(
        chunk_size=settings.chunking.size, chunk_overlap=settings.chunking.overlap
    )


@runtime_checkable
class _Closeable(Protocol):
    def close(self) -> None: ...


class Container:
    """The application's components, built from the settings once, on first use.

    Lazy, so `aporia-rag fetch` never creates a model client, `search` never
    creates a chat model, and a misconfigured provider only fails the
    commands that need it. Built once, so the indexer and the retriever share
    one embedder and one store.

    The container owns what it builds: use it as a context manager, and on
    leaving it closes every component that holds resources (an HTTP
    connection pool, say). Closing is not part of the ports: the indexer and
    retriever only borrow components, and only the owner should close them.
    """

    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self.transport = transport  # for every outgoing HTTP request; None: the real network
        self._resources = ExitStack()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the components built so far, newest first. Safe to call twice."""
        self._resources.close()

    @cached_property
    def embedder(self) -> Embedder:
        return self._owned(build_embedder(self.settings, transport=self.transport))

    @cached_property
    def vector_store(self) -> VectorStore:
        return self._owned(build_vector_store(self.settings, self.embedder.space))

    @cached_property
    def chunker(self) -> Chunker:
        return build_chunker(self.settings)

    @cached_property
    def indexer(self) -> Indexer:
        return Indexer(self.chunker, self.embedder, self.vector_store)

    @cached_property
    def retriever(self) -> Retriever:
        return Retriever(self.embedder, self.vector_store)

    @cached_property
    def chat_model(self) -> ChatModel:
        return self._owned(build_chat_model(self.settings, transport=self.transport))

    @cached_property
    def generator(self) -> Generator:
        return Generator(self.chat_model)

    @cached_property
    def relevance_grader(self) -> RelevanceGrader:
        return RelevanceGrader(self.chat_model)

    @cached_property
    def query_rewriter(self) -> QueryRewriter:
        return QueryRewriter(self.chat_model)

    @cached_property
    def corrective_rag(self) -> CorrectiveRAG:
        return CorrectiveRAG(
            self.retriever, self.relevance_grader, self.generator, self.query_rewriter
        )

    def _owned[T](self, component: T) -> T:
        if isinstance(component, _Closeable):
            self._resources.callback(component.close)
        return component


# --------------------------------------------------------------------------
# Built-in providers
# --------------------------------------------------------------------------


def _ollama_client(settings: Settings, transport: httpx.BaseTransport | None) -> ollama.Client:
    return ollama.Client(
        host=settings.ollama.host, timeout=settings.ollama.timeout_seconds, transport=transport
    )


def _ollama_embedder(settings: Settings, transport: httpx.BaseTransport | None) -> Embedder:
    embedding = settings.embedding
    return OllamaEmbedder(
        _ollama_client(settings, transport),
        embedding.model,
        prompt_format=prompt_format_for(embedding.model, instruction=embedding.query_instruction),
        dimensions=embedding.dimensions,
        batch_size=embedding.batch_size,
        keep_alive=settings.ollama.keep_alive,
    )


def _ollama_chat_model(settings: Settings, transport: httpx.BaseTransport | None) -> ChatModel:
    return OllamaChatModel(
        _ollama_client(settings, transport),
        settings.chat.model,
        context_length=settings.chat.context_length,
        keep_alive=settings.ollama.keep_alive,
    )


def _chroma_store(settings: Settings, space: EmbeddingSpace) -> VectorStore:
    return ChromaVectorStore.persistent(settings.storage.index_dir, space)


EMBEDDERS.register("ollama", _ollama_embedder)
CHAT_MODELS.register("ollama", _ollama_chat_model)
VECTOR_STORES.register("chroma", _chroma_store)
