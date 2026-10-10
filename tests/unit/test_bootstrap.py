"""Tests for aporia_rag.bootstrap, the composition root (test plan: Component 14).

The composition root is the one place that knows which concrete adapters
exist. These tests pin the extension point: a new provider is a new factory
in a registry, and the builders pick it up by name from the settings.
"""

from pathlib import Path

import httpx
import pytest

from aporia_rag.adapters.chroma import ChromaVectorStore
from aporia_rag.adapters.ollama import OllamaEmbedder
from aporia_rag.bootstrap import (
    EMBEDDERS,
    VECTOR_STORES,
    Container,
    EmbedderFactory,
    ProviderRegistry,
    build_embedder,
    build_vector_store,
)
from aporia_rag.config import EmbeddingSettings, OllamaSettings, Settings, StorageSettings
from aporia_rag.domain.errors import ConfigurationError
from aporia_rag.domain.models import Document, EmbeddingSpace, SourceMetadata
from aporia_rag.domain.ports import Embedder
from support.fakes import FakeOllamaServer, HashingEmbedder


def test_builds_an_ollama_embedder_from_the_settings() -> None:  # 14.1
    server = FakeOllamaServer(models=["embeddinggemma:300m"], dimensions=128)
    settings = Settings(
        _env_file=None,
        ollama=OllamaSettings(host="http://gpu-box:11434", keep_alive="30m"),
        embedding=EmbeddingSettings(model="embeddinggemma:300m", dimensions=128),
    )

    embedder = build_embedder(settings, transport=server.transport)
    vector = embedder.embed_query("What is virtue?")

    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.space == EmbeddingSpace(
        provider="ollama", model="embeddinggemma:300m", dimensions=128, prompt_format="gemma-task"
    )
    assert len(vector) == 128
    [request] = server.requests
    assert request["input"] == ["task: search result | query: What is virtue?"]
    assert request["keep_alive"] == "30m"


def test_query_instruction_from_the_settings_is_used() -> None:  # 14.1
    server = FakeOllamaServer()
    settings = Settings(
        _env_file=None, embedding=EmbeddingSettings(query_instruction="Find the passage")
    )

    build_embedder(settings, transport=server.transport).embed_query("q")

    assert server.requests[0]["input"] == ["Instruct: Find the passage\nQuery:q"]


def test_unknown_embedding_provider_lists_the_available_ones() -> None:  # 14.2
    settings = Settings(_env_file=None, embedding=EmbeddingSettings(provider="voyage"))

    with pytest.raises(ConfigurationError, match=r"embedding provider 'voyage'.*ollama"):
        build_embedder(settings)


def test_unknown_vector_store_provider_lists_the_available_ones(tmp_path: Path) -> None:  # 14.2
    settings = Settings(
        _env_file=None, storage=StorageSettings(provider="qdrant", index_dir=tmp_path / "index")
    )

    with pytest.raises(ConfigurationError, match=r"vector store provider 'qdrant'.*chroma"):
        build_vector_store(settings, HashingEmbedder().space)
    assert not (tmp_path / "index").exists()


def test_a_registered_provider_is_built_without_changing_the_builders() -> None:  # 14.3
    registry = ProviderRegistry[EmbedderFactory]("embedding")
    built: list[Embedder] = []

    def hashing_factory(settings: Settings, transport: httpx.BaseTransport | None) -> Embedder:
        embedder = HashingEmbedder(model=settings.embedding.model)
        built.append(embedder)
        return embedder

    registry.register("hashing", hashing_factory)
    settings = Settings(
        _env_file=None, embedding=EmbeddingSettings(provider="hashing", model="toy")
    )

    embedder = build_embedder(settings, registry=registry)

    assert built == [embedder]
    assert embedder.space.model == "toy"
    assert registry.names() == ["hashing"]


def test_registering_a_provider_name_twice_is_an_error() -> None:  # 14.4
    registry = ProviderRegistry[EmbedderFactory]("embedding")
    registry.register("ollama", EMBEDDERS.get("ollama"))

    with pytest.raises(ValueError, match="already registered"):
        registry.register("ollama", EMBEDDERS.get("ollama"))


def test_default_registries_know_the_local_providers() -> None:  # 14.4
    assert "ollama" in EMBEDDERS.names()
    assert "chroma" in VECTOR_STORES.names()


def test_builds_a_chroma_store_in_the_configured_directory(tmp_path: Path) -> None:  # 14.5
    settings = Settings(_env_file=None, storage=StorageSettings(index_dir=tmp_path / "index"))
    space = HashingEmbedder().space

    store = build_vector_store(settings, space)

    assert isinstance(store, ChromaVectorStore)
    assert store.space == space
    assert store.count() == 0
    assert (tmp_path / "index").is_dir()


def _container_settings(tmp_path: Path, **sections: object) -> Settings:
    return Settings.model_validate(
        {"storage": StorageSettings(index_dir=tmp_path / "index")} | sections
    )


def test_container_builds_nothing_until_asked(tmp_path: Path) -> None:  # 14.6
    settings = _container_settings(tmp_path, embedding=EmbeddingSettings(provider="voyage"))

    container = Container(settings)  # an unusable provider is not an error yet...

    assert not (tmp_path / "index").exists()
    with pytest.raises(ConfigurationError, match="voyage"):
        _ = container.embedder  # ...only once something needs it


def test_container_builds_each_component_once(tmp_path: Path) -> None:  # 14.6
    container = Container(_container_settings(tmp_path), transport=FakeOllamaServer().transport)

    assert container.embedder is container.embedder
    assert container.vector_store is container.vector_store
    assert container.chunker is container.chunker
    assert container.indexer is container.indexer
    assert container.retriever is container.retriever
    assert container.vector_store.space == container.embedder.space


def test_index_built_by_the_container_is_searchable_through_it(tmp_path: Path) -> None:  # 14.7
    container = Container(_container_settings(tmp_path), transport=FakeOllamaServer().transport)
    document = Document(
        id="kant/first-section",
        text="Nothing can be called good without qualification, except a good will.",
        metadata=SourceMetadata(author="Immanuel Kant", title="Groundwork", source_id="test:kant"),
    )

    container.indexer.run([document])
    [result] = container.retriever.search("good will", k=1)

    assert result.chunk.document_id == "kant/first-section"


def test_leaving_the_container_closes_what_it_built(tmp_path: Path) -> None:  # 14.8
    server = FakeOllamaServer()

    with Container(_container_settings(tmp_path), transport=server.transport) as container:
        container.embedder.embed_query("q")
        assert not server.transport.closed

    assert server.transport.closed


def test_the_container_closes_even_when_the_block_raises(tmp_path: Path) -> None:  # 14.8
    server = FakeOllamaServer()

    def use_then_fail() -> None:
        with Container(_container_settings(tmp_path), transport=server.transport) as container:
            _ = container.embedder
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        use_then_fail()

    assert server.transport.closed


def test_a_container_that_built_nothing_closes_nothing(tmp_path: Path) -> None:  # 14.8
    server = FakeOllamaServer()

    with Container(_container_settings(tmp_path), transport=server.transport):
        pass  # e.g. `fetch`: closing must not build the embedder just to close it

    assert not server.transport.closed
