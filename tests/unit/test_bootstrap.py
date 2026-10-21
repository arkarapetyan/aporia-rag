"""Tests for aporia_rag.bootstrap, the composition root (test plan: Component 14).

The composition root is the one place that knows which concrete adapters
exist. These tests pin the extension point: a new provider is a new factory
in a registry, and the builders pick it up by name from the settings.
"""

from pathlib import Path

import httpx
import pytest

from aporia_rag.adapters.chroma import ChromaVectorStore
from aporia_rag.adapters.ollama import OllamaChatModel, OllamaEmbedder
from aporia_rag.adapters.web import DdgsWebSearch
from aporia_rag.bootstrap import (
    CHAT_MODELS,
    EMBEDDERS,
    VECTOR_STORES,
    Container,
    EmbedderFactory,
    ProviderRegistry,
    build_chat_model,
    build_embedder,
    build_vector_store,
)
from aporia_rag.config import (
    ChatSettings,
    EmbeddingSettings,
    OllamaSettings,
    Settings,
    StorageSettings,
    WebSettings,
)
from aporia_rag.domain.errors import ConfigurationError
from aporia_rag.domain.models import (
    ChatMessage,
    Document,
    EmbeddingSpace,
    RetrievedChunk,
    SourceMetadata,
    web_passage,
)
from aporia_rag.domain.ports import Embedder
from aporia_rag.graph.corrective import Scope
from support.builders import make_chunk
from support.fakes import FakeOllamaServer, HashingEmbedder, ScriptedWebSearch


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
    assert "ollama" in CHAT_MODELS.names()
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


def test_builds_an_ollama_chat_model_from_the_settings() -> None:  # 14.9
    server = FakeOllamaServer(models=["llama3.1:8b"], reply="Virtue is a mean [1].")
    settings = Settings(
        _env_file=None,
        ollama=OllamaSettings(host="http://gpu-box:11434", keep_alive="30m"),
        chat=ChatSettings(model="llama3.1:8b", context_length=16384),
    )

    model = build_chat_model(settings, transport=server.transport)
    reply = model.chat([ChatMessage(role="user", content="What is virtue?")], max_tokens=64)

    assert isinstance(model, OllamaChatModel)
    assert reply == "Virtue is a mean [1]."
    [request] = server.requests
    assert (request["model"], request["keep_alive"]) == ("llama3.1:8b", "30m")
    assert request["options"]["num_ctx"] == 16384


def test_unknown_chat_provider_lists_the_available_ones() -> None:  # 14.9
    settings = Settings(_env_file=None, chat=ChatSettings(provider="anthropic"))

    with pytest.raises(ConfigurationError, match=r"chat provider 'anthropic'.*ollama"):
        build_chat_model(settings)


def test_searching_never_builds_a_chat_model(tmp_path: Path) -> None:  # 14.10
    settings = _container_settings(tmp_path, chat=ChatSettings(provider="anthropic"))
    container = Container(settings, transport=FakeOllamaServer().transport)

    _ = container.retriever  # an unusable chat provider is no error for `search`...
    with pytest.raises(ConfigurationError, match="anthropic"):
        _ = container.generator  # ...only for what answers questions


def test_the_generator_and_grader_share_the_containers_chat_model_and_it_is_closed(
    tmp_path: Path,
) -> None:  # 14.10
    server = FakeOllamaServer(reply="Only a good will [1].")
    passage = RetrievedChunk(chunk=make_chunk("kant/first-section#0"), score=0.5)
    question = "Is anything good without qualification?"

    with Container(_container_settings(tmp_path), transport=server.transport) as container:
        assert container.chat_model is container.chat_model
        assert container.generator is container.generator
        assert container.relevance_grader is container.relevance_grader
        assert container.query_rewriter is container.query_rewriter
        kept = container.relevance_grader.relevant(question, [passage])
        answer = container.generator.answer(question, kept)
        assert not server.transport.closed

    assert answer.text == "Only a good will [1]."
    assert len(server.structured_requests) == 1  # the verdict went through the same model
    assert server.transport.closes == 1  # one chat model client; nothing else was built


def test_the_containers_corrective_graph_answers_through_its_components(
    tmp_path: Path,
) -> None:  # 14.11
    server = FakeOllamaServer(reply="Only a good will [1].")
    document = Document(
        id="kant/first-section",
        text="Nothing can be called good without qualification, except a good will.",
        metadata=SourceMetadata(author="Immanuel Kant", title="Groundwork", source_id="test:kant"),
    )

    with Container(_container_settings(tmp_path), transport=server.transport) as container:
        assert container.corrective_rag is container.corrective_rag
        container.indexer.run([document])
        outcome = container.corrective_rag.answer(
            "Is anything good without qualification?", Scope(), k=1, authors=("Immanuel Kant",)
        )

    assert outcome.answer is not None
    assert outcome.answer.text == "Only a good will [1]."
    assert server.transport.closes == 2  # the embedder's client and the chat model's, once each


def test_the_containers_graph_searches_the_web_given_to_the_container(
    tmp_path: Path,
) -> None:  # 14.11, 14.12
    page = web_passage(
        url="https://en.wikipedia.org/wiki/Good_will", title="Good will", text="A good will is..."
    )
    web = ScriptedWebSearch([page])

    def replies(prompt: str) -> str:
        if prompt.startswith("Question: "):  # the rewrite: an author outside the corpus
            return '{"query": "good will", "author": "Ludwig Wittgenstein"}'
        return f'{{"relevant": {str(prompt.startswith("Passage from en.wikipedia.org")).lower()}}}'

    server = FakeOllamaServer(reply="A good will [1].", structured_reply=replies)
    document = Document(
        id="kant/first-section",
        text="Nothing can be called good without qualification, except a good will.",
        metadata=SourceMetadata(author="Immanuel Kant", title="Groundwork", source_id="test:kant"),
    )
    question = "Is anything good without qualification?"

    settings = _container_settings(tmp_path)
    with Container(settings, transport=server.transport, web_search=web) as container:
        assert container.web_search is web
        container.indexer.run([document])
        outcome = container.corrective_rag.answer(
            question, Scope(), k=1, authors=("Immanuel Kant",)
        )

    assert web.calls == [(question, 1)]
    assert outcome.answer is not None
    assert [source.chunk for source in outcome.answer.sources] == [page]


def test_builds_the_ddgs_web_search_by_default_once_and_only_when_used(
    tmp_path: Path,
) -> None:  # 14.12
    unusable = Container(_container_settings(tmp_path, web=WebSettings(provider="tavily")))
    _ = unusable.retriever  # an unusable web search is no error until something needs it
    with pytest.raises(ConfigurationError, match=r"web search provider 'tavily'.*ddgs"):
        _ = unusable.web_search

    container = Container(_container_settings(tmp_path))

    assert isinstance(container.web_search, DdgsWebSearch)
    assert container.web_search is container.web_search
