"""Tests for aporia_rag.adapters.ollama (test plan: 10.14-10.19).

The adapter talks to a real `ollama.Client`; only the network underneath it
is replaced, by `FakeOllamaServer` behind an `httpx.MockTransport`. So these
tests also check that we use the client library the way it expects.
"""

import httpx
import ollama
import pytest

from aporia_rag.adapters.ollama import OllamaEmbedder
from aporia_rag.adapters.prompt_formats import QwenInstructFormat
from aporia_rag.domain.errors import (
    ModelNotAvailableError,
    ProviderError,
    ProviderUnavailableError,
)
from aporia_rag.domain.models import EmbeddingSpace, Passage
from support.fakes import FakeOllamaServer, hash_vector, refusing_transport

MODEL = "qwen3-embedding:0.6b"
QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)


def _embedder(
    transport: httpx.BaseTransport,
    *,
    dimensions: int | None = None,
    keep_alive: str | None = None,
    batch_size: int = 32,
) -> OllamaEmbedder:
    client = ollama.Client(host="http://ollama.test:11434", transport=transport)
    return OllamaEmbedder(
        client,
        MODEL,
        prompt_format=QwenInstructFormat(),
        dimensions=dimensions,
        keep_alive=keep_alive,
        batch_size=batch_size,
    )


def test_sends_formatted_inputs_without_truncation() -> None:  # 10.14
    server = FakeOllamaServer(models=[MODEL])
    embedder = _embedder(server.transport, keep_alive="10m")

    embedder.embed_documents([Passage(title="Kant", text="The good will."), Passage(text="War.")])
    embedder.embed_query("What is the good will?")

    passages_request, query_request = server.requests
    assert passages_request == {
        "model": MODEL,
        "input": ["Kant\n\nThe good will.", "War."],
        "truncate": False,  # an over-long chunk must fail loudly, not be cut silently
        "keep_alive": "10m",
    }
    assert query_request["input"] == [f"{QUERY_PREFIX}What is the good will?"]


def test_dimensions_and_keep_alive_are_sent_only_when_configured() -> None:  # 10.14
    server = FakeOllamaServer(models=[MODEL], dimensions=32)

    _embedder(server.transport, dimensions=32).embed_query("q")
    _embedder(server.transport).embed_query("q")

    with_dimensions, without_dimensions = server.requests
    assert with_dimensions["dimensions"] == 32
    assert "dimensions" not in without_dimensions
    assert "keep_alive" not in without_dimensions


def test_returns_the_servers_vectors_in_order() -> None:  # 10.15
    server = FakeOllamaServer(models=[MODEL], dimensions=64)
    embedder = _embedder(server.transport, batch_size=2)

    vectors = embedder.embed_documents([Passage(text=text) for text in ("one", "two", "three")])
    query_vector = embedder.embed_query("four")

    assert vectors == [hash_vector(text, 64) for text in ("one", "two", "three")]
    assert query_vector == hash_vector(f"{QUERY_PREFIX}four", 64)
    assert [len(request["input"]) for request in server.requests] == [2, 1, 1]


def test_model_not_pulled_says_how_to_pull_it() -> None:  # 10.16
    server = FakeOllamaServer(models=["some-other-model"])

    with pytest.raises(ModelNotAvailableError, match=f"ollama pull {MODEL}"):
        _embedder(server.transport).embed_query("q")


def test_ollama_not_running_is_reported_as_unavailable() -> None:  # 10.17
    with pytest.raises(ProviderUnavailableError, match="Ollama"):
        _embedder(refusing_transport()).embed_query("q")


def _timing_out_transport() -> httpx.MockTransport:
    def time_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    return httpx.MockTransport(time_out)


@pytest.mark.parametrize(
    ("transport", "message"),
    [
        (FakeOllamaServer(status=500).transport, "simulated server failure"),
        (_timing_out_transport(), "timed out"),
    ],
    ids=["server-error", "timeout"],
)
def test_other_failures_are_provider_errors(
    transport: httpx.BaseTransport, message: str
) -> None:  # 10.18
    with pytest.raises(ProviderError, match=message) as raised:
        _embedder(transport).embed_query("q")

    assert type(raised.value) is ProviderError  # not one of the more specific subclasses


def test_embedding_space_identifies_provider_model_size_and_format() -> None:  # 10.19
    transport = FakeOllamaServer().transport

    assert _embedder(transport).space == EmbeddingSpace(
        provider="ollama", model=MODEL, dimensions=None, prompt_format="qwen-instruct"
    )
    assert _embedder(transport, dimensions=256).space.dimensions == 256


def test_close_releases_the_connections() -> None:  # 10.20
    server = FakeOllamaServer()
    embedder = _embedder(server.transport)
    embedder.embed_query("q")

    embedder.close()

    assert server.transport.closed
