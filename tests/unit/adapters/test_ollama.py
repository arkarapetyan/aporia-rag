"""Tests for aporia_rag.adapters.ollama (test plan: 10.14-10.21 and 18.1-18.10).

The adapter talks to a real `ollama.Client`; only the network underneath it
is replaced, by `FakeOllamaServer` behind an `httpx.MockTransport`. So these
tests also check that we use the client library the way it expects.
"""

import httpx
import ollama
import pytest
from pydantic import BaseModel, Field

from aporia_rag.adapters.ollama import OllamaChatModel, OllamaEmbedder
from aporia_rag.adapters.prompt_formats import QwenInstructFormat
from aporia_rag.config import ChatSettings
from aporia_rag.domain.errors import (
    ModelNotAvailableError,
    ProviderError,
    ProviderUnavailableError,
)
from aporia_rag.domain.models import ChatMessage, EmbeddingSpace, Passage
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


def test_queries_are_embedded_on_the_cpu_and_passages_anywhere() -> None:  # 10.21
    server = FakeOllamaServer(models=[MODEL])
    embedder = _embedder(server.transport)

    embedder.embed_documents([Passage(text="War.")])
    embedder.embed_query("What is war?")

    passages_request, query_request = server.requests
    assert "options" not in passages_request  # indexing has the GPU to itself
    assert query_request["options"] == {"num_gpu": 0}  # a loaded chat model keeps the GPU


def test_close_releases_the_connections() -> None:  # 10.20
    server = FakeOllamaServer()
    embedder = _embedder(server.transport)
    embedder.embed_query("q")

    embedder.close()

    assert server.transport.closed


# --------------------------------------------------------------------------
# Chat model (Component 18)
# --------------------------------------------------------------------------

CHAT_MODEL = ChatSettings().model  # the one FakeOllamaServer serves by default
CONVERSATION = [
    ChatMessage(role="system", content="Answer from the passages."),
    ChatMessage(role="user", content="What is the good will?"),
]


def _chat_model(
    transport: httpx.BaseTransport, *, context_length: int = 4096, keep_alive: str | None = None
) -> OllamaChatModel:
    client = ollama.Client(host="http://ollama.test:11434", transport=transport)
    return OllamaChatModel(client, CHAT_MODEL, context_length=context_length, keep_alive=keep_alive)


def test_chat_sends_the_messages_in_order_with_thinking_off() -> None:  # 18.1
    server = FakeOllamaServer()

    _chat_model(server.transport, keep_alive="10m").chat(CONVERSATION)
    _chat_model(server.transport).chat(CONVERSATION)

    with_keep_alive, without_keep_alive = server.requests
    assert with_keep_alive["model"] == CHAT_MODEL
    assert with_keep_alive["messages"] == [
        {"role": "system", "content": "Answer from the passages."},
        {"role": "user", "content": "What is the good will?"},
    ]
    assert with_keep_alive["think"] is False  # a thinking model answers directly
    assert with_keep_alive["stream"] is False
    assert with_keep_alive["keep_alive"] == "10m"
    assert "keep_alive" not in without_keep_alive


def test_chat_returns_the_reply_without_surrounding_whitespace() -> None:  # 18.2
    server = FakeOllamaServer(reply="\n  A good will is good in itself [1].  \n")

    assert _chat_model(server.transport).chat(CONVERSATION) == (
        "A good will is good in itself [1]."
    )


@pytest.mark.parametrize("reply", ["", " \n "], ids=["empty", "whitespace"])
def test_an_empty_reply_is_a_provider_error(reply: str) -> None:  # 18.3
    server = FakeOllamaServer(reply=reply)

    with pytest.raises(ProviderError, match=f"{CHAT_MODEL}.*empty"):
        _chat_model(server.transport).chat(CONVERSATION)


@pytest.mark.parametrize(
    ("transport", "error", "message"),
    [
        (
            FakeOllamaServer(models=[MODEL]).transport,
            ModelNotAvailableError,
            f"ollama pull {CHAT_MODEL}",
        ),
        (refusing_transport(), ProviderUnavailableError, "Ollama"),
        (FakeOllamaServer(status=500).transport, ProviderError, "simulated server failure"),
        (_timing_out_transport(), ProviderError, "timed out"),
    ],
    ids=["not-pulled", "not-running", "server-error", "timeout"],
)
def test_chat_failures_are_translated_like_the_embedders(
    transport: httpx.BaseTransport, error: type[ProviderError], message: str
) -> None:  # 18.4
    with pytest.raises(ProviderError, match=message) as raised:
        _chat_model(transport).chat(CONVERSATION)

    assert type(raised.value) is error


def test_chat_model_close_releases_the_connections() -> None:  # 18.5
    server = FakeOllamaServer()
    model = _chat_model(server.transport)
    model.chat(CONVERSATION)

    model.close()

    assert server.transport.closed


class Verdict(BaseModel):
    relevant: bool


def test_structured_chat_sends_the_schema_with_thinking_off_and_temperature_zero() -> None:
    # 18.6
    server = FakeOllamaServer()

    _chat_model(server.transport).chat_structured(CONVERSATION, Verdict)

    [request] = server.requests
    assert request["messages"][-1] == {"role": "user", "content": "What is the good will?"}
    assert request["format"] == Verdict.model_json_schema()
    assert request["think"] is False
    assert request["options"]["temperature"] == 0  # the same passage, the same verdict


def test_structured_chat_returns_the_reply_parsed_into_the_schema() -> None:  # 18.7
    server = FakeOllamaServer(structured_reply=lambda prompt: '{"relevant": false}')

    verdict = _chat_model(server.transport).chat_structured(CONVERSATION, Verdict)

    assert verdict == Verdict(relevant=False)


@pytest.mark.parametrize(
    "reply", ['{"relevant": "perhaps"}', "Yes, it is relevant."], ids=["wrong-field", "prose"]
)
def test_a_structured_reply_that_does_not_fit_is_a_provider_error(reply: str) -> None:  # 18.8
    server = FakeOllamaServer(structured_reply=lambda prompt: reply)

    with pytest.raises(ProviderError, match=f"{CHAT_MODEL}.*Verdict"):
        _chat_model(server.transport).chat_structured(CONVERSATION, Verdict)


def test_structured_chat_failures_are_translated_like_chat() -> None:  # 18.8
    server = FakeOllamaServer(models=[MODEL])

    with pytest.raises(ModelNotAvailableError, match=f"ollama pull {CHAT_MODEL}"):
        _chat_model(server.transport).chat_structured(CONVERSATION, Verdict)


def test_every_request_sets_the_same_context_length() -> None:  # 18.9
    server = FakeOllamaServer()
    model = _chat_model(server.transport, context_length=8192)

    model.chat(CONVERSATION)
    model.chat_structured(CONVERSATION, Verdict)

    # A different context length would make Ollama reload the model between the two.
    chat, structured = server.requests
    assert chat["options"] == {"num_ctx": 8192}
    assert structured["options"] == {"num_ctx": 8192, "temperature": 0}


class NonBlank(BaseModel):
    text: str = Field(pattern=r"\S")  # unanchored: Ollama cannot build a grammar from it


def test_a_schema_ollama_cannot_turn_into_a_grammar_is_a_provider_error() -> None:  # 18.10
    server = FakeOllamaServer(structured_reply=lambda prompt: '{"text": "a"}')

    with pytest.raises(ProviderError, match=r"Pattern must start with '\^' and end with '\$'"):
        _chat_model(server.transport).chat_structured(CONVERSATION, NonBlank)
