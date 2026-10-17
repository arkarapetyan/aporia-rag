"""Embeddings and chat from a local Ollama server.

Uses the official `ollama` client: `/api/embed`, which takes a batch of
inputs per request, and `/api/chat`. The client is injected, so the
composition root decides the host and timeout, and tests can give it a fake
transport. Both adapters translate the client's failures the same way, in
`_translated_errors`.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from http import HTTPStatus
from typing import Any, Final

import httpx
import ollama
from pydantic import BaseModel, ValidationError

from aporia_rag.adapters.embedding import BatchingEmbedder, EmbeddingPurpose
from aporia_rag.adapters.prompt_formats import PromptFormat
from aporia_rag.domain.errors import (
    ModelNotAvailableError,
    ProviderError,
    ProviderUnavailableError,
)
from aporia_rag.domain.models import ChatMessage, EmbeddingSpace, Passage
from aporia_rag.domain.ports import Vector

# Ollama runs a model with this option on the CPU only (no layers on the GPU).
_ON_CPU: Final = {"num_gpu": 0}
# Always the most likely token: the same messages give the same decision.
_REPEATABLE: Final = {"temperature": 0}


class OllamaEmbedder(BatchingEmbedder):
    """An `Embedder` backed by an Ollama embedding model such as `qwen3-embedding:0.6b`.

    Queries are embedded on the CPU. On a small GPU the embedder (2.4 GB of
    VRAM with its working memory) and a chat model cannot both stay loaded,
    and Ollama evicts one to load the other: seconds of reloading on every
    question. One short query takes a fraction of a second on the CPU, and
    the GPU stays with the chat model. Passages are not restricted: indexing
    embeds thousands of them, with nothing else loaded.
    """

    def __init__(
        self,
        client: ollama.Client,
        model: str,
        *,
        prompt_format: PromptFormat,
        dimensions: int | None = None,
        batch_size: int = 32,
        keep_alive: str | None = None,
    ) -> None:
        space = EmbeddingSpace(
            provider="ollama",
            model=model,
            dimensions=dimensions,
            prompt_format=prompt_format.name,
        )
        super().__init__(space, batch_size=batch_size)
        self._client = client
        self._format = prompt_format
        self._keep_alive = keep_alive

    def close(self) -> None:
        """Release the client's pooled HTTP connections."""
        _close(self._client)

    def _embed_batch(self, passages: Sequence[Passage], purpose: EmbeddingPurpose) -> list[Vector]:
        if purpose is EmbeddingPurpose.QUERY:
            inputs = [self._format.query(passage.text) for passage in passages]
        else:
            inputs = [self._format.document(passage) for passage in passages]
        with _translated_errors(self.space.model):
            response = self._client.embed(
                model=self.space.model,
                input=inputs,
                # Without this, Ollama silently cuts inputs longer than the
                # model's context. A chunk embedded from half its text would
                # still be retrieved by its whole text: better to fail loudly.
                truncate=False,
                dimensions=self.space.dimensions,
                options=_ON_CPU if purpose is EmbeddingPurpose.QUERY else None,
                keep_alive=self._keep_alive,
            )
        return [list(vector) for vector in response.embeddings]


class OllamaChatModel:
    """A `ChatModel` backed by an Ollama model such as `qwen3.5:35b-a3b`.

    Every request sets the same context length. Ollama's default can be as
    small as 4096 tokens, too few for a survey of all the authors, and Ollama
    reloads a model whose context length changes from one request to the next.
    """

    def __init__(
        self,
        client: ollama.Client,
        model: str,
        *,
        context_length: int,
        keep_alive: str | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._context_length = context_length
        self._keep_alive = keep_alive

    def close(self) -> None:
        """Release the client's pooled HTTP connections."""
        _close(self._client)

    def chat(self, messages: Sequence[ChatMessage]) -> str:
        return self._reply(messages)

    def chat_structured[M: BaseModel](self, messages: Sequence[ChatMessage], schema: type[M]) -> M:
        # Ollama constrains generation to the JSON schema, so the reply is
        # valid JSON; whether it fits the schema is still checked.
        reply = self._reply(messages, format=schema.model_json_schema(), options=_REPEATABLE)
        try:
            return schema.model_validate_json(reply)
        except ValidationError as error:
            raise ProviderError(
                f"Ollama's {self._model} replied {reply!r}, which is not a valid {schema.__name__}"
            ) from error

    def _reply(
        self,
        messages: Sequence[ChatMessage],
        *,
        format: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        with _translated_errors(self._model):
            response = self._client.chat(
                model=self._model,
                messages=[
                    {"role": message.role, "content": message.content} for message in messages
                ],
                # Answer directly. A thinking model (Qwen3, Qwen3.5) would
                # otherwise reason first, which takes several times as long;
                # models that cannot think are not affected.
                think=False,
                format=format,
                options={"num_ctx": self._context_length, **(options or {})},
                keep_alive=self._keep_alive,
            )
        reply = (response.message.content or "").strip()
        if not reply:
            raise ProviderError(f"Ollama's {self._model} returned an empty reply")
        return reply


def _close(client: ollama.Client) -> None:
    # Public API, but unannotated in ollama 0.6 (`def close(self):`).
    client.close()  # type: ignore[no-untyped-call]


@contextmanager
def _translated_errors(model: str) -> Iterator[None]:
    """Turn the client's failures into the application's `ProviderError`s."""
    try:
        yield
    except ConnectionError as error:  # the client's translation of httpx.ConnectError
        raise ProviderUnavailableError(f"cannot reach Ollama: {error}") from error
    except ollama.ResponseError as error:
        if error.status_code == HTTPStatus.NOT_FOUND:
            raise ModelNotAvailableError(
                f"Ollama does not have the model {model!r}; run `ollama pull {model}`"
            ) from error
        raise ProviderError(f"Ollama failed: {error}") from error
    except httpx.HTTPError as error:  # e.g. a timeout, which the client lets through
        raise ProviderError(f"Ollama request failed: {error}") from error
