"""Embeddings from a local Ollama server.

Uses the official `ollama` client's `/api/embed` endpoint, which takes a
batch of inputs per request. The client is injected, so the composition
root decides the host and timeout, and tests can give it a fake transport.
"""

from collections.abc import Sequence
from http import HTTPStatus

import httpx
import ollama

from aporia_rag.adapters.embedding import BatchingEmbedder, EmbeddingPurpose
from aporia_rag.adapters.prompt_formats import PromptFormat
from aporia_rag.domain.errors import (
    ModelNotAvailableError,
    ProviderError,
    ProviderUnavailableError,
)
from aporia_rag.domain.models import EmbeddingSpace, Passage
from aporia_rag.domain.ports import Vector


class OllamaEmbedder(BatchingEmbedder):
    """An `Embedder` backed by an Ollama embedding model such as `qwen3-embedding:0.6b`."""

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
        # Public API, but unannotated in ollama 0.6 (`def close(self):`).
        self._client.close()  # type: ignore[no-untyped-call]

    def _embed_batch(self, passages: Sequence[Passage], purpose: EmbeddingPurpose) -> list[Vector]:
        if purpose is EmbeddingPurpose.QUERY:
            inputs = [self._format.query(passage.text) for passage in passages]
        else:
            inputs = [self._format.document(passage) for passage in passages]
        model = self.space.model
        try:
            response = self._client.embed(
                model=model,
                input=inputs,
                # Without this, Ollama silently cuts inputs longer than the
                # model's context. A chunk embedded from half its text would
                # still be retrieved by its whole text: better to fail loudly.
                truncate=False,
                dimensions=self.space.dimensions,
                keep_alive=self._keep_alive,
            )
        except ConnectionError as error:  # the client's translation of httpx.ConnectError
            raise ProviderUnavailableError(f"cannot reach Ollama: {error}") from error
        except ollama.ResponseError as error:
            if error.status_code == HTTPStatus.NOT_FOUND:
                raise ModelNotAvailableError(
                    f"Ollama does not have the model {model!r}; run `ollama pull {model}`"
                ) from error
            raise ProviderError(f"Ollama failed to embed: {error}") from error
        except httpx.HTTPError as error:  # e.g. a timeout, which the client lets through
            raise ProviderError(f"Ollama request failed: {error}") from error
        return [list(vector) for vector in response.embeddings]
