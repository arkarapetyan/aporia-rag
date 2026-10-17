"""Application settings, read from the environment and an optional `.env` file.

Every setting has a default that runs fully locally, so nothing needs to be
configured to start. To change one, set an environment variable named
`APORIA_<SECTION>__<FIELD>` (note the double underscore between section and
field), or put the same line in a `.env` file in the working directory:

    APORIA_EMBEDDING__MODEL=embeddinggemma:300m
    APORIA_OLLAMA__HOST=http://gpu-box:11434

Real environment variables win over `.env`, and `.env` wins over defaults.
See `.env.example` for every setting.

`provider` fields are names looked up in the registries in `bootstrap.py`.
They are plain strings, not an enum of today's providers, so registering a
new provider does not mean editing this file.
"""

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aporia_rag.domain.errors import ConfigurationError


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class OllamaSettings(_Section):
    host: str = "http://localhost:11434"
    timeout_seconds: float = Field(default=120.0, gt=0)
    keep_alive: str | None = Field(
        default="10m",
        description="How long Ollama keeps a model in memory after a request ('10m', '-1').",
    )


class EmbeddingSettings(_Section):
    provider: str = "ollama"
    model: str = "qwen3-embedding:0.6b"
    dimensions: int | None = Field(
        default=None,
        gt=0,
        description="Shorter vectors from Matryoshka models; None keeps the model's own size.",
    )
    batch_size: int = Field(default=32, gt=0)
    query_instruction: str | None = Field(
        default=None, description="Overrides the model's default query instruction, if it has one."
    )


class ChatSettings(_Section):
    provider: str = "ollama"
    model: str = Field(
        default="qwen3.5:35b-a3b-q4_K_M",
        description="A mixture of experts: ~3B of 35B parameters per token. Needs ~22 GB of RAM.",
    )
    context_length: int = Field(
        default=8192,
        gt=0,
        description="Tokens the model sees at once, prompt and reply. A survey of all authors "
        "needs ~4,500.",
    )


class ChunkingSettings(_Section):
    size: int = Field(default=1000, gt=0, description="Target chunk size in characters.")
    overlap: int = Field(default=150, ge=0, description="Characters repeated between chunks.")

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> Self:
        if self.overlap >= self.size:
            raise ValueError(f"overlap ({self.overlap}) must be smaller than size ({self.size})")
        return self


class StorageSettings(_Section):
    provider: str = "chroma"
    corpus_dir: Path = Path("data/raw")
    index_dir: Path = Path("data/index")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APORIA_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # a shared .env may hold other tools' variables
        frozen=True,
    )

    ollama: OllamaSettings = OllamaSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    chat: ChatSettings = ChatSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    storage: StorageSettings = StorageSettings()


def load_settings() -> Settings:
    """Read the settings, reporting invalid values as a ConfigurationError."""
    try:
        return Settings()
    except ValidationError as error:
        raise ConfigurationError(f"invalid settings: {error}") from error
