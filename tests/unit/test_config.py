"""Tests for aporia_rag.config (test plan: Component 9).

`_env_file=None` keeps a developer's own `.env` out of the tests; the
suite-wide fixture in conftest.py does the same for APORIA_* variables.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from aporia_rag.config import Settings, load_settings
from aporia_rag.domain.errors import ConfigurationError


def test_defaults_run_everything_locally() -> None:  # 9.1
    settings = Settings(_env_file=None)

    assert settings.ollama.host == "http://localhost:11434"
    assert (settings.embedding.provider, settings.embedding.model) == (
        "ollama",
        "qwen3-embedding:0.6b",
    )
    assert settings.embedding.dimensions is None  # the model's native size
    assert settings.embedding.query_instruction is None  # the model's documented default
    assert (settings.chat.provider, settings.chat.model) == ("ollama", "qwen3.5:35b-a3b-q4_K_M")
    assert settings.storage.provider == "chroma"
    assert settings.storage.corpus_dir == Path("data/raw")
    assert settings.storage.index_dir == Path("data/index")
    assert (settings.chunking.size, settings.chunking.overlap) == (1000, 150)


def test_nested_environment_variables_override_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # 9.2
    monkeypatch.setenv("APORIA_EMBEDDING__MODEL", "embeddinggemma:300m")
    monkeypatch.setenv("APORIA_EMBEDDING__DIMENSIONS", "256")
    monkeypatch.setenv("APORIA_OLLAMA__HOST", "http://gpu-box:11434")

    settings = Settings(_env_file=None)

    assert settings.embedding.model == "embeddinggemma:300m"
    assert settings.embedding.dimensions == 256
    assert settings.ollama.host == "http://gpu-box:11434"
    assert settings.embedding.provider == "ollama"  # untouched siblings keep their defaults


def test_values_are_read_from_a_dotenv_file_and_the_environment_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # 9.3
    env_file = tmp_path / ".env"
    env_file.write_text(
        "APORIA_EMBEDDING__MODEL=embeddinggemma:300m\n"
        "APORIA_CHUNKING__SIZE=800\n"
        "SOME_OTHER_TOOL_KEY=ignored\n",  # a shared .env may hold other tools' settings
        encoding="utf-8",
    )
    monkeypatch.setenv("APORIA_CHUNKING__SIZE", "1200")

    settings = Settings(_env_file=env_file)

    assert settings.embedding.model == "embeddinggemma:300m"
    assert settings.chunking.size == 1200


def test_load_settings_reads_the_dotenv_file_in_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # 9.3
    (tmp_path / ".env").write_text("APORIA_EMBEDDING__MODEL=nomic-embed-text\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert load_settings().embedding.model == "nomic-embed-text"


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("APORIA_EMBEDDING__BATCH_SIZE", "0", "batch_size"),
        ("APORIA_EMBEDDING__DIMENSIONS", "-1", "dimensions"),
        ("APORIA_CHUNKING__OVERLAP", "1000", "overlap"),  # not smaller than the size
    ],
)
def test_invalid_values_are_rejected(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str, message: str
) -> None:  # 9.4
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError, match=message):
        Settings(_env_file=None)


def test_load_settings_reports_invalid_values_as_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:  # 9.4
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.setenv("APORIA_EMBEDDING__BATCH_SIZE", "0")

    with pytest.raises(ConfigurationError, match="batch_size"):
        load_settings()
