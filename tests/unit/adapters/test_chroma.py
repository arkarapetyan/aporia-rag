"""Chroma-specific behaviour (test plan: 11.13-11.17).

Everything a VectorStore must do is in the shared contract suite; this file
covers what only a persistent, named-collection store has to get right.
"""

import re
from pathlib import Path
from typing import Any

import chromadb
import pytest
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.shared_system_client import SharedSystemClient
from chromadb.config import Settings
from hypothesis import given
from hypothesis import strategies as st

from aporia_rag.adapters.chroma import ChromaVectorStore, client_settings, collection_name
from aporia_rag.domain.errors import IndexMismatchError
from aporia_rag.domain.models import EmbeddingSpace
from support.builders import make_record

SPACE = EmbeddingSpace(
    provider="ollama", model="qwen3-embedding:0.6b", dimensions=None, prompt_format="qwen-instruct"
)


def test_data_survives_reopening_the_store(tmp_path: Path) -> None:  # 11.13
    ChromaVectorStore.persistent(tmp_path / "index", SPACE).upsert(
        [make_record("kant#0", [1.0, 0.0, 0.0], fingerprint="f1")]
    )
    SharedSystemClient.clear_system_cache()  # forget the open client: really re-read the disk

    reopened = ChromaVectorStore.persistent(tmp_path / "index", SPACE)

    assert reopened.fingerprints() == {"kant#0": "f1"}
    assert [result.chunk.id for result in reopened.query([1.0, 0.0, 0.0], k=1)] == ["kant#0"]


def test_collection_recorded_for_another_space_is_refused(
    chroma_client: ClientAPI,
) -> None:  # 11.14
    # As if a collection with this name had been created by something else
    # (an older version, or a different space whose name happens to collide).
    chroma_client.create_collection(
        collection_name(SPACE),
        metadata={"space_id": "ollama/other-model/native/plain"},
        embedding_function=None,
    )

    with pytest.raises(IndexMismatchError, match="other-model"):
        ChromaVectorStore(chroma_client, SPACE)


_CHROMA_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{1,61}[a-zA-Z0-9]")
_spaces = st.builds(
    EmbeddingSpace,
    provider=st.text(min_size=1),
    model=st.text(min_size=1),
    dimensions=st.none() | st.integers(min_value=1, max_value=8192),
    prompt_format=st.text(min_size=1),
)


@given(_spaces)
def test_collection_names_follow_chromas_rules(space: EmbeddingSpace) -> None:  # 11.15
    name = collection_name(space)

    assert _CHROMA_NAME.fullmatch(name), name
    assert ".." not in name


@given(_spaces, _spaces)
def test_different_spaces_get_different_collections(
    first: EmbeddingSpace, second: EmbeddingSpace
) -> None:  # 11.15
    if first != second:
        assert collection_name(first) != collection_name(second)


def test_collection_name_is_readable() -> None:  # 11.15
    assert collection_name(SPACE).startswith("ollama-qwen3-embedding-0-6b-native-qwen-instruct-")


def test_large_upserts_are_split_into_batches(
    chroma_client: ClientAPI, monkeypatch: pytest.MonkeyPatch
) -> None:  # 11.16
    batch_sizes: list[int] = []
    original_upsert = Collection.upsert

    def recording_upsert(self: Collection, ids: list[str], **kwargs: Any) -> None:
        batch_sizes.append(len(ids))
        original_upsert(self, ids, **kwargs)

    monkeypatch.setattr(Collection, "upsert", recording_upsert)
    store = ChromaVectorStore(chroma_client, SPACE, batch_size=2)

    store.upsert([make_record(f"kant#{index}", [1.0, float(index), 0.0]) for index in range(5)])

    assert batch_sizes == [2, 2, 1]
    assert store.count() == 5


def test_persistent_client_has_telemetry_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # 11.17
    settings_used: list[Settings] = []
    real_client = chromadb.PersistentClient

    def recording_client(*args: Any, **kwargs: Any) -> ClientAPI:
        settings_used.append(kwargs["settings"])
        return real_client(*args, **kwargs)

    monkeypatch.setattr(chromadb, "PersistentClient", recording_client)

    ChromaVectorStore.persistent(tmp_path / "index", SPACE)

    assert [settings.anonymized_telemetry for settings in settings_used] == [False]
    assert client_settings().anonymized_telemetry is False
