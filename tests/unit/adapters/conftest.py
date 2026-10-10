"""Fixtures shared by the vector store tests."""

from collections.abc import Iterator

import chromadb
import pytest
from chromadb.api import ClientAPI
from chromadb.config import Settings


@pytest.fixture
def chroma_client() -> Iterator[ClientAPI]:
    """An in-memory Chroma client, emptied after the test.

    Chroma shares one in-memory system per process, so without the reset
    one test's collections would leak into the next.
    """
    client = chromadb.EphemeralClient(
        settings=Settings(allow_reset=True, anonymized_telemetry=False)
    )
    yield client
    client.reset()
