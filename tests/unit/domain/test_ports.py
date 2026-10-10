"""Tests for the helpers in aporia_rag.domain.ports (test plan: 1.14-1.15, 1.17).

The ports themselves are Protocols with no behaviour; these helpers are the
rules that go with them, written once so that no store, indexer or
retriever can get them subtly different.
"""

import pytest

from aporia_rag.domain.errors import IndexMismatchError, VectorDimensionError
from aporia_rag.domain.ports import (
    FILTERABLE_FIELDS,
    check_filters,
    check_same_space,
    common_dimension,
)
from support.fakes import HashingEmbedder, InMemoryVectorStore


class TestCheckFilters:
    def test_accepts_none_and_known_fields(self) -> None:  # 1.14
        check_filters(None)
        check_filters({"author": "Plato", "section": "Book I"})

    def test_rejects_an_unknown_field_and_lists_the_known_ones(self) -> None:  # 1.14
        with pytest.raises(ValueError, match="autor") as raised:
            check_filters({"autor": "Plato"})

        for field in FILTERABLE_FIELDS:
            assert field in str(raised.value)


class TestCommonDimension:
    def test_no_vectors_have_no_dimension(self) -> None:  # 1.15
        assert common_dimension([]) is None

    def test_returns_the_shared_size(self) -> None:  # 1.15
        assert common_dimension([[1.0, 0.0], [0.0, 1.0]]) == 2
        assert common_dimension([[1.0, 0.0]], expected=2) == 2

    @pytest.mark.parametrize(
        ("vectors", "expected"),
        [
            ([[1.0, 0.0], [1.0, 0.0, 0.0]], None),  # disagree with each other
            ([[1.0, 0.0]], 3),  # disagree with what is expected
            ([[]], None),  # an empty vector is never valid
        ],
        ids=["mixed", "unexpected", "empty"],
    )
    def test_rejects_inconsistent_sizes(
        self, vectors: list[list[float]], expected: int | None
    ) -> None:  # 1.15
        with pytest.raises(VectorDimensionError):
            common_dimension(vectors, expected)

    def test_dimension_error_is_also_a_value_error(self) -> None:  # 1.15
        assert issubclass(VectorDimensionError, ValueError)


def test_same_space_is_accepted_and_another_is_refused() -> None:  # 1.17
    embedder = HashingEmbedder(model="model-a")
    check_same_space(embedder, InMemoryVectorStore(embedder.space))

    with pytest.raises(IndexMismatchError, match=r"model-a.*model-b"):
        check_same_space(embedder, InMemoryVectorStore(HashingEmbedder(model="model-b").space))
