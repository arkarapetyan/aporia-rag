"""Tests for aporia_rag.ingestion.loader (test plan: Component 8)."""

from pathlib import Path

import pytest

from aporia_rag.domain.errors import CorpusError
from aporia_rag.domain.models import SourceMetadata
from aporia_rag.ingestion.corpus import EditionLayout, WorkSpec
from aporia_rag.ingestion.loader import iter_documents
from aporia_rag.ingestion.structure import StructureError

REPUBLIC = WorkSpec(
    id="plato-republic",
    author="Plato",
    title="The Republic",
    translator="Benjamin Jowett",
    gutenberg_id=1497,
    layout=EditionLayout(start=r"BOOK I\.", sections=[r"BOOK [IVX]+\."]),
)
SENECA = WorkSpec(
    id="seneca-minor-dialogues",
    author="Lucius Annaeus Seneca",
    title="Minor Dialogues",
    gutenberg_id=64576,
    layout=EditionLayout(start=r"OF ANGER, BOOK I\.", sections=[r"OF ANGER, BOOK [IVX]+\."]),
)


def _as_downloaded(body: str) -> str:
    return (
        "The Project Gutenberg eBook\n\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\n"
        f"{body}\n\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\n\nLicense text.\n"
    )


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    republic = (
        "Contents\n\nBOOK I.\n\n"
        "I went down\nyesterday to the Piraeus.\n\n"  # hard-wrapped, as in the raw files
        "BOOK II.\n\nWith these words."
    )
    seneca = "OF ANGER, BOOK I.\n\nYou have demanded of me, Novatus."
    (tmp_path / "pg1497.txt").write_text(_as_downloaded(republic), encoding="utf-8")
    (tmp_path / "pg64576.txt").write_text(_as_downloaded(seneca), encoding="utf-8")
    return tmp_path


def test_yields_one_document_per_section_with_metadata(cache_dir: Path) -> None:  # 8.1
    documents = list(iter_documents([REPUBLIC], cache_dir))

    assert [document.text for document in documents] == [
        "I went down yesterday to the Piraeus.",
        "With these words.",
    ]
    assert documents[0].metadata == SourceMetadata(
        author="Plato",
        title="The Republic",
        translator="Benjamin Jowett",
        source_id="gutenberg:1497",
        section="Book I",
    )


def test_document_ids_are_the_work_id_and_a_slug_of_the_label(cache_dir: Path) -> None:  # 8.2
    documents = list(iter_documents([REPUBLIC, SENECA], cache_dir))

    assert [document.id for document in documents] == [
        "plato-republic/book-i",
        "plato-republic/book-ii",
        "seneca-minor-dialogues/of-anger-book-i",
    ]


def test_duplicate_labels_in_a_work_raise(cache_dir: Path) -> None:  # 8.3
    layout = REPUBLIC.layout.model_copy(update={"section_labels": ["Book I", "Book I"]})
    clashing = REPUBLIC.model_copy(update={"layout": layout})

    with pytest.raises(StructureError, match="duplicate"):
        list(iter_documents([clashing], cache_dir))


def test_is_lazy(cache_dir: Path) -> None:  # 8.4
    (cache_dir / "pg64576.txt").unlink()
    documents = iter_documents([REPUBLIC, SENECA], cache_dir)

    assert next(documents).id == "plato-republic/book-i"
    assert next(documents).id == "plato-republic/book-ii"
    with pytest.raises(FileNotFoundError):
        next(documents)


def test_missing_file_names_the_work_and_the_fix(tmp_path: Path) -> None:  # 8.5
    with pytest.raises(FileNotFoundError, match=r"seneca-minor-dialogues.*aporia-rag fetch"):
        list(iter_documents([SENECA], tmp_path))


@pytest.mark.parametrize(
    "content",
    [
        "<html>404 Not Found</html>",  # not a Gutenberg text at all
        _as_downloaded("A preface, but no BOOK I heading anywhere."),  # wrong layout
    ],
    ids=["not-gutenberg", "wrong-layout"],
)
def test_unusable_file_raises_corpus_error_naming_the_work(
    tmp_path: Path, content: str
) -> None:  # 8.6
    (tmp_path / "pg1497.txt").write_text(content, encoding="utf-8")

    with pytest.raises(CorpusError, match="plato-republic"):
        list(iter_documents([REPUBLIC], tmp_path))
