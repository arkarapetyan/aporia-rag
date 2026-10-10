"""Contract tests on the real downloaded corpus (test plan: I.2-I.7).

Opt-in (`make test-integration`), and skipped per work when its file is not in
data/raw. These pin down what the structuring spike verified on the real
editions, so a change to a layout or a clean-up rule that breaks a real work
fails here even when every synthetic unit test still passes.
"""

import re
from pathlib import Path

import pytest

from aporia_rag.domain.models import Document
from aporia_rag.ingestion.corpus import load_manifest
from aporia_rag.ingestion.loader import iter_documents

pytestmark = pytest.mark.integration

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
WORKS = {work.id: work for work in load_manifest().works}
EXPECTED_SECTIONS = {
    "plato-republic": 10,
    "aristotle-nicomachean-ethics": 119,
    "cicero-de-officiis": 3,
    "seneca-minor-dialogues": 14,
    "kant-metaphysic-of-morals": 4,
    "mill-utilitarianism": 5,
    "nietzsche-genealogy-of-morals": 4,
    "descartes-discourse-on-method": 7,
    "hume-enquiry": 19,
    "schopenhauer-world-as-will-vol1": 6,
    "hobbes-leviathan": 49,
    "mill-on-liberty": 5,
}
LATIN_FUNCTION_WORDS = {
    "et",
    "est",
    "non",
    "quod",
    "ut",
    "cum",
    "sed",
    "quae",
    "qui",
    "enim",
    "autem",
}


def _documents(work_id: str) -> list[Document]:
    work = WORKS[work_id]
    if not (RAW_DIR / f"pg{work.gutenberg_id}.txt").is_file():
        pytest.skip(f"{work_id} not downloaded: run `aporia-rag fetch`")
    return list(iter_documents([work], RAW_DIR))


def _paragraphs(documents: list[Document]) -> list[str]:
    return [paragraph for document in documents for paragraph in document.text.split("\n\n")]


def test_manifest_and_expectations_cover_the_same_works() -> None:
    assert set(EXPECTED_SECTIONS) == set(WORKS)


@pytest.mark.parametrize(("work_id", "expected"), EXPECTED_SECTIONS.items())
def test_every_work_structures_into_the_expected_sections(
    work_id: str, expected: int
) -> None:  # I.2
    assert len(_documents(work_id)) == expected


def test_republic_starts_with_plato_not_jowett() -> None:  # I.3
    assert _documents("plato-republic")[0].text.startswith("I went down yesterday to the Piraeus")


def test_leviathan_keeps_chapter_xl() -> None:  # I.4
    sections = [document.metadata.section or "" for document in _documents("hobbes-leviathan")]

    assert any(re.search(r"Chapter XL\b", section) for section in sections)


def test_de_officiis_is_english_only() -> None:  # I.5
    documents = _documents("cicero-de-officiis")

    for paragraph in _paragraphs(documents):
        words = re.findall(r"[a-z]+", paragraph.lower())
        assert sum(word in LATIN_FUNCTION_WORDS for word in words) < 2, paragraph[:80]
    assert documents[0].text.startswith("I. My dear son Marcus")


@pytest.mark.parametrize("work_id", sorted(EXPECTED_SECTIONS))
def test_no_boilerplate_footnotes_or_index(work_id: str) -> None:  # I.6
    for paragraph in _paragraphs(_documents(work_id)):
        assert "PROJECT GUTENBERG" not in paragraph.upper()
        assert not re.match(r"\[(?:\d+|[A-Z]{1,2})\]|\[Footnote", paragraph), paragraph[:80]
        assert paragraph not in {"INDEX", "INDEX.", "NOTES", "FOOTNOTES"}


def test_no_margin_notes_survive_in_de_officiis() -> None:  # I.8
    for document in _documents("cicero-de-officiis"):
        assert not re.search(r"#[^#]+#", document.text), document.id


def test_document_ids_are_unique_across_the_corpus() -> None:  # I.7
    ids = [document.id for work_id in EXPECTED_SECTIONS for document in _documents(work_id)]

    assert len(ids) == len(set(ids))
