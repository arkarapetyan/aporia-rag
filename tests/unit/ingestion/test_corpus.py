"""Tests for aporia_rag.ingestion.corpus (test plan: Component 3)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from aporia_rag.ingestion.corpus import CorpusManifest, EditionLayout, WorkSpec, load_manifest

CONFIRMED_GUTENBERG_IDS = {
    1497,
    8438,
    47001,
    64576,
    5682,
    11224,
    52319,
    59,
    9662,
    38427,
    3207,
    34901,
}


LAYOUT = {"start": r"BOOK I\.", "sections": [r"BOOK [IVX]+\."]}


def _work(**overrides: object) -> WorkSpec:
    fields: dict[str, object] = {
        "id": "plato-republic",
        "author": "Plato",
        "title": "The Republic",
        "gutenberg_id": 1497,
        "layout": LAYOUT,
    }
    return WorkSpec.model_validate(fields | overrides)


def _layout(**overrides: object) -> EditionLayout:
    return EditionLayout.model_validate(LAYOUT | overrides)


class TestWorkSpec:
    @pytest.mark.parametrize(
        "bad_id", ["Plato-Republic", "plato republic", "-plato", "plato-", "plato--republic", ""]
    )
    def test_id_must_be_a_lowercase_slug(self, bad_id: str) -> None:  # 3.1
        with pytest.raises(ValidationError, match="id"):
            _work(id=bad_id)

    @pytest.mark.parametrize("bad_gutenberg_id", [0, -3])
    def test_gutenberg_id_must_be_positive(self, bad_gutenberg_id: int) -> None:  # 3.2
        with pytest.raises(ValidationError, match="gutenberg_id"):
            _work(gutenberg_id=bad_gutenberg_id)

    def test_translator_is_optional(self) -> None:  # 3.3
        assert _work().translator is None
        assert _work(translator="Benjamin Jowett").translator == "Benjamin Jowett"


class TestCorpusManifest:
    def test_rejects_duplicate_work_ids(self) -> None:  # 3.4
        with pytest.raises(ValidationError, match="duplicate work id: plato-republic"):
            CorpusManifest(works=[_work(), _work(gutenberg_id=2)])

    def test_rejects_duplicate_gutenberg_ids(self) -> None:  # 3.5
        with pytest.raises(ValidationError, match="duplicate gutenberg_id: 1497"):
            CorpusManifest(works=[_work(), _work(id="plato-republic-again")])


class TestLoadManifest:
    def test_parses_a_toml_file(self, tmp_path: Path) -> None:  # 3.6
        path = tmp_path / "corpus.toml"
        path.write_text(
            "[[works]]\n"
            'id = "plato-republic"\n'
            'author = "Plato"\n'
            'title = "The Republic"\n'
            'translator = "Benjamin Jowett"\n'
            "gutenberg_id = 1497\n"
            "[works.layout]\n"
            "start = 'BOOK I\\.'\n"
            "sections = ['BOOK [IVX]+\\.']\n",
            encoding="utf-8",
        )

        manifest = load_manifest(path)

        assert manifest.works == [_work(translator="Benjamin Jowett")]

    def test_packaged_manifest_lists_the_confirmed_corpus(self) -> None:  # 3.7
        manifest = load_manifest()

        assert len(manifest.works) == len(CONFIRMED_GUTENBERG_IDS)
        assert {work.gutenberg_id for work in manifest.works} == CONFIRMED_GUTENBERG_IDS


class TestEditionLayout:
    def test_every_work_needs_a_layout(self) -> None:  # 3.8
        fields = _work().model_dump(exclude={"layout"})
        with pytest.raises(ValidationError, match="layout"):
            WorkSpec.model_validate(fields)

    @pytest.mark.parametrize("field", ["start", "end", "sections"])
    def test_rejects_invalid_regular_expressions(self, field: str) -> None:  # 3.9
        broken = ["BOOK ["] if field == "sections" else "BOOK ["
        with pytest.raises(ValidationError, match="regular expression"):
            _layout(**{field: broken})

    @pytest.mark.parametrize("levels", [[], ["BOOK .*", "CHAPTER .*", "VERSE .*"]], ids=["0", "3"])
    def test_needs_one_or_two_section_levels(self, levels: list[str]) -> None:  # 3.10
        with pytest.raises(ValidationError, match="sections"):
            _layout(sections=levels)
