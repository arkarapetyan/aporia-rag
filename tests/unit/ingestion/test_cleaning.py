"""Tests for aporia_rag.ingestion.cleaning (test plan: Component 5)."""

import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

from aporia_rag.ingestion.cleaning import clean_gutenberg_text, extract_body, normalize_text

START = "*** START OF THE PROJECT GUTENBERG EBOOK THE REPUBLIC ***"
END = "*** END OF THE PROJECT GUTENBERG EBOOK THE REPUBLIC ***"
HEADER = "The Project Gutenberg eBook of The Republic\n\nThis ebook is for the use of anyone.\n\n"
LICENSE = "\n\nSection 1. General Terms of Use and Redistributing Project Gutenberg works\n"


def _ebook(body: str, *, start: str = START, end: str = END) -> str:
    return f"{HEADER}{start}\n\n{body}\n\n{end}{LICENSE}"


class TestExtractBody:
    def test_keeps_only_the_text_between_the_markers(self) -> None:  # 5.1
        assert extract_body(_ebook("Socrates speaks.")) == "Socrates speaks."

    @pytest.mark.parametrize(
        ("start", "end"),
        [
            (
                "*** START OF THIS PROJECT GUTENBERG EBOOK LEVIATHAN ***",
                "*** END OF THIS PROJECT GUTENBERG EBOOK LEVIATHAN ***",
            ),
            (
                "***START OF THE PROJECT GUTENBERG EBOOK ON LIBERTY***",
                "***END OF THE PROJECT GUTENBERG EBOOK ON LIBERTY***",
            ),
            (
                "*** START OF THE PROJECT GUTENBERG E-BOOK ETHICS ***",
                "*** END OF THE PROJECT GUTENBERG E-BOOK ETHICS ***",
            ),
            (
                "*** Start of the Project Gutenberg eBook Ethics ***",
                "*** End of the Project Gutenberg eBook Ethics ***",
            ),
        ],
        ids=["this", "no-space", "e-book", "mixed-case"],
    )
    def test_accepts_marker_variants(self, start: str, end: str) -> None:  # 5.2
        assert extract_body(_ebook("Body.", start=start, end=end)) == "Body."

    def test_missing_start_marker_raises(self) -> None:  # 5.3
        with pytest.raises(ValueError, match="START"):
            extract_body(f"{HEADER}Body.\n\n{END}{LICENSE}")

    def test_missing_end_marker_raises(self) -> None:  # 5.3
        with pytest.raises(ValueError, match="END"):
            extract_body(f"{HEADER}{START}\n\nBody.{LICENSE}")

    @pytest.mark.parametrize(
        "trailer",
        [
            "End of the Project Gutenberg EBook of The Republic, by Plato",
            "End of Project Gutenberg's Leviathan, by Thomas Hobbes",
        ],
    )
    def test_removes_the_old_style_end_line(self, trailer: str) -> None:  # 5.4
        assert extract_body(_ebook(f"Socrates speaks.\n\n\n{trailer}")) == "Socrates speaks."


class TestNormalizeText:
    def test_normalises_line_endings_and_drops_the_bom(self) -> None:  # 5.5
        raw = "\ufeffFirst line\r\nsame paragraph.\r\n\r\nSecond.\rStill second."

        assert normalize_text(raw) == "First line same paragraph.\n\nSecond. Still second."

    def test_unwraps_hard_wrapped_lines(self) -> None:  # 5.6
        assert normalize_text("It is the\ninterest of\nthe stronger.") == (
            "It is the interest of the stronger."
        )

    def test_separates_paragraphs_with_exactly_one_blank_line(self) -> None:  # 5.7
        assert normalize_text("One.\n\n\n\n\nTwo.\n \n\t\nThree.") == "One.\n\nTwo.\n\nThree."

    def test_removes_underscore_italics_but_not_inner_underscores(self) -> None:  # 5.8
        raw = "In the _Republic_ and the _res publica_ alike, snake_case stays."

        assert normalize_text(raw) == (
            "In the Republic and the res publica alike, snake_case stays."
        )

    def test_collapses_runs_of_spaces_and_tabs(self) -> None:  # 5.9
        assert normalize_text("Too   many\t\tspaces.") == "Too many spaces."


def test_clean_gutenberg_text_extracts_then_normalises() -> None:  # 5.10
    raw = _ebook("It is the\ninterest of the\nstronger.\n\n\n_Yes._").replace("\n", "\r\n")

    assert clean_gutenberg_text(raw) == "It is the interest of the stronger.\n\nYes."


_MESSY_TEXT = st.text(alphabet="ab_. \t\n\r", max_size=300)


class TestNormalizeTextProperties:
    @given(_MESSY_TEXT)
    def test_is_idempotent(self, raw: str) -> None:  # 5.11
        once = normalize_text(raw)
        assert normalize_text(once) == once

    @given(_MESSY_TEXT)
    def test_produces_tidy_lines(self, raw: str) -> None:  # 5.12
        result = normalize_text(raw)
        assert "\n\n\n" not in result
        assert all(line == line.strip() for line in result.split("\n"))

    @given(_MESSY_TEXT)
    def test_preserves_all_content_characters_in_order(self, raw: str) -> None:  # 5.13
        def content(text: str) -> str:
            return re.sub(r"[\s_]", "", text)

        assert content(normalize_text(raw)) == content(raw)
