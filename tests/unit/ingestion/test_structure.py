"""Tests for aporia_rag.ingestion.structure (test plan: Component 7).

Every input is a tiny synthetic edition reproducing one convention found in
the real Gutenberg files; the real files are covered by the opt-in contract
tests in tests/integration/test_corpus_structure.py.
"""

import pytest

from aporia_rag.ingestion.corpus import EditionLayout
from aporia_rag.ingestion.structure import Section, StructureError, format_label, structure_work


def _text(*paragraphs: str) -> str:
    return "\n\n".join(paragraphs)


def _layout(**overrides: object) -> EditionLayout:
    fields: dict[str, object] = {"start": r"BOOK I\.", "sections": [r"BOOK [IVX]+\."]}
    return EditionLayout.model_validate(fields | overrides)


def _labels(sections: list[Section]) -> list[str]:
    return [section.label for section in sections]


def _body(*paragraphs: str, english_only: bool = False) -> list[str]:
    """Paragraphs kept from a one-section edition made of `paragraphs`."""
    [section] = structure_work(_text("BOOK I.", *paragraphs), _layout(english_only=english_only))
    return section.text.split("\n\n")


REPUBLIC = _text(
    "THE REPUBLIC",
    "INTRODUCTION AND ANALYSIS.",
    "Jowett explains the dialogue at great length.",
    "BOOK I.",
    "I went down yesterday to the Piraeus.",
    "BOOK II.",
    "With these words I was thinking that I had made an end.",
    "INDEX",
    "Justice, nature of, 1, 2.",
)

HUME = _text(
    "SECTION IV.",
    "SCEPTICAL DOUBTS CONCERNING THE OPERATIONS OF THE UNDERSTANDING.",
    "PART I.",
    "All the objects of human reason may be divided into two kinds.",
    "PART II.",
    "But we have not yet attained any tolerable satisfaction.",
    "SECTION V.",
    "The passion for philosophy, like that for religion, seems liable to this inconvenience.",
)
HUME_LAYOUT = _layout(start=r"SECTION IV\.", sections=[r"SECTION [IVX]+\.", r"PART [IVX]+\."])


class TestBoundaries:
    def test_text_before_the_start_is_excluded(self) -> None:  # 7.1
        sections = structure_work(REPUBLIC, _layout(end="INDEX"))

        assert sections[0] == Section("Book I", "I went down yesterday to the Piraeus.")
        assert all("Jowett" not in section.text for section in sections)

    def test_text_from_the_end_onward_is_excluded(self) -> None:  # 7.2
        sections = structure_work(REPUBLIC, _layout(end="INDEX"))

        assert sections[-1] == Section(
            "Book II", "With these words I was thinking that I had made an end."
        )

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [({"start": r"BOOK XI\."}, "start"), ({"end": "APPENDIX"}, "end")],
        ids=["start", "end"],
    )
    def test_missing_boundary_raises(self, overrides: dict[str, str], message: str) -> None:  # 7.3
        with pytest.raises(StructureError, match=message):
            structure_work(REPUBLIC, _layout(**overrides))

    def test_start_must_be_a_section_heading(self) -> None:  # 7.4
        with pytest.raises(StructureError, match="heading"):
            structure_work(REPUBLIC, _layout(start=r"INTRODUCTION AND ANALYSIS\."))


class TestSections:
    def test_each_level1_heading_starts_a_section(self) -> None:  # 7.5
        assert _labels(structure_work(REPUBLIC, _layout(end="INDEX"))) == ["Book I", "Book II"]

    def test_two_levels_combine_and_a_new_level1_resets_level2(self) -> None:  # 7.6
        assert _labels(structure_work(HUME, HUME_LAYOUT)) == [
            "Section IV, Part I",
            "Section IV, Part II",
            "Section V",
        ]

    def test_headings_are_not_part_of_the_text(self) -> None:  # 7.7
        texts = [section.text for section in structure_work(HUME, HUME_LAYOUT)]

        assert not any("SECTION" in text or "PART" in text for text in texts)

    def test_heading_directly_followed_by_a_heading_yields_no_empty_section(self) -> None:  # 7.8
        text = _text("PART I. OF MAN", "CHAPTER I. OF SENSE", "Concerning the Thoughts of man.")
        layout = _layout(
            start=r"PART I\. OF MAN", sections=[r"PART [IVX]+\. .*", r"CHAPTER [IVXL]+\.? .*"]
        )

        assert structure_work(text, layout) == [
            Section("Part I. Of Man, Chapter I. Of Sense", "Concerning the Thoughts of man.")
        ]

    def test_explicit_labels_replace_level1_headings(self) -> None:  # 7.9
        text = _text(
            "THE FIRST BOOK OF THE DIALOGUES, ADDRESSED TO LUCILIUS.",
            "You have asked me, Lucilius, why so many evils befall good men.",
            "THE SECOND BOOK OF THE DIALOGUES, ADDRESSED TO SERENUS.",
            "I might truly say, Serenus, that there is a wide difference.",
        )
        layout = _layout(
            start=r"THE FIRST BOOK .*",
            sections=[r"THE \w+ BOOK OF THE DIALOGUES, .*"],
            section_labels=["Of Providence", "On the Firmness of the Wise Man"],
        )

        assert _labels(structure_work(text, layout)) == [
            "Of Providence",
            "On the Firmness of the Wise Man",
        ]

    def test_explicit_labels_must_match_the_number_of_headings(self) -> None:  # 7.10
        with pytest.raises(StructureError, match="labels"):
            structure_work(REPUBLIC, _layout(end="INDEX", section_labels=["Book I"]))


class TestSharedCleanUp:
    @pytest.mark.parametrize(
        "note",
        [
            "[3] dicendum Edd.; discendum MSS.",
            "[A] Cicero is alluding to his Republic.",
            "[AB] The universal and the individual.",
            "[Footnote A: The author has reason for believing this.]",
            "* A maxim is the subjective principle of volition.",
        ],
    )
    def test_footnote_paragraphs_are_dropped(self, note: str) -> None:  # 7.11
        assert _body("Moral goodness is our subject.", note) == ["Moral goodness is our subject."]

    @pytest.mark.parametrize(
        "apparatus",
        [
            "FOOTNOTES:",
            "#Philosophy and oratory.#",
            "[Illustration: Cicero, from a bust]",
            "SEC_1",
            "{22}",
        ],
    )
    def test_apparatus_paragraphs_are_dropped(self, apparatus: str) -> None:  # 7.12
        assert _body("Moral goodness is our subject.", apparatus) == [
            "Moral goodness is our subject."
        ]

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Glaucon[3] replied.", "Glaucon replied."),
            ("A maiden[M] showed a man's soul.", "A maiden showed a man's soul."),
            ("Cicero alludes[AB] to it.", "Cicero alludes to it."),
            ("A great living philosopher,(1) who wrote.", "A great living philosopher, who wrote."),
            ("The maxim * that I should follow.", "The maxim that I should follow."),
            ("{22} We come now to anger.", "We come now to anger."),
            ("*5* Moreover, the subject is common.", "Moreover, the subject is common."),
            (
                "Force is for brutes. #Excuse for war.# The only excuse is peace.",
                "Force is for brutes. The only excuse is peace.",
            ),
        ],
    )
    def test_inline_references_are_removed(self, raw: str, expected: str) -> None:  # 7.13
        assert _body(raw) == [expected]

    def test_greek_wrapper_is_unwrapped(self) -> None:  # 7.14
        assert _body("The virtue [Greek: arete] of the soul.") == ["The virtue arete of the soul."]

    @pytest.mark.parametrize("residue", ["OF THE ORIGIN OF IDEAS.", "27.", "I."])
    def test_paragraphs_without_lowercase_are_dropped(self, residue: str) -> None:  # 7.15
        assert _body(residue, "Every one will readily allow it.") == [
            "Every one will readily allow it."
        ]

    @pytest.mark.parametrize(
        "prose",
        ["Consider the following: (1) pleasure, (2) pain.", "I. [Serenus.] When I examine myself."],
    )
    def test_look_alikes_are_kept(self, prose: str) -> None:  # 7.16
        assert _body(prose) == [prose]


LATIN = (
    "Quamquam te, Marce fili, annum iam audientem Cratippum, abundare oportet praeceptis, "
    "quod est et doctoris et urbis."
)
ENGLISH = (
    "My dear son Marcus, you have now been studying a full year under Cratippus, "
    "and that too in Athens."
)
VERSE = "Cedant arma togae, concedat laurea laudi."  # no function word of either language


class TestEnglishOnly:
    def test_latin_is_dropped_and_english_kept(self) -> None:  # 7.17
        assert _body(LATIN, ENGLISH, english_only=True) == [ENGLISH]

    def test_accents_do_not_hide_latin(self) -> None:  # 7.18
        accented = "N\u00falla sancta s\u00f3cietas, n\u00e9c fides r\u00e9gni \u00e9st."

        assert _body(ENGLISH, accented, english_only=True) == [ENGLISH]

    def test_no_evidence_follows_a_latin_paragraph(self) -> None:  # 7.19
        assert _body(LATIN, VERSE, ENGLISH, english_only=True) == [ENGLISH]

    def test_no_evidence_follows_an_english_paragraph(self) -> None:  # 7.19
        assert _body(ENGLISH, VERSE, english_only=True) == [ENGLISH, VERSE]

    def test_one_sided_evidence_beats_the_previous_paragraph(self) -> None:  # 7.20
        latin_page_opening = "Restat quarta pars, quae decore continetur."

        assert _body(ENGLISH, latin_page_opening, english_only=True) == [ENGLISH]

    def test_latin_is_kept_without_english_only(self) -> None:  # 7.21
        assert _body(LATIN, ENGLISH) == [LATIN, ENGLISH]


class TestFormatLabel:
    @pytest.mark.parametrize(
        ("heading", "expected"),
        [
            ("BOOK IV.", "Book IV"),  # 7.22
            ("SECTION IV.", "Section IV"),
            ("CHAPTER XXVI. OF CIVILL LAWES", "Chapter XXVI. Of Civill Lawes"),  # 7.23
            ("A REVIEW, AND CONCLUSION", "A Review, and Conclusion"),  # 7.24
            ("OF DARKNESSE FROM VAIN PHILOSOPHY", "Of Darknesse from Vain Philosophy"),
            ("PART I. OF MAN", "Part I. Of Man"),
            ("ARISTOTLE'S ETHICS", "Aristotle's Ethics"),  # 7.25
            ("Chapter III.", "Chapter III"),  # 7.26
        ],
    )
    def test_formats_headings_as_citation_labels(self, heading: str, expected: str) -> None:
        assert format_label(heading) == expected
