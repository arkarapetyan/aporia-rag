"""Tests for aporia_rag.evaluation.gold, the gold question sets (test plan: 16.1-16.3, 23.9)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from aporia_rag import evaluation
from aporia_rag.evaluation.gold import Evidence, GoldQuestion, GoldSet, load_gold_set
from aporia_rag.ingestion.corpus import load_manifest

LOOP_SET = Path(evaluation.__file__).parent / "gold_loop.toml"
KANT = {"document": "kant-metaphysic-of-morals/first-section", "quote": "except a good will"}


def _question(**overrides: object) -> GoldQuestion:
    fields: dict[str, object] = {
        "id": "kant-good-will",
        "question": "Is anything good without qualification?",
        "evidence": [KANT],
    }
    return GoldQuestion.model_validate(fields | overrides)


class TestGoldQuestion:
    @pytest.mark.parametrize("field", ["document", "quote"])
    def test_evidence_needs_a_document_and_a_quote(self, field: str) -> None:  # 16.1
        with pytest.raises(ValidationError, match=field):
            Evidence.model_validate(KANT | {field: "  "})

    @pytest.mark.parametrize("bad_id", ["Kant Good Will", "kant--will", ""])
    def test_id_must_be_a_slug(self, bad_id: str) -> None:  # 16.1
        with pytest.raises(ValidationError, match="id"):
            _question(id=bad_id)

    def test_question_must_not_be_blank(self) -> None:  # 16.1
        with pytest.raises(ValidationError, match="question"):
            _question(question=" ")

    def test_evidence_must_be_given(self) -> None:  # 16.1
        fields = _question().model_dump()
        del fields["evidence"]

        with pytest.raises(ValidationError, match="evidence"):
            GoldQuestion.model_validate(fields)

    def test_empty_evidence_is_a_question_the_corpus_does_not_answer(self) -> None:  # 16.1
        assert not _question(evidence=[]).has_evidence
        assert _question().has_evidence

    def test_rejects_unknown_fields(self) -> None:  # 16.1
        with pytest.raises(ValidationError, match="answer"):
            _question(answer="a good will")


class TestGoldSet:
    def test_needs_at_least_one_question(self) -> None:  # 16.2
        with pytest.raises(ValidationError, match="questions"):
            GoldSet(questions=[])

    def test_rejects_duplicate_question_ids(self) -> None:  # 16.2
        with pytest.raises(ValidationError, match="duplicate question id: kant-good-will"):
            GoldSet(questions=[_question(), _question(question="Another wording?")])


class TestLoadGoldSet:
    def test_parses_a_toml_file(self, tmp_path: Path) -> None:  # 16.3
        path = tmp_path / "gold.toml"
        path.write_text(
            "[[questions]]\n"
            'id = "kant-good-will"\n'
            'question = "Is anything good without qualification?"\n'
            "[[questions.evidence]]\n"
            'document = "kant-metaphysic-of-morals/first-section"\n'
            'quote = "except a good will"\n',
            encoding="utf-8",
        )

        assert load_gold_set(path) == GoldSet(questions=[_question()])

    def test_packaged_gold_set_covers_every_work(self) -> None:  # 16.3
        gold = load_gold_set()

        works = {
            evidence.document.split("/", 1)[0]
            for question in gold.questions
            for evidence in question.evidence
        }
        assert works == {work.id for work in load_manifest().works}


def test_packaged_loop_set_has_questions_with_and_without_evidence() -> None:  # 23.9
    gold = load_gold_set(LOOP_SET)

    with_evidence = [question for question in gold.questions if question.has_evidence]
    assert with_evidence
    assert len(with_evidence) < len(gold.questions)
