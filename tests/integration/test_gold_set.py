"""The packaged gold sets against the real corpus (test plan: I.9).

Opt-in (`make test-integration`), and skipped unless the whole corpus is in
data/raw. A gold question whose quote is in no chunk could never be a hit,
for any model: it would quietly lower every score. This catches a typo in a
quote, a renamed section, or a chunking change that splits a quote apart.
"""

from pathlib import Path

import pytest

from aporia_rag import evaluation
from aporia_rag.config import ChunkingSettings
from aporia_rag.domain.models import Chunk
from aporia_rag.evaluation.gold import GoldQuestion, load_gold_set
from aporia_rag.ingestion.chunker import RecursiveChunker
from aporia_rag.ingestion.corpus import load_manifest
from aporia_rag.ingestion.gutenberg import cache_path
from aporia_rag.ingestion.loader import iter_documents

pytestmark = pytest.mark.integration

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
QUESTIONS = [
    *load_gold_set().questions,
    *load_gold_set(Path(evaluation.__file__).parent / "gold_loop.toml").questions,
]


@pytest.fixture(scope="module")
def chunks_by_document() -> dict[str, list[Chunk]]:
    works = load_manifest().works
    if not all(cache_path(RAW_DIR, work.gutenberg_id).is_file() for work in works):
        pytest.skip("the corpus is not fully downloaded: run `aporia-rag fetch`")
    chunking = ChunkingSettings()
    chunker = RecursiveChunker(chunk_size=chunking.size, chunk_overlap=chunking.overlap)
    return {document.id: chunker.chunk(document) for document in iter_documents(works, RAW_DIR)}


@pytest.mark.parametrize("question", QUESTIONS, ids=lambda question: question.id)
def test_every_quote_is_inside_a_chunk_of_its_document(
    question: GoldQuestion, chunks_by_document: dict[str, list[Chunk]]
) -> None:  # I.9
    for evidence in question.evidence:
        assert evidence.document in chunks_by_document, evidence.document
        chunks = chunks_by_document[evidence.document]
        assert any(evidence.is_passage_hit(chunk) for chunk in chunks), evidence.quote
