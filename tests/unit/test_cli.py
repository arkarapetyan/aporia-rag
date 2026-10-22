"""Tests for aporia_rag.cli (test plan: Components 6, 15, 17, 20 and 23).

The CLI is driven exactly as from a shell, through `main(argv)`. The seams
are the injected httpx transport, which serves the mirror, the embedding
model and the chat model from memory, the injected settings, which point the corpus and
the index at a temporary directory, and the injected web search (ddgs does not use httpx).
"""

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from aporia_rag.cli import main
from aporia_rag.config import ChatSettings, Settings, StorageSettings
from aporia_rag.domain.errors import ProviderError
from aporia_rag.domain.models import web_passage
from aporia_rag.generation.generator import SURVEY_PROMPT
from support.fakes import RELEVANT, FakeOllamaServer, ScriptedWebSearch, refusing_transport

BASE_URL = "https://mirror.test"
MIRROR_FILES = {
    "/1/4/9/1497/1497-0.txt": b"The Republic",
    "/5/59/59-0.txt": b"Discourse on the Method",
}


class RecordingServer:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        content = self.files.get(request.url.path)
        return httpx.Response(200, content=content) if content else httpx.Response(404)


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.toml"
    layout = "[works.layout]\nstart = 'BOOK I'\nsections = ['BOOK .*']\n"
    path.write_text(
        '[[works]]\nid = "plato-republic"\nauthor = "Plato"\ntitle = "The Republic"\n'
        f"gutenberg_id = 1497\n{layout}\n"
        '[[works]]\nid = "descartes-discourse"\nauthor = "Descartes"\n'
        f'title = "Discourse on the Method"\ngutenberg_id = 59\n{layout}',
        encoding="utf-8",
    )
    return path


def _fetch(manifest: Path, cache_dir: Path, server: RecordingServer) -> int:
    argv = [
        "fetch",
        "--manifest", str(manifest),
        "--cache-dir", str(cache_dir),
        "--mirror", BASE_URL,
        "--delay", "0",
    ]  # fmt: skip
    return main(argv, transport=httpx.MockTransport(server))


def test_fetch_downloads_every_work_and_reports_each(
    manifest: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:  # 6.1
    cache_dir = tmp_path / "raw"

    exit_code = _fetch(manifest, cache_dir, RecordingServer(MIRROR_FILES))

    assert exit_code == 0
    assert (cache_dir / "pg1497.txt").read_text(encoding="utf-8") == "The Republic"
    assert (cache_dir / "pg59.txt").read_text(encoding="utf-8") == "Discourse on the Method"
    output = capsys.readouterr().out
    assert "plato-republic" in output
    assert "descartes-discourse" in output


def test_second_run_is_served_from_the_cache(
    manifest: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:  # 6.2
    server = RecordingServer(MIRROR_FILES)
    _fetch(manifest, tmp_path / "raw", server)
    requests_after_first_run = len(server.requests)

    exit_code = _fetch(manifest, tmp_path / "raw", server)

    assert exit_code == 0
    assert len(server.requests) == requests_after_first_run
    assert capsys.readouterr().out.count("cached") == 2


def test_missing_work_is_reported_and_the_rest_still_fetched(
    manifest: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:  # 6.3
    server = RecordingServer({"/5/59/59-0.txt": b"Discourse on the Method"})

    exit_code = _fetch(manifest, tmp_path / "raw", server)

    assert exit_code == 1
    assert (tmp_path / "raw" / "pg59.txt").is_file()
    assert "plato-republic" in capsys.readouterr().err


# --------------------------------------------------------------------------
# index and search (Component 15)
# --------------------------------------------------------------------------

KANT_BODY = (
    "FIRST SECTION\n\n"
    "Nothing can possibly be conceived in the world, or even out of it, which can be\n"
    "called good without qualification, except a good will.\n\n"
    "SECOND SECTION\n\n"
    "Everything in nature works according to laws."
)
HOBBES_BODY = (
    "CHAPTER XIII\n\n"
    "Whatsoever therefore is consequent to a time of Warre, where every man is Enemy to\n"
    "every man; the same is consequent to the time, wherein men live without other\n"
    "security, than what their own strength, and their own invention shall furnish them\n"
    "withall. In such condition, there is no place for Industry; and the life of man,\n"
    "solitary, poore, nasty, brutish, and short."
)
INDEX_MANIFEST = """
[[works]]
id = "kant-groundwork"
author = "Immanuel Kant"
title = "Fundamental Principles of the Metaphysic of Morals"
translator = "Thomas Kingsmill Abbott"
gutenberg_id = 5682
[works.layout]
start = 'FIRST SECTION'
sections = ['(FIRST|SECOND) SECTION']

[[works]]
id = "hobbes-leviathan"
author = "Thomas Hobbes"
title = "Leviathan"
gutenberg_id = 3207
[works.layout]
start = 'CHAPTER XIII'
sections = ['CHAPTER [IVXL]+']
"""


def _as_downloaded(body: str) -> str:
    return (
        "*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\n"
        f"{body}\n\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\n"
    )


@pytest.fixture
def index_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.toml"
    path.write_text(INDEX_MANIFEST, encoding="utf-8")
    return path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    corpus_dir = tmp_path / "raw"
    corpus_dir.mkdir()
    (corpus_dir / "pg5682.txt").write_text(_as_downloaded(KANT_BODY), encoding="utf-8")
    (corpus_dir / "pg3207.txt").write_text(_as_downloaded(HOBBES_BODY), encoding="utf-8")
    return Settings(
        _env_file=None,
        storage=StorageSettings(corpus_dir=corpus_dir, index_dir=tmp_path / "index"),
    )


def _run(
    *argv: str,
    manifest: Path,
    settings: Settings,
    transport: httpx.BaseTransport,
    web: ScriptedWebSearch | None = None,
) -> int:
    return main(
        [*argv, "--manifest", str(manifest)],
        transport=transport,
        settings=settings,
        web_search=web or ScriptedWebSearch(),  # finds nothing
    )


def test_index_reports_progress_and_counts(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 15.1
    server = FakeOllamaServer()

    exit_code = _run(
        "index", manifest=index_manifest, settings=settings, transport=server.transport
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Immanuel Kant" in output
    assert "Thomas Hobbes" in output
    assert "3 added, 0 updated, 0 unchanged, 0 deleted" in output


def test_second_index_embeds_nothing(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 15.2
    server = FakeOllamaServer()
    _run("index", manifest=index_manifest, settings=settings, transport=server.transport)
    requests_after_first_run = len(server.requests)

    exit_code = _run(
        "index", manifest=index_manifest, settings=settings, transport=server.transport
    )

    assert exit_code == 0
    assert len(server.requests) == requests_after_first_run
    assert "0 added, 0 updated, 3 unchanged, 0 deleted" in capsys.readouterr().out


def test_rebuild_embeds_everything_again(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 15.3
    server = FakeOllamaServer()
    _run("index", manifest=index_manifest, settings=settings, transport=server.transport)
    requests_after_first_run = len(server.requests)
    capsys.readouterr()

    exit_code = _run(
        "index", "--rebuild", manifest=index_manifest, settings=settings, transport=server.transport
    )

    assert exit_code == 0
    assert len(server.requests) > requests_after_first_run
    assert "3 added, 0 updated, 0 unchanged, 0 deleted" in capsys.readouterr().out


def _index(manifest: Path, settings: Settings, server: FakeOllamaServer) -> None:
    assert _run("index", manifest=manifest, settings=settings, transport=server.transport) == 0


def test_search_prints_ranked_cited_results(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 15.4
    server = FakeOllamaServer()
    _index(index_manifest, settings, server)
    capsys.readouterr()

    exit_code = _run(
        "search",
        "What can be called good without qualification?",
        "-k",
        "2",
        manifest=index_manifest,
        settings=settings,
        transport=server.transport,
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    first_result = re.compile(
        r"^1\. +-?[01]\.\d{3} +Immanuel Kant, Fundamental Principles of the Metaphysic of "
        r"Morals, First Section +\[kant-groundwork/first-section#0\]$",
        re.MULTILINE,
    )
    assert first_result.search(output), output
    assert "except a good will" in output  # the snippet
    assert re.search(r"^2\. ", output, re.MULTILINE)
    assert not re.search(r"^3\. ", output, re.MULTILINE)


def test_search_needs_a_positive_k(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 15.4
    with pytest.raises(SystemExit):
        _run(
            "search", "virtue", "-k", "0",
            manifest=index_manifest, settings=settings, transport=FakeOllamaServer().transport,
        )  # fmt: skip

    assert "must be positive" in capsys.readouterr().err


def test_search_can_be_restricted_to_one_author(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 15.5
    server = FakeOllamaServer()
    _index(index_manifest, settings, server)
    capsys.readouterr()

    exit_code = _run(
        "search",
        "What can be called good without qualification?",
        "--author",
        "hobbes",  # any part of the name, any case
        manifest=index_manifest,
        settings=settings,
        transport=server.transport,
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Thomas Hobbes" in output
    assert "Immanuel Kant" not in output
    assert output.rstrip().endswith("...")  # a long passage is shortened to a snippet


def test_search_for_an_unknown_author_lists_the_authors(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 15.5
    server = FakeOllamaServer()
    _index(index_manifest, settings, server)

    exit_code = _run(
        "search",
        "virtue",
        "--author",
        "spinoza",
        manifest=index_manifest,
        settings=settings,
        transport=server.transport,
    )

    assert exit_code == 1
    error = capsys.readouterr().err
    assert "spinoza" in error
    assert "Immanuel Kant" in error
    assert "Thomas Hobbes" in error


def test_search_on_an_empty_index_says_to_run_index(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 15.6
    server = FakeOllamaServer()

    exit_code = _run(
        "search", "virtue", manifest=index_manifest, settings=settings, transport=server.transport
    )

    assert exit_code == 1
    assert "aporia-rag index" in capsys.readouterr().err
    assert server.requests == []  # no point embedding a question with nothing to search


def test_search_all_authors_shows_the_best_passage_of_each_author(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 15.7
    server = FakeOllamaServer()
    _index(index_manifest, settings, server)
    capsys.readouterr()

    exit_code = _run(
        "search", "What can be called good without qualification?", "--all-authors",
        manifest=index_manifest, settings=settings, transport=server.transport,
    )  # fmt: skip

    assert exit_code == 0
    assert _shown(capsys.readouterr().out) == [
        "kant-groundwork/first-section#0",  # Kant's best; his second section is left out
        "hobbes-leviathan/chapter-xiii#0",
    ]


def test_search_cannot_take_all_authors_and_one_author(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 15.7
    with pytest.raises(SystemExit):
        _run(
            "search", "virtue", "--author", "kant", "--all-authors",
            manifest=index_manifest, settings=settings, transport=FakeOllamaServer().transport,
        )  # fmt: skip

    assert "not allowed with argument --author" in capsys.readouterr().err


SCOPE_LINE = "Searching only the authors named in the question: "


def _shown(output: str) -> list[str]:
    """The chunk ids of the results `search` printed, in rank order."""
    return re.findall(r"^\d+\. .*\[(.+)\]$", output, re.MULTILINE)


@pytest.mark.parametrize(
    ("question", "first_line", "shown"),
    [
        (
            "What does hobbes call good without qualification?",
            f"{SCOPE_LINE}Thomas Hobbes.",
            ["hobbes-leviathan/chapter-xiii#0"],
        ),
        (
            "What do Kant and Hobbes call good without qualification?",
            f"{SCOPE_LINE}Immanuel Kant, Thomas Hobbes.",
            ["hobbes-leviathan/chapter-xiii#0", "kant-groundwork/first-section#0"],  # one each
        ),
    ],
    ids=["one-author", "two-authors"],
)
def test_search_keeps_to_the_authors_named_in_the_question(
    index_manifest: Path,
    settings: Settings,
    capsys: pytest.CaptureFixture[str],
    *,
    question: str,
    first_line: str,
    shown: list[str],
) -> None:  # 15.8
    server = FakeOllamaServer()
    _index(index_manifest, settings, server)
    capsys.readouterr()

    exit_code = _run(
        "search", question, manifest=index_manifest, settings=settings, transport=server.transport
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert output.startswith(f"{first_line}\n\n1. ")
    assert _shown(output) == shown


@pytest.mark.parametrize(
    ("option", "shown"),
    [
        ("--author=kant", ["kant-groundwork/first-section#0"]),
        ("--all-authors", ["kant-groundwork/first-section#0", "hobbes-leviathan/chapter-xiii#0"]),
    ],
)
def test_search_options_take_precedence_over_the_names_in_the_question(
    index_manifest: Path,
    settings: Settings,
    capsys: pytest.CaptureFixture[str],
    option: str,
    shown: list[str],
) -> None:  # 15.8
    server = FakeOllamaServer()
    _index(index_manifest, settings, server)
    capsys.readouterr()

    exit_code = _run(
        "search", "What does Hobbes call good without qualification?", option, "-k", "1",
        manifest=index_manifest, settings=settings, transport=server.transport,
    )  # fmt: skip

    assert exit_code == 0
    output = capsys.readouterr().out
    assert SCOPE_LINE not in output
    assert sorted(_shown(output)) == sorted(shown)


def test_ollama_not_running_is_a_one_line_error(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 15.7
    exit_code = _run(
        "index", manifest=index_manifest, settings=settings, transport=refusing_transport()
    )

    assert exit_code == 1
    error = capsys.readouterr().err
    assert error.startswith("error: ")
    assert "Ollama" in error
    assert len(error.strip().splitlines()) == 1


def test_index_before_fetch_says_to_run_fetch(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 15.8
    for downloaded in settings.storage.corpus_dir.iterdir():
        downloaded.unlink()

    exit_code = _run(
        "index", manifest=index_manifest, settings=settings, transport=FakeOllamaServer().transport
    )

    assert exit_code == 1
    assert "aporia-rag fetch" in capsys.readouterr().err


def test_corrupt_corpus_file_is_a_one_line_error(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 15.9
    corrupt = settings.storage.corpus_dir / "pg5682.txt"
    corrupt.write_text("<html>404 Not Found</html>", encoding="utf-8")

    exit_code = _run(
        "index", manifest=index_manifest, settings=settings, transport=FakeOllamaServer().transport
    )

    assert exit_code == 1
    error = capsys.readouterr().err
    assert error.startswith("error: kant-groundwork")
    assert len(error.strip().splitlines()) == 1


@pytest.mark.parametrize("command", [["index"], ["search", "virtue"]])
def test_commands_release_their_connections(
    index_manifest: Path, settings: Settings, command: list[str]
) -> None:  # 15.10
    server = FakeOllamaServer()
    _index(index_manifest, settings, FakeOllamaServer())  # so `search` has something to search

    _run(*command, manifest=index_manifest, settings=settings, transport=server.transport)

    assert server.transport.closed


# --------------------------------------------------------------------------
# evaluate (Component 17)
# --------------------------------------------------------------------------

GOLD = """
[[questions]]
id = "good-will"
question = "What can be called good without qualification?"
[[questions.evidence]]
document = "kant-groundwork/first-section"
quote = "except a good will"

[[questions]]
id = "state-of-nature"
question = "What is the life of man like without a common power?"
[[questions.evidence]]
document = "hobbes-leviathan/chapter-xiii"
quote = "nasty, brutish, and short"
"""
QWEN = "qwen3-embedding:0.6b"
GEMMA = "embeddinggemma:300m"


@pytest.fixture
def gold(tmp_path: Path) -> Path:
    path = tmp_path / "gold.toml"
    path.write_text(GOLD, encoding="utf-8")
    return path


def _with_model(settings: Settings, model: str) -> Settings:
    return settings.model_copy(
        update={"embedding": settings.embedding.model_copy(update={"model": model})}
    )


def _evaluate(gold: Path, settings: Settings, server: FakeOllamaServer, *models: str) -> int:
    argv = ["evaluate", "--gold", str(gold)]
    for model in models:
        argv += ["--model", model]
    return main(argv, transport=server.transport, settings=settings)


def test_evaluate_prints_metrics_and_every_questions_rank(
    index_manifest: Path, gold: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 17.1
    server = FakeOllamaServer()
    _index(index_manifest, settings, server)
    capsys.readouterr()

    exit_code = _evaluate(gold, settings, server)

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "2 gold questions" in output
    assert re.search(r"Hit@1 +Hit@5 +Hit@10 +MRR +Hit@5 +MRR", output), output
    # The whole test corpus is 3 chunks, so every answer is somewhere in the top 10.
    assert re.search(rf"^{re.escape(QWEN)} +.* 1\.00 ", output, re.MULTILINE), output
    assert re.search(r"^good-will +\d \(\d\)$", output, re.MULTILINE), output
    assert re.search(r"^state-of-nature +\d \(\d\)$", output, re.MULTILINE), output


def test_evaluate_compares_models_on_their_own_indexes(
    index_manifest: Path, gold: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 17.2
    server = FakeOllamaServer(models=[QWEN, GEMMA])
    for model in (QWEN, GEMMA):
        _index(index_manifest, _with_model(settings, model), server)
    capsys.readouterr()
    server.requests.clear()

    exit_code = _evaluate(gold, settings, server, QWEN, GEMMA)

    assert exit_code == 0
    assert {request["model"] for request in server.requests} == {QWEN, GEMMA}
    output = capsys.readouterr().out
    assert re.search(rf"^question +{re.escape(QWEN)} +{re.escape(GEMMA)}$", output, re.MULTILINE)
    assert re.search(r"^good-will +\d \(\d\) +\d \(\d\)$", output, re.MULTILINE), output


def test_evaluate_a_model_without_an_index_says_how_to_build_it(
    index_manifest: Path, gold: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 17.3
    server = FakeOllamaServer(models=[QWEN, GEMMA])
    _index(index_manifest, settings, server)  # only the default model is indexed

    exit_code = _evaluate(gold, settings, server, QWEN, GEMMA)

    assert exit_code == 1
    error = capsys.readouterr().err
    assert error.startswith("error: ")
    assert f"APORIA_EMBEDDING__MODEL={GEMMA} aporia-rag index" in error
    assert len(error.strip().splitlines()) == 1


def test_evaluate_releases_every_models_connections(
    index_manifest: Path, gold: Path, settings: Settings
) -> None:  # 17.4
    for model in (QWEN, GEMMA):
        _index(index_manifest, _with_model(settings, model), FakeOllamaServer(models=[model]))
    server = FakeOllamaServer(models=[QWEN, GEMMA])

    _evaluate(gold, settings, server, QWEN, GEMMA)

    assert server.transport.closes == 2  # one embedder client per model


def test_evaluate_dump_writes_every_retrieved_chunk(
    index_manifest: Path,
    gold: Path,
    settings: Settings,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:  # 17.5
    server = FakeOllamaServer()
    _index(index_manifest, settings, server)
    capsys.readouterr()
    _evaluate(gold, settings, server)
    report = capsys.readouterr().out
    dump = tmp_path / "eval" / "retrievals.jsonl"  # the folder is created

    exit_code = main(
        ["evaluate", "--gold", str(gold), "--dump", str(dump)],
        transport=server.transport,
        settings=settings,
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert output.startswith(report)  # the same report, then where the dump went
    assert str(dump) in output.removeprefix(report)
    records = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
    # 2 questions x all 3 chunks of the test corpus (fewer than the top 10)
    assert len(records) == 6
    assert {record["model"] for record in records} == {QWEN}
    assert {record["question_id"] for record in records} == {"good-will", "state-of-nature"}
    assert any(record["passage_hit"] for record in records)


OUTSIDE_THE_CORPUS = """
[[questions]]
id = "outside-the-corpus"
question = "What is a language game?"
evidence = []
"""


def test_evaluate_does_not_score_questions_without_evidence(
    index_manifest: Path,
    settings: Settings,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:  # 17.6
    server = FakeOllamaServer()
    _index(index_manifest, settings, server)
    capsys.readouterr()
    gold = tmp_path / "gold.toml"
    gold.write_text(GOLD + OUTSIDE_THE_CORPUS, encoding="utf-8")

    exit_code = _evaluate(gold, settings, server)

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "3 gold questions (1 without evidence, not scored)" in output
    assert "(one question is 50.0 points)" in output
    assert re.search(rf"^{re.escape(QWEN)} +1\.00 ", output, re.MULTILINE), output
    assert re.search(r"^outside-the-corpus +no evidence$", output, re.MULTILINE), output


def _good_will_only(prompt: str) -> str:
    """Keep the Kant passages for the good-will question, and nothing for any other question."""
    question = prompt.rsplit("Question:", 1)[-1]
    keep = "good without qualification" in question and "Thomas Hobbes" not in prompt
    return f'{{"relevant": {str(keep).lower()}}}'


def test_evaluate_grade_compares_the_grader_with_the_gold_set(
    index_manifest: Path, gold: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 23.7
    server = FakeOllamaServer(structured_reply=_good_will_only)
    _index(index_manifest, settings, server)
    capsys.readouterr()
    _evaluate(gold, settings, server)
    report = capsys.readouterr().out

    exit_code = main(
        ["evaluate", "--gold", str(gold), "--grade"], transport=server.transport, settings=settings
    )

    assert exit_code == 0
    assert len(server.structured_requests) == 6  # 2 questions x the 3 chunks of the test corpus
    assert not server.chat_requests
    output = capsys.readouterr().out
    assert output.startswith(report)  # the retrieval report first, unchanged
    grading = output.removeprefix(report)
    assert f"relevance grader {ChatSettings().model} on the chunks {QWEN} retrieved" in grading
    # good-will: both Kant chunks kept, one of them evidence; state-of-nature: nothing kept
    for k in (5, 10):
        row = rf"^top {k} +6 +2 +2 +1 +0\.50 +0\.50 +\d+\.\d$"
        assert re.search(row, grading, re.MULTILINE), grading
    assert re.search(r"^ +needless \(grader only\): state-of-nature$", grading, re.MULTILINE)
    assert re.search(r"^ +missed \(gold set only\): -$", grading, re.MULTILINE)
    disagreements = [
        r"^good-will +\d +kant-groundwork/second-section#0 +kept, not evidence$",
        r"^state-of-nature +\d +hobbes-leviathan/chapter-xiii#0 +evidence, left out$",
    ]
    for disagreement in disagreements:
        assert re.search(disagreement, grading, re.MULTILINE), grading


def test_evaluate_without_grade_never_asks_the_chat_model(
    index_manifest: Path, gold: Path, settings: Settings
) -> None:  # 23.8
    server = FakeOllamaServer()
    _index(index_manifest, settings, server)

    _evaluate(gold, settings, server)

    assert not server.structured_requests
    assert not server.chat_requests


# --------------------------------------------------------------------------
# ask (Component 20)
# --------------------------------------------------------------------------

GOOD_WILL_QUESTION = "What can be called good without qualification?"
REPLY = "Only a good will is good without qualification [1]."


def _passage_headers(text: str) -> list[str]:
    """The "[n] citation" lines of a prompt or of the printed passage list."""
    return re.findall(r"^\[\d+\] [^(\n]+?(?= \(|$)", text, re.MULTILINE)


def test_ask_prints_the_answer_and_the_passages_numbered_as_the_model_saw_them(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 20.1
    server = FakeOllamaServer(reply=REPLY)
    _index(index_manifest, settings, server)
    capsys.readouterr()

    exit_code = _run(
        "ask", GOOD_WILL_QUESTION, "-k", "2",
        manifest=index_manifest, settings=settings, transport=server.transport,
    )  # fmt: skip

    assert exit_code == 0
    output = capsys.readouterr().out
    assert output.startswith(f"{REPLY}\n\nPassages:\n")
    assert (
        "[1] Immanuel Kant, Fundamental Principles of the Metaphysic of Morals, First Section "
        "(kant-groundwork/first-section#0)"
    ) in output
    [chat] = server.chat_requests
    prompt = chat["messages"][-1]["content"]
    assert len(_passage_headers(output)) == 2
    assert _passage_headers(output) == _passage_headers(prompt)


def test_ask_takes_k_and_author_like_search(
    index_manifest: Path, settings: Settings
) -> None:  # 20.2
    server = FakeOllamaServer()
    _index(index_manifest, settings, server)

    exit_code = _run(
        "ask", GOOD_WILL_QUESTION, "-k", "1", "--author", "hobbes",
        manifest=index_manifest, settings=settings, transport=server.transport,
    )  # fmt: skip

    assert exit_code == 0
    [chat] = server.chat_requests
    assert _passage_headers(chat["messages"][-1]["content"]) == [
        "[1] Thomas Hobbes, Leviathan, Chapter XIII"  # not Kant, the best match overall
    ]


def test_ask_on_an_empty_index_says_to_run_index(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 20.3
    server = FakeOllamaServer()

    exit_code = _run(
        "ask", "virtue", manifest=index_manifest, settings=settings, transport=server.transport
    )

    assert exit_code == 1
    assert "aporia-rag index" in capsys.readouterr().err
    assert server.requests == []  # neither embedded nor asked


def test_ask_with_a_chat_model_not_pulled_says_how_to_pull_it(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 20.4
    server = FakeOllamaServer(models=[QWEN])  # the embedding model only
    _index(index_manifest, settings, server)
    capsys.readouterr()

    exit_code = _run(
        "ask", "virtue", manifest=index_manifest, settings=settings, transport=server.transport
    )

    assert exit_code == 1
    error = capsys.readouterr().err
    assert error.startswith("error: ")
    assert f"ollama pull {settings.chat.model}" in error
    assert len(error.strip().splitlines()) == 1


def test_ask_reports_how_long_retrieval_and_generation_took(
    index_manifest: Path,
    settings: Settings,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # 20.6
    server = FakeOllamaServer()
    _index(index_manifest, settings, server)
    capsys.readouterr()
    clock = iter([100.0, 115.1])  # start, done
    monkeypatch.setattr("aporia_rag.cli.perf_counter", lambda: next(clock))
    # retrieve, grade, generate, other authors: start and end of each
    stages = iter([0.0, 0.4, 0.4, 6.4, 6.4, 12.9, 12.9, 15.0])
    monkeypatch.setattr("aporia_rag.pipelines.corrective.perf_counter", lambda: next(stages))

    exit_code = _run(
        "ask", GOOD_WILL_QUESTION, "--others",
        manifest=index_manifest, settings=settings, transport=server.transport,
    )  # fmt: skip

    assert exit_code == 0
    assert capsys.readouterr().out.endswith(
        "\nanswered in 15.1s (retrieval 0.4s, grading 6.0s, generation 6.5s, other authors 2.1s)\n"
    )


def _not_hobbes(prompt: str) -> str:
    return f'{{"relevant": {str("Thomas Hobbes" not in prompt).lower()}}}'


REWRITTEN = "nasty brutish short"
OUTSIDE = "Ludwig Wittgenstein"
PAGE = web_passage(
    url="https://en.wikipedia.org/wiki/Language_game",
    title="Language game",
    text="A language-game is a philosophical concept developed by Ludwig Wittgenstein.",
)


def _verdicts_and_rewrite(
    is_relevant: Callable[[str], bool], query: str = REWRITTEN, author: str = "Thomas Hobbes"
) -> Callable[[str], str]:
    """Verdicts by `is_relevant`; the rewriter (its prompt is the bare question) gets `query`,
    naming `author` as the one most likely to answer."""

    def reply(prompt: str) -> str:
        if prompt.startswith("Question: "):
            return f'{{"query": "{query}", "author": "{author}"}}'
        return f'{{"relevant": {str(is_relevant(prompt)).lower()}}}'

    return reply


def _rewrite_line(query: str = REWRITTEN) -> str:
    return f'Nothing relevant was found, so the question was searched again as: "{query}"'


def test_ask_answers_from_the_relevant_passages_and_lists_those_left_out(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 20.7
    server = FakeOllamaServer(reply=REPLY, structured_reply=_not_hobbes)
    _index(index_manifest, settings, server)
    capsys.readouterr()

    exit_code = _run(
        "ask", GOOD_WILL_QUESTION, "-k", "3",
        manifest=index_manifest, settings=settings, transport=server.transport,
    )  # fmt: skip

    assert exit_code == 0
    assert len(server.structured_requests) == 3  # one verdict per retrieved passage
    [chat] = server.chat_requests
    prompt = chat["messages"][-1]["content"]
    assert "Thomas Hobbes" not in prompt
    assert _passage_headers(prompt) == [
        "[1] Immanuel Kant, Fundamental Principles of the Metaphysic of Morals, First Section",
        "[2] Immanuel Kant, Fundamental Principles of the Metaphysic of Morals, Second Section",
    ]
    output = capsys.readouterr().out
    assert _passage_headers(output) == _passage_headers(prompt)
    assert (
        "\nLeft out as not relevant:\n"
        "- Thomas Hobbes, Leviathan, Chapter XIII (hobbes-leviathan/chapter-xiii#0)\n"
    ) in output


def test_ask_with_no_relevant_passage_says_so_without_an_answer(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 20.8
    # The first search finds Hobbes and Kant's first section; the rewritten one, Kant's second.
    query = "Everything in nature works according to laws"
    server = FakeOllamaServer(structured_reply=_verdicts_and_rewrite(lambda prompt: False, query))
    _index(index_manifest, settings, server)
    capsys.readouterr()

    web = ScriptedWebSearch()  # finds nothing

    exit_code = _run(
        "ask", "What is a picture theory of meaning?", "-k", "2",
        manifest=index_manifest, settings=settings, transport=server.transport, web=web,
    )  # fmt: skip

    assert exit_code == 0
    assert server.chat_requests == []  # graded, but never asked for an answer
    assert web.calls == [("What is a picture theory of meaning?", 2)]
    output = capsys.readouterr().out
    assert output.startswith(
        f"{_rewrite_line(query)}\n\n"
        "Nothing relevant was found again, so the web was searched.\n\n"
        "None of the 3 passages found answers this question.\n\n"
        "Left out as not relevant:\n"
    )
    left_out = output.split("Left out as not relevant:\n", 1)[1].split("\n\n", 1)[0]
    assert len(left_out.splitlines()) == 3  # both searches, so every decision can be checked
    assert all(re.match(r"^- .+ \(.+#\d+\)$", line) for line in left_out.splitlines())
    assert "Other authors" not in output


HOBBES_POINTER = (
    "\nOther authors who discuss this:\n"
    "- Thomas Hobbes, Leviathan, Chapter XIII (hobbes-leviathan/chapter-xiii#0)\n"
)


@pytest.mark.parametrize("hobbes_is_relevant", [True, False])
def test_ask_lists_the_other_authors_who_discuss_the_question(
    index_manifest: Path,
    settings: Settings,
    capsys: pytest.CaptureFixture[str],
    hobbes_is_relevant: bool,
) -> None:  # 20.9
    verdict = f'{{"relevant": {str(hobbes_is_relevant).lower()}}}'
    server = FakeOllamaServer(
        structured_reply=lambda prompt: verdict if "Thomas Hobbes" in prompt else RELEVANT
    )
    _index(index_manifest, settings, server)
    capsys.readouterr()

    exit_code = _run(
        "ask", GOOD_WILL_QUESTION, "-k", "1", "--others",  # retrieves Kant only
        manifest=index_manifest, settings=settings, transport=server.transport,
    )  # fmt: skip

    assert exit_code == 0
    assert len(server.structured_requests) == 2  # the Kant passage, then Hobbes's best passage
    assert (HOBBES_POINTER in capsys.readouterr().out) is hobbes_is_relevant


def test_ask_for_one_author_does_not_look_for_others(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 20.10
    server = FakeOllamaServer()
    _index(index_manifest, settings, server)
    capsys.readouterr()

    exit_code = _run(
        "ask", GOOD_WILL_QUESTION, "-k", "1", "--author", "kant",
        manifest=index_manifest, settings=settings, transport=server.transport,
    )  # fmt: skip

    assert exit_code == 0
    assert len(server.structured_requests) == 1  # the Kant passage only
    assert "Other authors" not in capsys.readouterr().out


def _not_hobbes_then_survey(prompt: str) -> str:
    if prompt.startswith("[1] "):  # the survey: the numbered passages, then the question
        return '{"paragraphs": ["Only a good will is good without qualification."]}'
    return _not_hobbes(prompt)


def test_ask_all_authors_surveys_the_relevant_best_passage_of_each_author(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 20.11
    server = FakeOllamaServer(structured_reply=_not_hobbes_then_survey)
    _index(index_manifest, settings, server)
    indexed = len(server.requests)
    capsys.readouterr()

    exit_code = _run(
        "ask", GOOD_WILL_QUESTION, "--all-authors",
        manifest=index_manifest, settings=settings, transport=server.transport,
    )  # fmt: skip

    assert exit_code == 0
    embedded = [body for body in server.requests[indexed:] if "input" in body]
    assert len(embedded) == 1  # the question, once: no search for other authors
    assert server.chat_requests == []
    *verdicts, survey = server.structured_requests
    assert len(verdicts) == 2  # Kant's best passage and Hobbes's
    assert survey["messages"][0]["content"] == SURVEY_PROMPT
    assert _passage_headers(survey["messages"][-1]["content"]) == [
        "[1] Immanuel Kant, Fundamental Principles of the Metaphysic of Morals, First Section"
    ]
    output = capsys.readouterr().out
    assert output.startswith(
        "Immanuel Kant [1]\nOnly a good will is good without qualification.\n\nPassages:\n"
    )
    assert (
        "\nLeft out as not relevant:\n"
        "- Thomas Hobbes, Leviathan, Chapter XIII (hobbes-leviathan/chapter-xiii#0)\n"
    ) in output
    assert "Other authors" not in output


def test_ask_answers_from_the_one_author_named_in_the_question(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 20.12
    server = FakeOllamaServer(reply=REPLY)
    _index(index_manifest, settings, server)
    capsys.readouterr()

    exit_code = _run(
        "ask", "What does Hobbes call good without qualification?",
        manifest=index_manifest, settings=settings, transport=server.transport,
    )  # fmt: skip

    assert exit_code == 0
    assert len(server.structured_requests) == 1  # Hobbes's one passage: no search for others
    [chat] = server.chat_requests
    assert _passage_headers(chat["messages"][-1]["content"]) == [
        "[1] Thomas Hobbes, Leviathan, Chapter XIII"
    ]
    output = capsys.readouterr().out
    assert output.startswith(f"{SCOPE_LINE}Thomas Hobbes.\n\n{REPLY}\n")
    assert "Other authors" not in output


def _subject_verdicts_then_survey_of_two(prompt: str) -> str:
    if prompt.startswith("[1] "):  # the survey: the numbered passages, then the question
        return '{"paragraphs": ["The first.", "The second."]}'
    if "\nAuthors compared: " in prompt:  # the subject of the comparison
        return '{"subject": "the good"}'
    return RELEVANT


def test_ask_compares_the_authors_named_in_the_question(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 20.13
    server = FakeOllamaServer(structured_reply=_subject_verdicts_then_survey_of_two)
    _index(index_manifest, settings, server)
    capsys.readouterr()

    exit_code = _run(
        "ask", "How do Kant and Hobbes differ on the good?",
        manifest=index_manifest, settings=settings, transport=server.transport,
    )  # fmt: skip

    assert exit_code == 0
    _subject, *verdicts, survey = server.structured_requests
    asked = [
        "What does Immanuel Kant say about the good?",
        "What does Thomas Hobbes say about the good?",
    ]
    # the best passage of each named author, graded against that author's question
    assert [
        verdict["messages"][-1]["content"].rsplit("Question: ", 1)[1] for verdict in verdicts
    ] == asked
    assert survey["messages"][0]["content"] == SURVEY_PROMPT
    assert capsys.readouterr().out.startswith(
        f"{SCOPE_LINE}Immanuel Kant, Thomas Hobbes.\n\n"
        "Asked of each author on their own:\n"
        f"- {asked[0]}\n- {asked[1]}\n\n"
        "Immanuel Kant [1]\nThe first.\n\nThomas Hobbes [2]\nThe second.\n"  # in the names' order
    )


def test_ask_searches_again_with_a_rewritten_query_when_nothing_is_relevant(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 20.14
    server = FakeOllamaServer(
        reply=REPLY,
        structured_reply=_verdicts_and_rewrite(lambda prompt: "Thomas Hobbes" in prompt),
    )
    _index(index_manifest, settings, server)
    capsys.readouterr()

    exit_code = _run(
        "ask", GOOD_WILL_QUESTION, "-k", "1",  # finds Kant first, who is not kept
        manifest=index_manifest, settings=settings, transport=server.transport,
    )  # fmt: skip

    assert exit_code == 0
    [chat] = server.chat_requests
    assert chat["messages"][-1]["content"].endswith(f"Question: {GOOD_WILL_QUESTION}")
    output = capsys.readouterr().out
    assert output.startswith(
        f"{_rewrite_line()}\n\n{REPLY}\n\nPassages:\n"
        "[1] Thomas Hobbes, Leviathan, Chapter XIII (hobbes-leviathan/chapter-xiii#0)\n"
    )
    assert (
        "\nLeft out as not relevant:\n"
        "- Immanuel Kant, Fundamental Principles of the Metaphysic of Morals, First Section "
        "(kant-groundwork/first-section#0)\n"
    ) in output
    assert re.search(r"\nanswered in [\d.]+s \(rewriting [\d.]+s, retrieval [\d.]+s, ", output)


def test_ask_releases_both_models_connections(
    index_manifest: Path, settings: Settings
) -> None:  # 20.5
    server = FakeOllamaServer()
    _index(index_manifest, settings, FakeOllamaServer())

    _run("ask", "virtue", manifest=index_manifest, settings=settings, transport=server.transport)

    assert server.transport.closes == 2  # the embedder's client and the chat model's


KANT_LEFT_OUT = (
    "Left out as not relevant:\n"
    "- Immanuel Kant, Fundamental Principles of the Metaphysic of Morals, First Section "
    "(kant-groundwork/first-section#0)\n"
)


def _only_the_web_page(prompt: str) -> bool:
    return prompt.startswith("Passage from en.wikipedia.org")


def test_ask_answers_from_the_web_when_the_author_is_outside_the_corpus(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 20.15
    server = FakeOllamaServer(
        reply=REPLY,
        structured_reply=_verdicts_and_rewrite(_only_the_web_page, author=OUTSIDE),
    )
    _index(index_manifest, settings, server)
    capsys.readouterr()
    web = ScriptedWebSearch([PAGE])

    exit_code = _run(
        "ask", GOOD_WILL_QUESTION, "-k", "1", "--others",  # finds Kant first, not kept
        manifest=index_manifest, settings=settings, transport=server.transport, web=web,
    )  # fmt: skip

    assert exit_code == 0
    assert web.calls == [(GOOD_WILL_QUESTION, 1)]
    # Kant's verdict, the rewrite, the page's verdict: no verdict on another author (Hobbes)
    assert len(server.structured_requests) == 3
    [chat] = server.chat_requests
    assert _passage_headers(chat["messages"][-1]["content"]) == [
        "[1] en.wikipedia.org, Language game"
    ]
    output = capsys.readouterr().out
    assert output.startswith(
        "Nothing relevant was found, and the rewriter named no author in the corpus "
        f'("{OUTSIDE}"), so the web was searched.\n\n'
        f"{REPLY}\n\nWeb sources:\n"
        "[1] en.wikipedia.org, Language game (https://en.wikipedia.org/wiki/Language_game)\n\n"
        f"{KANT_LEFT_OUT}"
    )
    assert "Other authors" not in output
    assert re.search(
        r"\nanswered in [\d.]+s \(rewriting [\d.]+s, retrieval [\d.]+s, grading [\d.]+s, "
        r"web search [\d.]+s, generation [\d.]+s\)\n$",
        output,
    )


def test_ask_says_why_when_the_web_search_fails(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 20.16
    server = FakeOllamaServer(
        structured_reply=_verdicts_and_rewrite(lambda prompt: False, author=OUTSIDE)
    )
    _index(index_manifest, settings, server)
    capsys.readouterr()
    web = ScriptedWebSearch(failure=ProviderError("ddgs: 202 Ratelimit"))

    exit_code = _run(
        "ask", GOOD_WILL_QUESTION, "-k", "1", "--others",  # Kant only: Hobbes is another
        manifest=index_manifest, settings=settings, transport=server.transport, web=web,
    )  # fmt: skip

    assert exit_code == 0
    assert capsys.readouterr().out.startswith(
        "Nothing relevant was found, and the rewriter named no author in the corpus "
        f'("{OUTSIDE}"), so the web was searched; the search failed (ddgs: 202 Ratelimit).\n\n'
        "The one passage found does not answer this question.\n\n"
        f"{KANT_LEFT_OUT}\n"
    )
    assert len(server.structured_requests) == 2  # Kant's verdict and the rewrite: nothing else


def test_ask_looks_for_other_authors_only_when_asked(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 20.17
    server = FakeOllamaServer()  # everything is relevant: Hobbes would be listed
    _index(index_manifest, settings, server)
    capsys.readouterr()

    exit_code = _run(
        "ask", GOOD_WILL_QUESTION, "-k", "1",  # retrieves Kant only
        manifest=index_manifest, settings=settings, transport=server.transport,
    )  # fmt: skip

    assert exit_code == 0
    assert len(server.structured_requests) == 1  # the Kant passage only
    output = capsys.readouterr().out
    assert "Other authors" not in output
    assert "other authors" not in output.rsplit("\nanswered in ", 1)[1]


def test_ask_lists_the_passages_it_did_not_grade(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 20.18
    server = FakeOllamaServer(reply=REPLY)  # everything is relevant
    _index(index_manifest, settings, server)
    capsys.readouterr()

    exit_code = _run(
        "ask", GOOD_WILL_QUESTION, "-k", "3",  # Kant's first section, Hobbes, Kant's second
        manifest=index_manifest, settings=settings, transport=server.transport,
    )  # fmt: skip

    assert exit_code == 0
    assert len(server.structured_requests) == 2  # 2 kept: the third is not graded
    output = capsys.readouterr().out
    assert len(_passage_headers(output.split("\n\nNot graded", 1)[0])) == 2
    assert (
        "\nNot graded, as 2 relevant passages came first:\n"
        "- Immanuel Kant, Fundamental Principles of the Metaphysic of Morals, Second Section "
        "(kant-groundwork/second-section#0)\n"
    ) in output


def test_ask_says_when_it_wrote_the_answer_again_and_what_still_fails(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 20.19
    # The fake server gives the same reply twice: the problem remains after the revision.
    server = FakeOllamaServer(reply='Kant calls it "the only good thing" [1].')
    _index(index_manifest, settings, server)
    capsys.readouterr()

    exit_code = _run(
        "ask", GOOD_WILL_QUESTION, "-k", "1",
        manifest=index_manifest, settings=settings, transport=server.transport,
    )  # fmt: skip

    assert exit_code == 0
    assert len(server.chat_requests) == 2  # the answer, then once more
    output = capsys.readouterr().out
    assert (
        "\nThe first answer had 1 problem with its passages, so it was written again.\n"
        "\nStill not borne out by its passages:\n"
        '- a quotation not word for word: "the only good thing"\n'
    ) in output
    assert re.search(r"generation [\d.]+s, revision [\d.]+s\)\n$", output)


# --------------------------------------------------------------------------
# evaluate --answers (Component 29)
# --------------------------------------------------------------------------

# A made-up citation: every answer, from either system, has a citation problem.
MADE_UP_CITATION = "Only a good will is good without qualification [4]."


def _kept_for_evaluation(prompt: str) -> bool:
    """Kant for the good-will question, the web page for the language game, nothing else."""
    question = prompt.rsplit("Question:", 1)[-1]
    if "good without qualification" in question:
        return "Immanuel Kant" in prompt
    return "language game" in question and "wikipedia.org" in prompt


def _evaluate_answers(
    index_manifest: Path, gold: Path, settings: Settings, tmp_path: Path
) -> tuple[FakeOllamaServer, ScriptedWebSearch, Path]:
    server = FakeOllamaServer(
        reply=MADE_UP_CITATION, structured_reply=_verdicts_and_rewrite(_kept_for_evaluation)
    )
    _index(index_manifest, settings, server)
    outside = tmp_path / "outside.toml"
    outside.write_text(OUTSIDE_THE_CORPUS, encoding="utf-8")
    web = ScriptedWebSearch([PAGE])
    answers = tmp_path / "eval" / "answers.jsonl"
    server.requests.clear()
    exit_code = main(
        [
            "evaluate", "--gold", str(gold), "--gold", str(outside),
            "--answers", str(answers), "--manifest", str(index_manifest),
        ],
        transport=server.transport, settings=settings, web_search=web,
    )  # fmt: skip
    assert exit_code == 0
    return server, web, answers


def _records(answers: Path, system: str) -> list[dict[str, Any]]:
    """The saved answers of one system, in gold-set order."""
    lines = answers.read_text(encoding="utf-8").splitlines()
    return [record for record in map(json.loads, lines) if record["system"] == system]


def test_evaluate_answers_compares_plain_and_corrective_rag(
    index_manifest: Path,
    gold: Path,
    settings: Settings,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:  # 29.7, 29.10
    capsys.readouterr()

    _, _, answers = _evaluate_answers(index_manifest, gold, settings, tmp_path)

    output = capsys.readouterr().out
    assert "3 gold questions (1 without evidence, not scored)" in output  # both sets, as one
    report = output.split("\nanswers to ", 1)[1]
    assert report.startswith(
        "3 gold questions (2 with evidence, 1 without), plain RAG from the top 5 passages\n"
    )
    rows = [
        r"^ +plain +corrective$",
        r"^answered +3 +2$",
        r"^  given a gold passage \(of 2\) +2 +1$",
        r"^gold share of the passages given +0\.33 +0\.50$",
        r"^outside questions answered \(of 1\)$",
        r"^  from the corpus +1 +0$",
        r"^  from the web +0 +1$",
        r"^  not answered +0 +0$",
        r"^answers with a check problem +3 +2$",
        r"^  written again +0 +2$",
        r"^median seconds per question +\d+\.\d +\d+\.\d$",
        r"^plain: routes plain 3; problems citation 3$",
        r"^corrective: routes normal 1, web 2; problems citation 2$",
    ]
    for row in rows:
        assert re.search(row, report, re.MULTILINE), report
    assert report.endswith(f"\nanswers written to {answers}\n")


def test_evaluate_without_gold_evaluates_the_packaged_retrieval_set(
    index_manifest: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:  # 29.10
    server = FakeOllamaServer()
    _index(index_manifest, settings, server)

    exit_code = main(["evaluate"], transport=server.transport, settings=settings)

    assert exit_code == 0
    assert "\n40 gold questions, top 10 chunks each" in capsys.readouterr().out


def test_plain_rag_answers_once_from_every_passage_and_is_checked(
    index_manifest: Path, gold: Path, settings: Settings, tmp_path: Path
) -> None:  # 29.8
    server, _, answers = _evaluate_answers(index_manifest, gold, settings, tmp_path)

    chats = [body for body in server.requests if "messages" in body]
    plain, first_of_the_graph = chats[:3], chats[3]
    # Plain RAG first: one answer per question, from all 3 passages of the test corpus (k = 5).
    assert all("format" not in body for body in plain)
    assert [len(_passage_headers(body["messages"][-1]["content"])) for body in plain] == [3, 3, 3]
    assert "format" in first_of_the_graph  # the graph's first verdict
    plain_records = _records(answers, "plain")
    assert [record["problems"] for record in plain_records] == [["citation"]] * 3
    assert {record["route"] for record in plain_records} == {"plain"}
    assert not any(record["revised"] for record in plain_records)


def test_the_corrective_system_is_the_graph(
    index_manifest: Path, gold: Path, settings: Settings, tmp_path: Path
) -> None:  # 29.9
    _, web, answers = _evaluate_answers(index_manifest, gold, settings, tmp_path)

    graph = {record["question_id"]: record for record in _records(answers, "corrective")}
    assert list(graph) == ["good-will", "state-of-nature", "outside-the-corpus"]
    good_will, state_of_nature, outside = graph.values()
    assert good_will["route"] == "normal"
    assert {source["id"].split("#")[0] for source in good_will["sources"]} == {
        "kant-groundwork/first-section",
        "kant-groundwork/second-section",
    }
    assert [source["gold"] for source in good_will["sources"]].count(True) == 1
    assert good_will["revised"] is True  # its made-up citation, once more
    assert good_will["problems"] == ["citation"]
    # Nothing kept, a rewrite, a second search, then the web: no page answers this one.
    assert state_of_nature["route"] == "web"
    assert state_of_nature["answer"] is None
    assert outside["route"] == "web"
    assert [source["id"] for source in outside["sources"]] == [PAGE.id]
    assert outside["sources"][0]["gold"] is False
    assert [query for query, _ in web.calls] == [
        "What is the life of man like without a common power?",
        "What is a language game?",
    ]
