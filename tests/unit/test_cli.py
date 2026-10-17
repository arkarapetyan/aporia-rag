"""Tests for aporia_rag.cli (test plan: Components 6, 15, 17 and 20).

The CLI is driven exactly as from a shell, through `main(argv)`. The seams
are the injected httpx transport, which serves the mirror, the embedding
model and the chat model from memory, and the injected settings, which point the corpus and
the index at a temporary directory.
"""

import json
import re
from pathlib import Path

import httpx
import pytest

from aporia_rag.cli import main
from aporia_rag.config import Settings, StorageSettings
from support.fakes import FakeOllamaServer, refusing_transport

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


def _run(*argv: str, manifest: Path, settings: Settings, transport: httpx.BaseTransport) -> int:
    return main([*argv, "--manifest", str(manifest)], transport=transport, settings=settings)


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
    clock = iter([100.0, 100.4, 112.9])  # start, passages retrieved, answer generated
    monkeypatch.setattr("aporia_rag.cli.perf_counter", lambda: next(clock))

    exit_code = _run(
        "ask", GOOD_WILL_QUESTION,
        manifest=index_manifest, settings=settings, transport=server.transport,
    )  # fmt: skip

    assert exit_code == 0
    assert capsys.readouterr().out.endswith(
        "\nanswered in 12.9s (retrieval 0.4s, generation 12.5s)\n"
    )


def test_ask_releases_both_models_connections(
    index_manifest: Path, settings: Settings
) -> None:  # 20.5
    server = FakeOllamaServer()
    _index(index_manifest, settings, FakeOllamaServer())

    _run("ask", "virtue", manifest=index_manifest, settings=settings, transport=server.transport)

    assert server.transport.closes == 2  # the embedder's client and the chat model's
