"""Tests for aporia_rag.ui, the web UI's backend (test plan: 31.4-31.9).

The app runs on a real `Container` whose model server is a `FakeOllamaServer`
and whose web search is scripted, over a two-section index built here.
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from aporia_rag.bootstrap import Container
from aporia_rag.config import Settings, StorageSettings
from aporia_rag.domain.models import Document, SourceMetadata, web_passage
from aporia_rag.retrieval.scope import DEFAULT_K
from aporia_rag.ui.app import create_app
from support.fakes import FakeOllamaServer, ScriptedWebSearch

AUTHORS = ("Immanuel Kant", "Thomas Hobbes")
KANT = Document(
    id="kant-groundwork/first-section",
    text="Nothing can be called good without qualification, except a good will.",
    metadata=SourceMetadata(
        author="Immanuel Kant",
        title="Groundwork",
        source_id="test:kant",
        section="First Section",
    ),
)
HOBBES = Document(
    id="hobbes-leviathan/chapter-xiii",
    text="And the life of man, solitary, poore, nasty, brutish, and short.",
    metadata=SourceMetadata(
        author="Thomas Hobbes",
        title="Leviathan",
        source_id="test:hobbes",
        section="Chapter XIII",
    ),
)
PAGE = web_passage(
    url="https://en.wikipedia.org/wiki/Language_game",
    title="Language game",
    text="A language-game is a philosophical concept developed by Ludwig Wittgenstein.",
)
GOOD_WILL = "What can be called good without qualification?"


def _kant_or_the_page(prompt: str) -> str:
    """Verdicts: Kant for the good-will question, the web page for the language game."""
    if prompt.startswith("Question: "):  # the rewriter
        return '{"author": "Ludwig Wittgenstein", "query": "language game"}'
    if prompt.startswith("[1] "):  # a survey, of the one relevant passage
        return '{"paragraphs": ["Kant on the good will."]}'
    question = prompt.rsplit("Question:", 1)[-1]
    keep = ("good without" in question and "Immanuel Kant" in prompt) or (
        "language game" in question and "wikipedia.org" in prompt
    )
    return f'{{"relevant": {str(keep).lower()}}}'


@pytest.fixture
def server() -> FakeOllamaServer:
    return FakeOllamaServer(
        reply="Only a good will is good without qualification [1].",
        structured_reply=_kant_or_the_page,
    )


@pytest.fixture
def container(tmp_path: Path, server: FakeOllamaServer) -> Iterator[Container]:
    settings = Settings(_env_file=None, storage=StorageSettings(index_dir=tmp_path / "index"))
    web = ScriptedWebSearch([PAGE])
    with Container(settings, transport=server.transport, web_search=web) as built:
        built.indexer.run([KANT, HOBBES])
        yield built


@pytest.fixture
def client(container: Container) -> TestClient:
    return TestClient(create_app(container, AUTHORS))


def _ask(client: TestClient, **request: Any) -> list[dict[str, Any]]:
    response = client.post("/api/ask", json=request)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in response.text.splitlines()]


def test_the_page_and_its_files_are_served(client: TestClient) -> None:  # 31.4
    page = client.get("/")
    script = client.get("/static/app.js")
    style = client.get("/static/style.css")

    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert '<script src="/static/app.js"' in page.text
    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]
    assert style.status_code == 200
    assert style.headers["content-type"].startswith("text/css")


def test_the_authors_and_the_default_number_of_passages(client: TestClient) -> None:  # 31.5
    assert client.get("/api/authors").json() == {"authors": list(AUTHORS), "k": DEFAULT_K}


def test_an_answer_streams_its_steps_then_the_outcome(client: TestClient) -> None:  # 31.6
    lines = _ask(client, question=GOOD_WILL)

    *steps, outcome = lines
    assert [line["event"] for line in steps] == ["step"] * 4
    assert [line["node"] for line in steps] == ["retrieve", "grade", "generate", "check"]
    assert steps[0]["summary"] == "Found 2 passages"
    assert all(line["seconds"] >= 0 for line in steps)
    assert outcome["event"] == "outcome"
    assert outcome["answer"] == "Only a good will is good without qualification [1]."
    assert outcome["route"] == "normal"
    assert outcome["survey"] is False
    assert outcome["sources"] == [
        {
            "number": 1,
            "id": "kant-groundwork/first-section#0",
            "citation": "Immanuel Kant, Groundwork, First Section",
            "text": KANT.text,
            "url": None,
        }
    ]
    assert [passage["id"] for passage in outcome["left_out"]] == ["hobbes-leviathan/chapter-xiii#0"]
    assert outcome["skipped"] == []
    assert outcome["others"] == []
    assert outcome["problems"] == []
    assert outcome["draft_problems"] == 0
    assert outcome["query"] is None
    assert outcome["web"] is False
    assert set(outcome["seconds"]) == {"retrieval", "grading", "generation"}


def test_a_web_answer_gives_the_pages_url_and_the_rewriters_author(client: TestClient) -> None:
    # 31.6: nothing in the corpus; the rewriter names an author outside it, so the web is searched
    *_, outcome = _ask(client, question="What is a language game?")

    assert outcome["route"] == "web"
    assert outcome["outside_author"] == "Ludwig Wittgenstein"
    [source] = outcome["sources"]
    assert source["url"] == "https://en.wikipedia.org/wiki/Language_game"
    assert source["citation"] == "en.wikipedia.org, Language game"


def test_the_options_are_those_of_ask(client: TestClient, server: FakeOllamaServer) -> None:
    # 31.7
    one_author = _ask(client, question=GOOD_WILL, author="hobbes", k=1)
    with_others = _ask(client, question=GOOD_WILL, k=1, others=True)
    survey = _ask(client, question=GOOD_WILL, all_authors=True)

    graded = [passage["id"] for passage in one_author[-1]["left_out"]]
    assert graded[0] == "hobbes-leviathan/chapter-xiii#0"  # then the web page
    assert not any(passage.startswith("kant") for passage in graded)
    assert one_author[-1]["named"] == []  # set by the reader, not named in the question
    assert with_others[-2]["node"] == "other_authors"
    assert with_others[-1]["others"] == []  # Hobbes's passage does not answer it
    assert survey[-1]["survey"] is True


def test_a_problem_in_the_answer_is_reported_with_its_label(
    client: TestClient, server: FakeOllamaServer
) -> None:  # 31.6
    server.reply = "A good will alone is good [4]."  # twice: the problem remains

    *_, outcome = _ask(client, question=GOOD_WILL)

    assert outcome["draft_problems"] == 1
    assert outcome["problems"] == [
        {
            "kind": "citation",
            "label": "a citation with no passage",
            "detail": "[4]",
            "sentence": "A good will alone is good [4].",
        }
    ]


@pytest.mark.parametrize(
    ("request_body", "status"),
    [
        ({"question": GOOD_WILL, "author": "spinoza"}, 400),
        ({"question": "   "}, 422),
        ({"question": GOOD_WILL, "k": 0}, 422),
    ],
    ids=["unknown-author", "blank-question", "no-passages"],
)
def test_a_request_that_cannot_be_answered_is_refused_before_any_step(
    client: TestClient, server: FakeOllamaServer, request_body: dict[str, Any], status: int
) -> None:  # 31.8
    requests_before = len(server.requests)

    response = client.post("/api/ask", json=request_body)

    assert response.status_code == status
    assert len(server.requests) == requests_before  # nothing ran
    if status == 400:
        assert "'spinoza' matches 0 authors" in response.json()["detail"]


def test_an_empty_index_is_refused_with_the_command_that_builds_it(tmp_path: Path) -> None:
    # 31.8
    settings = Settings(_env_file=None, storage=StorageSettings(index_dir=tmp_path / "empty"))
    with Container(settings, transport=FakeOllamaServer().transport) as empty:
        response = TestClient(create_app(empty, AUTHORS)).post(
            "/api/ask", json={"question": GOOD_WILL}
        )

    assert response.status_code == 400
    assert "aporia-rag index" in response.json()["detail"]


def test_a_failure_while_answering_ends_the_stream_with_its_reason(
    client: TestClient, server: FakeOllamaServer
) -> None:  # 31.9
    server.status = 500  # the model server fails from now on

    lines = _ask(client, question=GOOD_WILL)

    assert lines[-1]["event"] == "error"
    assert "simulated server failure" in lines[-1]["message"]


def test_a_bug_while_answering_ends_the_stream_and_is_logged(
    client: TestClient,
    container: Container,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:  # 31.9
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(container.corrective_rag, "answer", fail)
    # import-linter's run in test_architecture.py configures logging and disables the loggers
    # that exist by then, this one among them; the app itself never does.
    monkeypatch.setattr(logging.getLogger("aporia_rag.ui.app"), "disabled", False)

    lines = _ask(client, question=GOOD_WILL)

    assert lines == [{"event": "error", "message": "unexpected error: RuntimeError('boom')"}]
    assert "answering 'What can be called good without qualification?' failed" in caplog.text
