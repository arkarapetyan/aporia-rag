"""The web UI's backend: the page, the corpus's authors, and answers streamed step by step.

`POST /api/ask` takes `ask`'s options and answers with JSON lines
(`application/x-ndjson`): one `{"event": "step", ...}` per step of the
corrective pipeline as it finishes, then `{"event": "outcome", ...}`, or
`{"event": "error", "message": ...}` if answering failed. The page reads
the lines as they come, so the reader watches the loop work.

A request that cannot be answered (an author who is not in the corpus, an
empty index) is refused with status 400 before anything runs. One question
is answered at a time: on a laptop the chat model is the bottleneck, and two
questions at once would each take twice as long.

The rules (scope, other authors, the checks) are the pipeline's and
`retrieval.scope`'s; this module only turns requests and outcomes into JSON.
"""

import json
import logging
from collections.abc import Iterator, Sequence
from importlib.resources import files
from queue import Queue
from threading import Lock, Thread
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints

from aporia_rag.bootstrap import Container
from aporia_rag.domain.errors import AporiaError
from aporia_rag.domain.models import RetrievedChunk
from aporia_rag.pipelines.outcome import Outcome, Step
from aporia_rag.retrieval.scope import DEFAULT_K, Scope, scope_for

_log = logging.getLogger(__name__)
_STATIC = files("aporia_rag.ui").joinpath("static")


class AskRequest(BaseModel):
    """`ask`'s options."""

    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    author: str | None = None  # any part of an author's name
    all_authors: bool = False
    k: int = Field(default=DEFAULT_K, ge=1)
    others: bool = False


def create_app(container: Container, authors: Sequence[str]) -> FastAPI:
    """The app, answering from `container`'s components; `authors` are the corpus's."""
    app = FastAPI(title="AporiaRAG", docs_url=None, redoc_url=None)
    one_at_a_time = Lock()

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return _STATIC.joinpath("index.html").read_text(encoding="utf-8")

    @app.get("/api/authors")
    def corpus_authors() -> dict[str, Any]:
        return {"authors": list(authors), "k": DEFAULT_K}

    @app.post("/api/ask")
    def ask(request: AskRequest) -> StreamingResponse:
        try:
            scope = scope_for(
                request.question, authors, author=request.author, all_authors=request.all_authors
            )
            container.require_index()
        except AporiaError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        lines = _answer(container, request, scope, authors, one_at_a_time)
        return StreamingResponse(lines, media_type="application/x-ndjson")

    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
    return app


def _answer(
    container: Container,
    request: AskRequest,
    scope: Scope,
    authors: Sequence[str],
    lock: Lock,
) -> Iterator[str]:
    """The answer's JSON lines, each sent as soon as it is known.

    The pipeline runs in a thread of its own and hands each step to the
    response through a queue, so a step reaches the page when it finishes,
    not when the whole answer does.
    """
    events: Queue[dict[str, Any] | None] = Queue()

    def work() -> None:
        try:
            with lock:
                outcome = container.corrective_rag.answer(
                    request.question,
                    scope,
                    k=request.k,
                    authors=authors,
                    others=request.others,
                    on_step=lambda step: events.put(_step(step)),
                )
            events.put(_outcome(outcome, scope))
        except AporiaError as error:
            events.put({"event": "error", "message": str(error)})
        except Exception as error:  # a bug: say so on the page, and keep the traceback
            _log.exception("answering %r failed", request.question)
            events.put({"event": "error", "message": f"unexpected error: {error!r}"})
        finally:
            events.put(None)

    Thread(target=work, daemon=True).start()
    while (event := events.get()) is not None:
        yield json.dumps(event, ensure_ascii=False) + "\n"


def _step(step: Step) -> dict[str, Any]:
    return {"event": "step", "node": step.node, "summary": step.summary, "seconds": step.seconds}


def _outcome(outcome: Outcome, scope: Scope) -> dict[str, Any]:
    answer = outcome.answer
    return {
        "event": "outcome",
        "answer": None if answer is None else answer.text,
        "survey": bool(scope.survey),
        "named": list(scope.named),
        "route": outcome.route,
        "query": outcome.query,
        "outside_author": outcome.outside_author,
        "web": outcome.web,
        "web_error": outcome.web_error,
        "author_questions": list(outcome.author_questions),
        "sources": [
            {"number": number, **_passage(source)}
            for number, source in enumerate(() if answer is None else answer.sources, start=1)
        ],
        "left_out": [_passage(passage) for passage in outcome.left_out],
        "skipped": [_passage(passage) for passage in outcome.skipped],
        "others": [_passage(passage) for passage in outcome.others],
        "draft_problems": len(outcome.draft_problems),
        "problems": [
            {
                "kind": problem.kind,
                "label": problem.label,
                "detail": problem.detail,
                "sentence": problem.sentence,
            }
            for problem in outcome.problems
        ],
        "seconds": dict(outcome.seconds),
    }


def _passage(passage: RetrievedChunk) -> dict[str, Any]:
    chunk = passage.chunk
    source = chunk.metadata.source_id
    return {
        "id": chunk.id,
        "citation": chunk.metadata.citation,
        "text": chunk.text,
        "url": source if source.startswith(("http://", "https://")) else None,
    }
