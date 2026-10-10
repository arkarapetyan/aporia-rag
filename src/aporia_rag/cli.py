"""Command-line entry point: ``aporia-rag <command>``.

    aporia-rag fetch                 download the corpus into data/raw
    aporia-rag index [--rebuild]     embed the corpus into the vector index
    aporia-rag search QUESTION       show the passages retrieval finds for a question

Each command parses its arguments, asks the `Container` for the components
it needs, and prints. Expected failures (Ollama not running, a model not
pulled, the corpus not downloaded, invalid settings) are printed as one
`error:` line with exit code 1; a traceback always means a bug.
"""

import argparse
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx

from aporia_rag.bootstrap import Container
from aporia_rag.config import Settings, load_settings
from aporia_rag.domain.errors import AporiaError
from aporia_rag.indexing.indexer import WorkProgress
from aporia_rag.ingestion.corpus import WorkSpec, load_manifest
from aporia_rag.ingestion.gutenberg import (
    DEFAULT_MIRROR,
    POLITE_DELAY_SECONDS,
    EbookNotFoundError,
    GutenbergMirror,
    fetch_cached,
)
from aporia_rag.ingestion.loader import iter_documents

type _Command = Callable[[argparse.Namespace, Container], int]

_SNIPPET_LENGTH = 200


class CommandError(AporiaError):
    """A command was used in a way that cannot work (e.g. an unknown author)."""


def main(
    argv: Sequence[str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    settings: Settings | None = None,
) -> int:
    """Run the CLI.

    `transport` replaces the network (the mirror and the model server) and
    `settings` replaces the environment; both exist for tests.
    """
    args = _parser().parse_args(argv)
    command: _Command = args.run
    try:
        with Container(settings or load_settings(), transport=transport) as container:
            return command(args, container)
    except (AporiaError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aporia-rag",
        description="AporiaRAG: a self-correcting RAG agent over primary philosophical texts.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    fetch = commands.add_parser(
        "fetch",
        help="download the corpus from a Project Gutenberg mirror into a local cache",
        description="Download every work in the corpus manifest. Already-cached works are skipped.",
    )
    fetch.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="where downloaded texts are stored (default: the configured corpus directory)",
    )
    _add_manifest_option(fetch)
    fetch.add_argument(
        "--mirror", default=DEFAULT_MIRROR, help="mirror base URL (default: %(default)s)"
    )
    fetch.add_argument(
        "--delay",
        type=float,
        default=POLITE_DELAY_SECONDS,
        help="seconds between requests; Project Gutenberg asks for 2 (default: %(default)s)",
    )
    fetch.set_defaults(run=_fetch)

    index = commands.add_parser(
        "index",
        help="embed the downloaded corpus into the vector index",
        description=(
            "Chunk and embed the corpus with the configured embedding model. Only new or "
            "changed chunks are embedded, so running it again is cheap."
        ),
    )
    index.add_argument(
        "--rebuild", action="store_true", help="empty the index first and embed everything"
    )
    _add_manifest_option(index)
    index.set_defaults(run=_index)

    search = commands.add_parser(
        "search",
        help="show the passages retrieved for a question (no LLM involved)",
        description="Embed the question and print the most similar passages, best first.",
    )
    search.add_argument("question")
    search.add_argument(
        "-k", type=_positive_int, default=5, help="how many passages (default: %(default)s)"
    )
    search.add_argument(
        "--author", help="only this author's works (any part of the name, e.g. 'kant')"
    )
    _add_manifest_option(search)
    search.set_defaults(run=_search)
    return parser


def _add_manifest_option(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="corpus manifest TOML file (default: the packaged corpus)",
    )


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {number}")
    return number


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------


def _fetch(args: argparse.Namespace, container: Container) -> int:
    works = load_manifest(args.manifest).works
    cache_dir: Path = args.cache_dir or container.settings.storage.corpus_dir
    failures = 0
    with httpx.Client(transport=container.transport, timeout=60.0, follow_redirects=True) as client:
        mirror = GutenbergMirror(client, base_url=args.mirror, delay_seconds=args.delay)
        for work in works:
            label = f"{work.id} (#{work.gutenberg_id})"
            try:
                result = fetch_cached(work.gutenberg_id, cache_dir, mirror)
            except (EbookNotFoundError, httpx.HTTPError) as error:
                failures += 1
                print(f"FAILED   {label}: {error}", file=sys.stderr)
                continue
            status = "cached " if result.from_cache else "fetched"
            print(f"{status}  {label} -> {result.path}")
    if failures:
        print(f"{failures} of {len(works)} works failed", file=sys.stderr)
    return 1 if failures else 0


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------


def _index(args: argparse.Namespace, container: Container) -> int:
    works = load_manifest(args.manifest).works
    if args.rebuild:
        container.vector_store.clear()
    print(f"indexing {len(works)} works with {container.embedder.space.id}")
    report = container.indexer.run(
        iter_documents(works, container.settings.storage.corpus_dir),
        on_progress=_print_progress,
    )
    print(
        f"done in {report.seconds:.1f}s: {report.added} added, {report.updated} updated, "
        f"{report.unchanged} unchanged, {report.deleted} deleted"
    )
    return 0


def _print_progress(progress: WorkProgress) -> None:
    print(
        f"  {progress.author}, {progress.title}: "
        f"{progress.documents} sections, {progress.chunks} chunks"
    )


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------


def _search(args: argparse.Namespace, container: Container) -> int:
    works = load_manifest(args.manifest).works
    filters = None if args.author is None else {"author": _resolve_author(args.author, works)}
    if container.vector_store.count() == 0:
        raise CommandError(
            f"the index for {container.embedder.space.id} is empty; run `aporia-rag index` first"
        )
    results = container.retriever.search(args.question, k=args.k, filters=filters)
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        print(f"{rank}. {result.score:.3f}  {chunk.metadata.citation}  [{chunk.id}]")
        print(f"   {_snippet(chunk.text)}")
    return 0


def _resolve_author(name: str, works: Sequence[WorkSpec]) -> str:
    authors = sorted({work.author for work in works})
    matches = [author for author in authors if name.casefold() in author.casefold()]
    if len(matches) != 1:
        raise CommandError(
            f"--author {name!r} matches {len(matches)} authors; authors: {', '.join(authors)}"
        )
    return matches[0]


def _snippet(text: str) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    if len(flat) <= _SNIPPET_LENGTH:
        return flat
    return flat[:_SNIPPET_LENGTH].rsplit(" ", 1)[0] + " ..."
