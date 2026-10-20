"""Command-line entry point: ``aporia-rag <command>``.

    aporia-rag fetch                 download the corpus into data/raw
    aporia-rag index [--rebuild]     embed the corpus into the vector index
    aporia-rag search QUESTION       show the passages retrieval finds for a question
    aporia-rag ask QUESTION          answer from the relevant passages, citing them
    aporia-rag evaluate [--model M]  measure retrieval on the gold questions [--dump PATH]
                                     [--grade: and the relevance grader]

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
from time import perf_counter

import httpx

from aporia_rag.bootstrap import Container
from aporia_rag.config import Settings, load_settings
from aporia_rag.domain.errors import AporiaError
from aporia_rag.domain.models import RetrievedChunk
from aporia_rag.evaluation.gold import GoldSet, load_gold_set
from aporia_rag.evaluation.grading import (
    GradingReport,
    agreement,
    evaluate_grading,
    loop_starts,
    seconds_per_question,
)
from aporia_rag.evaluation.retrieval import (
    DEPTH,
    RetrievalReport,
    evaluate_retrieval,
    hit_rate,
    mean_reciprocal_rank,
    write_retrievals,
)
from aporia_rag.graph.corrective import Scope
from aporia_rag.graph.corrective import search as search_in_scope
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
from aporia_rag.retrieval.names import authors_named_in

type _Command = Callable[[argparse.Namespace, Container], int]

_SNIPPET_LENGTH = 200
# How many other authors `ask` grades after an answer. Each verdict is one
# model call (2-4 s on an 8 GB laptop GPU), so this bounds the extra wait.
_OTHER_AUTHORS = 4


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
    _add_question_options(search)
    search.set_defaults(run=_search)

    ask = commands.add_parser(
        "ask",
        help="answer a question from the relevant retrieved passages, citing them",
        description=(
            "Retrieve the passages most similar to the question, as `search` does, keep the "
            "ones the chat model judges relevant, and have it answer from them, citing each "
            "passage by its number. With --all-authors, the answer has a paragraph per author."
        ),
    )
    _add_question_options(ask)
    ask.set_defaults(run=_ask)

    evaluate = commands.add_parser(
        "evaluate",
        help="measure retrieval on the gold questions (Hit@k, MRR)",
        description=(
            f"Search every gold question for the top {DEPTH} chunks and report where the "
            "evidence is found. Each model is evaluated on its own index, which must exist."
        ),
    )
    evaluate.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="MODEL",
        help="embedding model to evaluate; repeat to compare (default: the configured model)",
    )
    evaluate.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="gold question set TOML file (default: the packaged set)",
    )
    evaluate.add_argument(
        "--dump",
        type=Path,
        default=None,
        metavar="PATH",
        help="also write every retrieved chunk, with its verdicts, to PATH (JSON Lines)",
    )
    evaluate.add_argument(
        "--grade",
        action="store_true",
        help="also grade every retrieved chunk with the chat model and compare with the gold set "
        "(one model call per chunk)",
    )
    evaluate.set_defaults(run=_evaluate)
    return parser


def _add_question_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("question")
    command.add_argument(
        "-k",
        type=_positive_int,
        default=5,
        help="how many passages (default: %(default)s; --all-authors takes one per author)",
    )
    scope = command.add_mutually_exclusive_group()
    scope.add_argument(
        "--author",
        help="only this author's works (any part of the name, e.g. 'kant'); without it and "
        "--all-authors, the authors named in the question",
    )
    scope.add_argument(
        "--all-authors",
        action="store_true",
        help="the best passage of each author, instead of the best passages overall",
    )
    _add_manifest_option(command)


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
    scope = _scope(args, load_manifest(args.manifest).works)
    _require_index(container)
    _print_scope(scope)
    found = search_in_scope(container.retriever, args.question, scope, k=args.k)
    for rank, result in enumerate(found, start=1):
        chunk = result.chunk
        print(f"{rank}. {result.score:.3f}  {chunk.metadata.citation}  [{chunk.id}]")
        print(f"   {_snippet(chunk.text)}")
    return 0


def _scope(args: argparse.Namespace, works: Sequence[WorkSpec]) -> Scope:
    """`--author` and `--all-authors` decide; without them, the authors the question names."""
    if args.author is not None:
        return Scope(author=_resolve_author(args.author, works))
    authors = _authors(works)
    if args.all_authors:
        return Scope(survey=tuple(authors))
    match authors_named_in(args.question, authors):
        case []:
            return Scope()
        case [author]:
            return Scope(author=author, named=(author,))
        case named:
            return Scope(survey=tuple(named), named=tuple(named))


def _print_scope(scope: Scope) -> None:
    if scope.named:
        print(f"Searching only the authors named in the question: {', '.join(scope.named)}.\n")


def _authors(works: Sequence[WorkSpec]) -> list[str]:
    return sorted({work.author for work in works})


def _resolve_author(name: str, works: Sequence[WorkSpec]) -> str:
    authors = _authors(works)
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


def _require_index(container: Container) -> None:
    """Refuse to search an empty index: every query would quietly find nothing."""
    if container.vector_store.count() == 0:
        space = container.embedder.space
        raise CommandError(
            f"the index for {space.id} is empty; build it first: "
            f"`APORIA_EMBEDDING__MODEL={space.model} aporia-rag index`"
        )


# --------------------------------------------------------------------------
# ask
# --------------------------------------------------------------------------


# The stages of an answer, in the order the time report lists them.
_STAGES = ("rewriting", "retrieval", "grading", "generation")


def _ask(args: argparse.Namespace, container: Container) -> int:
    works = load_manifest(args.manifest).works
    scope = _scope(args, works)
    _require_index(container)
    _print_scope(scope)
    started = perf_counter()
    outcome = container.corrective_rag.answer(
        args.question, scope, k=args.k, authors=_authors(works)
    )
    if outcome.author_questions:
        print("Asked of each author on their own:")
        for question in outcome.author_questions:
            print(f"- {question}")
        print()
    if outcome.query is not None:
        print("Nothing relevant was found, so the question was searched again as: ", end="")
        print(f'"{outcome.query}"\n')
    if outcome.answer is not None:
        print(outcome.answer.text)
        print("\nPassages:")
        for number, source in enumerate(outcome.answer.sources, start=1):
            print(f"[{number}] {_label(source)}")
    else:
        print(f"None of the {len(outcome.graded)} passages found answers this question.")
    if outcome.left_out:
        print("\nLeft out as not relevant:")
        for passage in outcome.left_out:
            print(f"- {_label(passage)}")
    answered = perf_counter()
    others = (
        _other_authors(args.question, container, works, outcome.graded)
        if scope.unrestricted
        else []
    )
    if others:
        print("\nOther authors who discuss this:")
        for passage in others:
            print(f"- {_label(passage)}")
    pointed = perf_counter()
    stages = [
        f"{stage} {outcome.seconds[stage]:.1f}s" for stage in _STAGES if stage in outcome.seconds
    ]
    stages.append(f"other authors {pointed - answered:.1f}s")
    print(f"\nanswered in {pointed - started:.1f}s ({', '.join(stages)})")
    return 0


def _other_authors(
    question: str, container: Container, works: Sequence[WorkSpec], found: Sequence[RetrievedChunk]
) -> list[RetrievedChunk]:
    """The best relevant passage of each author who had no passage among those found.

    Top-k retrieval can fill every slot with one author (Aristotle has 63
    chunks on friendship, Cicero 6), so each other author gets a search of
    their own; the grader, not the similarity score, decides who is listed.
    """
    present = {passage.chunk.metadata.author for passage in found}
    others = [author for author in _authors(works) if author not in present]
    candidates = container.retriever.best_per_author(question, others, limit=_OTHER_AUTHORS)
    return container.relevance_grader.relevant(question, candidates)


def _label(passage: RetrievedChunk) -> str:
    return f"{passage.chunk.metadata.citation} ({passage.chunk.id})"


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------

_METRICS = ("Hit@1", "Hit@5", f"Hit@{DEPTH}", "MRR", "Hit@5", "MRR")
_COLUMN = 7  # width of one metric column
_NO_EVIDENCE = "no evidence"
# The grader is measured at the default `k` of `ask` and at the full depth of the evaluation.
_GRADING_DEPTHS = (5, DEPTH)


def _evaluate(args: argparse.Namespace, container: Container) -> int:
    gold = load_gold_set(args.gold)
    models: list[str] = args.models or [container.settings.embedding.model]
    reports = {model: _evaluate_model(model, gold, container) for model in models}
    count = len(gold.questions)
    scored = sum(1 for question in gold.questions if question.has_evidence)
    unscored = f" ({count - scored} without evidence, not scored)" if scored < count else ""
    print(
        f"{count} gold questions{unscored}, top {DEPTH} chunks each "
        f"(one question is {100 / scored:.1f} points)\n"
    )
    _print_metrics(reports)
    print(
        "\npassage hit: the chunk contains the evidence quote; "
        "section hit: it only comes from the evidence section\n"
    )
    _print_ranks(gold, reports)
    if args.grade:
        grader = container.relevance_grader
        for model, report in reports.items():
            grading = evaluate_grading(report, gold, grader)
            _print_grading(model, container.settings.chat.model, grading)
    if args.dump is not None:
        write_retrievals(args.dump, reports, gold)
        print(f"\nretrieved chunks written to {args.dump}")
    return 0


def _evaluate_model(model: str, gold: GoldSet, container: Container) -> RetrievalReport:
    """Evaluate one embedding model on its own index, with otherwise the same settings."""
    settings = container.settings
    embedding = settings.embedding.model_copy(update={"model": model})
    model_settings = settings.model_copy(update={"embedding": embedding})
    with Container(model_settings, transport=container.transport) as model_container:
        _require_index(model_container)
        return evaluate_retrieval(model_container.retriever, gold)


def _print_metrics(reports: dict[str, RetrievalReport]) -> None:
    width = max(len("model"), *map(len, reports))
    passage_width, section_width = 4 * _COLUMN, 2 * _COLUMN
    labels = f"{'':{width}} {'passage hit':^{passage_width}}{'section hit':^{section_width}}"
    print(labels.rstrip())
    print(f"{'model':{width}} " + "".join(f"{name:>{_COLUMN}}" for name in _METRICS))
    for model, report in reports.items():
        passage, section = report.passage_ranks, report.section_ranks
        values = (
            hit_rate(passage, 1),
            hit_rate(passage, 5),
            hit_rate(passage, DEPTH),
            mean_reciprocal_rank(passage),
            hit_rate(section, 5),
            mean_reciprocal_rank(section),
        )
        print(f"{model:{width}} " + "".join(f"{value:>{_COLUMN}.2f}" for value in values))


def _print_ranks(gold: GoldSet, reports: dict[str, RetrievalReport]) -> None:
    print(f"rank of the first passage hit (section hit); - means not in the top {DEPTH}")
    width = max(len("question"), *(len(question.id) for question in gold.questions)) + 2
    columns = [max(len(model), len(_NO_EVIDENCE)) + 2 for model in reports]
    header = f"{'question':{width}}" + "".join(
        f"{model:{column}}" for model, column in zip(reports, columns, strict=True)
    )
    print(header.rstrip())
    for index, question in enumerate(gold.questions):
        cells = []
        for report, column in zip(reports.values(), columns, strict=True):
            result = report.results[index]
            if result.has_evidence:
                cell = f"{result.passage_rank or '-'} ({result.section_rank or '-'})"
            else:
                cell = _NO_EVIDENCE
            cells.append(f"{cell:{column}}")
        print(f"{question.id:{width}}{''.join(cells)}".rstrip())


def _print_grading(model: str, chat_model: str, report: GradingReport) -> None:
    """How the grader's verdicts on one model's retrievals meet the gold set."""
    print(f"\nrelevance grader {chat_model} on the chunks {model} retrieved")
    headers = ("chunks", "evidence", "kept", "kept evidence", "precision", "recall", "s/question")
    print(f"{'':7}" + "".join(f"{header:>{len(header) + 2}}" for header in headers))
    for k in _GRADING_DEPTHS:
        counts = agreement(report, k)
        values = (
            counts.total,
            counts.evidence,
            counts.kept,
            counts.kept_evidence,
            _ratio(counts.precision),
            _ratio(counts.recall),
            f"{seconds_per_question(report, k):.1f}",
        )
        cells = "".join(
            f"{value:>{len(header) + 2}}" for header, value in zip(headers, values, strict=True)
        )
        print(f"{f'top {k}':7}{cells}")
    print(
        "precision: the share of the kept chunks that are evidence; "
        "recall: the share of the evidence that is kept"
    )
    for k in _GRADING_DEPTHS:
        starts = loop_starts(report, k)
        print(f"\nno relevant chunk in the top {k}, so the corrective loop would start:")
        print(f"  needed (gold set and grader): {_ids(starts.needed)}")
        print(f"  needless (grader only): {_ids(starts.needless)}")
        print(f"  missed (gold set only): {_ids(starts.missed)}")
    _print_disagreements(report)


def _print_disagreements(report: GradingReport) -> None:
    rows = [
        (question.question_id, verdict)
        for question in report.questions
        for verdict in question.verdicts
        if verdict.evidence != verdict.relevant
    ]
    print(f"\ndisagreements with the gold set ({len(rows)}): question, rank, chunk")
    id_width = max((len(question_id) for question_id, _ in rows), default=0)
    chunk_width = max((len(verdict.chunk_id) for _, verdict in rows), default=0)
    for question_id, verdict in rows:
        label = "kept, not evidence" if verdict.relevant else "evidence, left out"
        chunk = f"{verdict.chunk_id:{chunk_width}}"
        print(f"{question_id:{id_width}}  {verdict.rank:>2}  {chunk}  {label}")


def _ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _ids(question_ids: Sequence[str]) -> str:
    return ", ".join(question_ids) or "-"
