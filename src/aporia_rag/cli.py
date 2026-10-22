"""Command-line entry point: ``aporia-rag <command>``.

    aporia-rag fetch                 download the corpus into data/raw
    aporia-rag index [--rebuild]     embed the corpus into the vector index
    aporia-rag search QUESTION       show the passages retrieval finds for a question
    aporia-rag ask QUESTION          answer from the relevant passages, citing them
    aporia-rag evaluate [--model M]  measure retrieval on the gold questions [--dump PATH]
                                     [--grade: and the relevance grader]
                                     [--answers PATH: and answers, plain vs corrective RAG]

Each command parses its arguments, asks the `Container` for the components
it needs, and prints. Expected failures (Ollama not running, a model not
pulled, the corpus not downloaded, invalid settings) are printed as one
`error:` line with exit code 1; a traceback always means a bug.
"""

import argparse
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter

import httpx

from aporia_rag.bootstrap import Container
from aporia_rag.config import Settings, load_settings
from aporia_rag.domain.errors import AporiaError
from aporia_rag.domain.models import RetrievedChunk
from aporia_rag.domain.ports import WebSearch
from aporia_rag.evaluation.answers import Summary, System, evaluate_answers, summarize
from aporia_rag.evaluation.answers import write_answers as write_answer_records
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
from aporia_rag.indexing.indexer import WorkProgress
from aporia_rag.ingestion.corpus import load_manifest
from aporia_rag.ingestion.gutenberg import (
    DEFAULT_MIRROR,
    POLITE_DELAY_SECONDS,
    EbookNotFoundError,
    GutenbergMirror,
    fetch_cached,
)
from aporia_rag.ingestion.loader import iter_documents
from aporia_rag.pipelines.corrective import ENOUGH, OTHER_AUTHORS
from aporia_rag.pipelines.outcome import Outcome
from aporia_rag.retrieval.scope import DEFAULT_K, Scope, scope_for
from aporia_rag.retrieval.scope import search as search_in_scope

type _Command = Callable[[argparse.Namespace, Container], int]

_SNIPPET_LENGTH = 200


def main(
    argv: Sequence[str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    settings: Settings | None = None,
    web_search: WebSearch | None = None,
) -> int:
    """Run the CLI.

    `transport` replaces the network (the mirror and the model server),
    `settings` the environment and `web_search` the web search, which does
    not go through `transport`; all three exist for tests.
    """
    args = _parser().parse_args(argv)
    command: _Command = args.run
    try:
        container = Container(
            settings or load_settings(), transport=transport, web_search=web_search
        )
        with container:
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
    ask.add_argument(
        "--others",
        action="store_true",
        help="after the answer, also grade the best passage of up to "
        f"{OTHER_AUTHORS} other authors and list those who discuss the question (one model "
        "call each; only when the question names no author)",
    )
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
        action="append",
        type=Path,
        dest="golds",
        metavar="GOLD",
        help="gold question set TOML file; repeat to evaluate several as one "
        "(default: the packaged retrieval set)",
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
    evaluate.add_argument(
        "--answers",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"also answer every gold question with plain RAG (the top {DEFAULT_K} passages) "
        "and with the corrective graph, as `ask` would with the configured models, compare "
        "them, and write every answer to PATH (JSON Lines; many model calls)",
    )
    _add_manifest_option(evaluate)
    evaluate.set_defaults(run=_evaluate)
    return parser


def _add_question_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("question")
    command.add_argument(
        "-k",
        type=_positive_int,
        default=DEFAULT_K,
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
    scope = _scope(args, load_manifest(args.manifest).authors)
    container.require_index()
    _print_scope(scope)
    found = search_in_scope(container.retriever, args.question, scope, k=args.k)
    for rank, result in enumerate(found, start=1):
        chunk = result.chunk
        print(f"{rank}. {result.score:.3f}  {chunk.metadata.citation}  [{chunk.id}]")
        print(f"   {_snippet(chunk.text)}")
    return 0


def _scope(args: argparse.Namespace, authors: Sequence[str]) -> Scope:
    return scope_for(args.question, authors, author=args.author, all_authors=args.all_authors)


def _print_scope(scope: Scope) -> None:
    if scope.named:
        print(f"Searching only the authors named in the question: {', '.join(scope.named)}.\n")


def _snippet(text: str) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    if len(flat) <= _SNIPPET_LENGTH:
        return flat
    return flat[:_SNIPPET_LENGTH].rsplit(" ", 1)[0] + " ..."


# --------------------------------------------------------------------------
# ask
# --------------------------------------------------------------------------


# The stages of an answer, in the order the time report lists them.
_STAGES = (
    "rewriting",
    "retrieval",
    "grading",
    "web search",
    "generation",
    "revision",
    "other authors",
)
# How `ask` names each kind of problem the answer checks find (`grading.citations`).
_PROBLEMS = {
    "citation": "a citation with no passage",
    "quote": "a quotation not word for word",
    "author": "an author named for another's passage",
}


def _ask(args: argparse.Namespace, container: Container) -> int:
    authors = load_manifest(args.manifest).authors
    scope = _scope(args, authors)
    container.require_index()
    _print_scope(scope)
    started = perf_counter()
    outcome = container.corrective_rag.answer(
        args.question, scope, k=args.k, authors=authors, others=args.others
    )
    if outcome.author_questions:
        print("Asked of each author on their own:")
        for question in outcome.author_questions:
            print(f"- {question}")
        print()
    if outcome.query is not None:
        print("Nothing relevant was found, so the question was searched again as: ", end="")
        print(f'"{outcome.query}"\n')
    if outcome.web:
        print(f"{_why_the_web(outcome)}\n")
    if outcome.answer is not None:
        print(outcome.answer.text)
        print("\nWeb sources:" if outcome.web else "\nPassages:")
        for number, source in enumerate(outcome.answer.sources, start=1):
            print(f"[{number}] {_label(source)}")
        _print_checks(outcome)
    else:
        print(_nothing_answers(len(outcome.graded)))
    _print_passages("Left out as not relevant:", outcome.left_out)
    _print_passages(f"Not graded, as {ENOUGH} relevant passages came first:", outcome.skipped)
    _print_passages("Other authors who discuss this:", outcome.others)
    stages = [
        f"{stage} {outcome.seconds[stage]:.1f}s" for stage in _STAGES if stage in outcome.seconds
    ]
    print(f"\nanswered in {perf_counter() - started:.1f}s ({', '.join(stages)})")
    return 0


def _print_passages(heading: str, passages: Sequence[RetrievedChunk]) -> None:
    """`heading`, then one line per passage, after a blank line; nothing when there are none."""
    if passages:
        print(f"\n{heading}")
        for passage in passages:
            print(f"- {_label(passage)}")


def _print_checks(outcome: Outcome) -> None:
    if count := len(outcome.draft_problems):
        problems = "problem" if count == 1 else "problems"
        print(f"\nThe first answer had {count} {problems} with its passages, ", end="")
        print("so it was written again.")
    if outcome.problems:
        print("\nStill not borne out by its passages:")
        for problem in outcome.problems:
            detail = f'"{problem.detail}"' if problem.kind == "quote" else problem.detail
            print(f"- {_PROBLEMS[problem.kind]}: {detail}")


def _nothing_answers(graded: int) -> str:
    if graded == 1:
        return "The one passage found does not answer this question."
    return f"None of the {graded} passages found answers this question."


def _why_the_web(outcome: Outcome) -> str:
    if outcome.outside_author is not None:
        why = (
            "Nothing relevant was found, and the rewriter named no author in the corpus "
            f'("{outcome.outside_author}")'
        )
    else:
        why = "Nothing relevant was found again"
    if outcome.web_error is not None:
        return f"{why}, so the web was searched; the search failed ({outcome.web_error})."
    return f"{why}, so the web was searched."


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
    gold = _load_gold(args.golds)
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
    if args.answers is not None:
        container.require_index()
        authors = load_manifest(args.manifest).authors
        plain, corrective = container.plain_rag, container.corrective_rag
        systems: dict[str, System] = {
            "plain": lambda q: plain.answer(q, scope_for(q, authors), k=DEFAULT_K, authors=authors),
            "corrective": lambda q: corrective.answer(
                q, scope_for(q, authors), k=DEFAULT_K, authors=authors
            ),
        }
        answers = evaluate_answers(systems, gold)
        _print_answers({name: summarize(results) for name, results in answers.items()})
        write_answer_records(args.answers, answers)
        print(f"\nanswers written to {args.answers}")
    return 0


def _load_gold(paths: Sequence[Path] | None) -> GoldSet:
    """The gold sets at `paths` as one (their question ids must differ), or the packaged one."""
    if not paths:
        return load_gold_set()
    return GoldSet(questions=[q for path in paths for q in load_gold_set(path).questions])


def _evaluate_model(model: str, gold: GoldSet, container: Container) -> RetrievalReport:
    """Evaluate one embedding model on its own index, with otherwise the same settings."""
    settings = container.settings
    embedding = settings.embedding.model_copy(update={"model": model})
    model_settings = settings.model_copy(update={"embedding": embedding})
    with Container(model_settings, transport=container.transport) as model_container:
        model_container.require_index()
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


_ANSWER_LABEL = 36  # width of the row labels of the answer comparison


def _print_answers(summaries: dict[str, Summary]) -> None:
    """One column per system: what its answers were given, where they came from, their checks."""
    first = next(iter(summaries.values()))
    print(
        f"\nanswers to {first.questions} gold questions ({first.in_corpus} with evidence, "
        f"{first.outside} without), plain RAG from the top {DEFAULT_K} passages\n"
    )
    columns = [max(len(name), 5) + 2 for name in summaries]

    def row(label: str, values: Sequence[object]) -> None:
        cells = "".join(f"{value!s:>{width}}" for value, width in zip(values, columns, strict=True))
        print(f"{label:{_ANSWER_LABEL}}{cells}".rstrip())

    every = summaries.values()
    row("", list(summaries))
    row("answered", [summary.answered for summary in every])
    row(f"  given a gold passage (of {first.in_corpus})", [s.grounded for s in every])
    row(
        "gold share of the passages given",
        [_ratio(s.gold_passages / s.passages if s.passages else None) for s in every],
    )
    print(f"outside questions answered (of {first.outside})")
    row("  from the corpus", [summary.outside_from_corpus for summary in every])
    row("  from the web", [summary.outside_from_web for summary in every])
    row("  not answered", [summary.outside_unanswered for summary in every])
    row("answers with a check problem", [summary.flagged for summary in every])
    row("  written again", [summary.revised for summary in every])
    row("median seconds per question", [f"{summary.median_seconds:.1f}" for summary in every])
    print()
    for name, summary in summaries.items():
        problems = _counts(summary.problems) or "none"
        print(f"{name}: routes {_counts(summary.routes)}; problems {problems}")


def _counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name} {count}" for name, count in counts.items())


def _ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _ids(question_ids: Sequence[str]) -> str:
    return ", ".join(question_ids) or "-"
