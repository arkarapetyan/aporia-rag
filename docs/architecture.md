# Architecture

How AporiaRAG is built and why. The measurements behind the decisions, each with the rule fixed
before it was run, are in [experiments.md](experiments.md); the tests, component by component,
are in [test-plan.md](test-plan.md).

## A question's way through the system

```
question ──> scope ──> search ──> grade ──> write ──> check ──> answer + passages
              │                    │  ▲                 │  ▲
              │                    │  │                 └──┘ written once more if the checks fail
              │                    │  └── rewrite the question, search again (once)
              │                    │  └── or search the web (last resort)
              │                    └── nothing answers: say so, list what was graded
              └── the authors the question names, or the reader's --author / --all-authors
```

Two entry points share everything below them: the CLI (`aporia-rag ask`, `search`, `evaluate`...)
and the web UI (`aporia-rag serve`). Each turns its options into a `Scope`, asks the container for
a pipeline, and only renders what comes back.

## Layers and the dependency rule

```
src/aporia_rag/
├── domain/        Models, errors and ports (Protocols). Depends only on pydantic.
├── ingestion/     Gutenberg mirror client, per-edition structuring, chunking, the corpus manifest.
├── indexing/      Keeps the vector index in step with the corpus.
├── retrieval/     Search, scope rules, authors named in a question, the query rewriter.
├── generation/    The answer and the survey, written by the chat model from numbered passages.
├── grading/       The relevance grader, and the citation checks on an answer (plain code).
├── evaluation/    Gold sets and metrics: retrieval, the grader, and answers (plain vs corrective).
├── adapters/      Ollama, Chroma and ddgs behind the ports.
├── config.py      Settings from the environment or .env.
├── pipelines/     Plain RAG, and the corrective pipeline built with LangGraph.
├── bootstrap.py   Composition root: provider registries and the Container.
├── ui/            FastAPI backend and the static page.
└── cli.py         `aporia-rag fetch | index | search | ask | evaluate | serve`.
```

**Dependencies point inward,** and the rule is enforced, not documented: four
[import-linter](https://import-linter.readthedocs.io) contracts in `pyproject.toml` run as tests.

- *layers*: `cli` > `ui` > `bootstrap` > `pipelines` > {`adapters`, `config`, `evaluation`,
  `generation`, `grading`, `indexing`, `ingestion`, `retrieval`} > `domain`. Nothing imports a layer
  above it, and the middle packages do not import each other: the indexer reaches Chroma only
  through the `VectorStore` port, and the citation checks reach the corpus's authors through a
  function they are given.
- *provider-libraries*: `chromadb`, `ollama` and `ddgs` are imported by `adapters` only.
- *orchestration-framework*: `langgraph` is imported by `pipelines.corrective` only.
- *web-framework*: `fastapi`, `starlette` and `uvicorn` are imported by `ui` and the CLI that
  serves it.

A violation fails the build with the offending import chain. This is what keeps the project
testable: the pipelines run in unit tests against small in-memory fakes of the same ports, with no
mocks of library internals, and swapping the chat model or the vector store means writing one
adapter.

**Behaviour lives below the entry points.** Which authors a question covers (`retrieval.scope`),
when other authors are looked for (the corrective pipeline), what a step reports, how each kind of
check problem is called (`grading.citations`), whether the index is empty (the container): the CLI
and the UI only turn options into calls and results into text. They were written for the CLI first
and moved when the UI needed them (component 30).

## Providers: ports, adapters, registries

The application is written against five ports (`domain/ports.py`), all `typing.Protocol`s that
speak only in domain types:

| Port | What it promises | Implementations |
|---|---|---|
| `Chunker` | `chunk(document) -> list[Chunk]` | `RecursiveChunker` |
| `Embedder` | `space`, `embed_documents(passages)`, `embed_query(text)` | `OllamaEmbedder` (tests: `HashingEmbedder`) |
| `VectorStore` | `space`, `upsert`, `query(vector, k, filters)`, `fingerprints`, `delete`, `clear`, `count` | `ChromaVectorStore` (tests: `InMemoryVectorStore`) |
| `ChatModel` | `chat(messages, max_tokens) -> str`, `chat_structured(messages, schema, max_tokens)`, at temperature 0 | `OllamaChatModel` (tests: `ScriptedChatModel`) |
| `WebSearch` | `search(query, limit) -> list[Chunk]` | `DdgsWebSearch` (tests: `ScriptedWebSearch`) |

**A port arrives with its first caller,** never ahead of it. `chat_structured` came with the
relevance grader, which needs a verdict code can rely on: the reply must fit a pydantic schema,
Ollama constrains generation to the schema's JSON schema (the major APIs have an equivalent), and a
reply that does not fit is a `ProviderError`, never a guess.

**Queries and passages are different calls.** Retrieval embedders are trained asymmetrically and
each provider marks the difference its own way (an instruction for Qwen3, a prefix for
EmbeddingGemma and Nomic, a parameter for Gemini and Voyage AI). `Embedder` states the intent, and
the templates are Strategy objects per model family (`adapters/prompt_formats.py`).
`BatchingEmbedder` (Template Method) holds what every embedder shares: batching, the purpose, and
checking one vector per input, all of one size.

**An embedding space is a value.** `EmbeddingSpace(provider, model, dimensions, prompt_format)`
says what a vector means. The indexer and the retriever refuse an embedder and a store from
different spaces, and Chroma keeps one collection per space, so switching models builds a new index
beside the old one instead of mixing vectors, and models can be compared side by side.

**Composition root and container.** `bootstrap.py` maps provider names from the settings to
factories (a `ProviderRegistry` per port kind), and its `Container` builds each component once, on
first use, and closes what it built when it exits. Adding a provider is one adapter, one
registration and one setting (`APORIA_EMBEDDING__PROVIDER=google`); nothing in `indexing`,
`retrieval` or `pipelines` changes. Factories take an optional `httpx` transport, which is how the
tests run whole commands, and the web UI, with no network.

**One family of expected errors.** Every failure the application expects is an `AporiaError`:
configuration, corpus, provider (unreachable, model not pulled, a reply cut at its cap) and index
mismatches. Adapters translate provider exceptions, so the CLI prints any of them as one line, the
UI shows the same sentence, and a traceback always means a bug.

**Fakes held to the real contract.** The vector-store contract suite runs every test against both
the in-memory store and real Chroma. `FakeOllamaServer` refuses what Ollama refuses (an unanchored
regex in a reply schema: the rewriter's first schemas passed every fake test and failed every live
call), cuts replies at their cap and reports why it stopped.

## The corpus and the index

**From a Gutenberg mirror, cached verbatim.** Project Gutenberg forbids automated access to its
website, so `GutenbergMirror` talks to a mirror with a 2-second wait. Downloads are cached in
`data/raw/`, and cleaning runs on the cached copy.

**Authors' words only.** Every edition wraps the author's text in someone else's: Jowett's
introduction is 46% of the *Republic* file, and the Loeb *De Officiis* is half Latin. Each work's
`layout` in `ingestion/corpus.toml` says where the text starts and ends and how sections are headed;
shared rules remove footnotes and apparatus; Latin paragraphs are dropped by a vote of function
words. The result is 245 sections, each cited by author, work and section.

**Chunks are exact spans.** A chunk's text is always `document.text[start_char:end_char]`
(validated, and property-tested), so a citation points at exact words. Chunking is recursive and
structure-aware (paragraph, sentence, clause, word), 1,000 characters with 150 of overlap that
starts at a sentence boundary: 7,705 chunks.

**Embedded with context, stored as source.** A chunk is embedded titled with its citation ("Immanuel
Kant, Fundamental Principles..., First Section"), and the store keeps the exact text. Indexing is
incremental: a fingerprint of what was embedded decides what is new, changed or gone. Similarity is
cosine. Ollama is told never to truncate, so an over-long chunk is an error, not half a vector.
Queries are embedded on the CPU, so on an 8 GB GPU the chat model keeps the GPU instead of being
reloaded for every question.

## Retrieval and scope

**Three scopes** (`retrieval.scope`): the k best passages of any author; the k best of one author;
or a survey, the best passage of each of several authors. `scope_for(question, authors, author=...,
all_authors=...)` decides: the reader's `--author` (any part of a name) or `--all-authors`, or else
the authors the question names by surname, matched as whole words in plain code. One named author
filters the search; two or more make a comparison, which the pipeline asks of each author on their
own. `Retriever.best_per_author` embeds the question once and searches once per author, so an
author with few passages on a subject still gets a turn (Aristotle has 63 chunks on friendship,
Cicero 6).

## The corrective pipeline

`pipelines.CorrectiveRAG` is a LangGraph state machine:

```
START -> [ask_each_author] -> retrieve -> grade -> generate -> check -> [other_authors] -> END
                                                                 |   ^
                                   (problems, the first time)    +-> revise

after grade, when nothing is kept:
  not rewritten yet -> rewrite -> retrieve     the author it names is in the corpus
                               -> web_search   ...or is not
  rewritten         -> web_search -> grade
  web searched      -> END, with no answer
```

**Model calls only where they are needed.** On the normal path the calls are one verdict per passage,
in the order found, until 2 are kept (early stop), and one answer. The loop's calls (one rewrite,
verdicts on new passages, web pages) run only when nothing is kept, and the second draft only when
the checks find problems. Every loop runs at most once, so the pipeline always ends, with "none of
the passages found answers this question" if it has to.

| Path | Model calls |
|---|---|
| Normal: something relevant on the first search | 1 verdict per passage until 2 are kept, 1 answer |
| Loop: nothing relevant, an author in the corpus named | + 1 rewrite, 1 verdict per new passage |
| Web: nothing relevant, or an author outside the corpus | + 1 rewrite, 1 verdict per page |
| Problems in the answer | + 1 second draft |
| Comparison of named authors | + 1 call for the subject, 1 verdict per author |
| `--others` | + 1 verdict for each of up to 4 other authors |

**The rewrite names an author first.** The rewriter's structured reply is `{author, query}`, in that
order: the model chooses whose words to search in, then writes the query in them ("why men
surrender natural liberty to a commonwealth"). Plain code checks the name against the corpus: an
author outside it sends the question to the web at once, with no second corpus search. A reply
schema steers a model as its prompt does, so the field order was measured like a prompt.

**The web, last.** The web gets the reader's question, for k pages; each page becomes a chunk with
its site as author (`domain.web_passage`, at most 2,000 characters), so the grader and the generator
take it like a passage, and only then does a question leave the machine. A survey never goes to the
web, and a failed web search ends without an answer and says why.

**Comparisons are asked per author.** Asked "How do Kant and Mill differ on lying?", the grader left
out every passage, even Kant's and Mill's own on lying; asked about one author at a time, it kept
both. So one call writes the comparison's subject, plain code asks it of each author, and each
author's best passage is found and graded with that author's own question.

**Other authors, on request.** `others=True` grades the best passage of up to 4 authors who had none
among those graded, and lists the relevant ones: only beside an answer from the corpus to an
unrestricted question.

**Steps as they finish.** `answer(..., on_step=...)` streams LangGraph's node updates and reports
each node as it finishes: its name, one sentence ("Graded 5 passages: 2 relevant, 3 not graded",
"Rewrote the question as ...") and its seconds. The sentences are written once, in Python, for
any client.

**`Outcome`** is what both pipelines return: the answer and its sources, what was graded, left out
and skipped, the route ("plain", "normal", "loop", "web"), the rewritten query, the problems of the
first and the final draft, other authors, and the seconds per stage.

**`pipelines.PlainRAG`** is the baseline: the same scope, retrieval, answer prompt and checks, with no
grader, no second search, no web and no second draft.

**Why LangGraph.** The loop is a small state machine, and LangGraph states it as nodes and
conditional edges: the web fallback was one node and three edges, and node updates stream for free.
Only `pipelines.corrective` imports it, and its nodes call the same components as everything else,
so they are tested with the same fakes. LangChain's model and retrieval wrappers are not used.

## Generation

**Numbered passages, numbered citations.** The model gets the passages numbered from 1, each headed
by its citation, then the numbers it may cite ("There are 2 passages: cite them as [1] and [2]."),
then the question. The prompt asks it to use only passages that bear on the question, to say when
none does, to give an author only what the author holds (philosophers state views to reject them),
to cite the passage behind each claim, and to quote only words that are in the passages.
`Answer.sources[n - 1]` is the passage cited as `[n]`. The prompt once gave "[2]" as an example of a
citation; 4 of the 5 answers that made up a citation number cited [2] from one passage, so the
numbers are now named instead.

**A survey's form is code.** `Generator.survey` asks for a structured reply with exactly one
paragraph per passage (the count is in the JSON schema), and each paragraph becomes a section
headed by its author and passage number. A prompt rule is a tendency; what must always hold is
structure.

**Settings every call shares.** Thinking is off (`think: false`): Qwen3.5 reasons at length by
default. The context length is fixed at 8,192 tokens for every call, since Ollama reloads a model
whose context changes and silently drops what does not fit. Every call states a reply cap (a verdict
16 tokens, a rewrite 64, an answer 1,024, a survey 256 per passage); a reply cut at its cap is a
`ProviderError`, so a reply that does not stop fails in seconds, not at the 120 s time-out.

## Grading and checking

**Nearest is not relevant.** Retrieval returns the passages nearest to the question, and every
passage the model got used to get a paragraph, often misread. `grading.RelevanceGrader` asks the
chat model about each passage, one structured call each (`{"relevant": true}`), so each verdict
depends on its passage alone and each prompt stays short. What it leaves out is listed, so every
decision can be checked.

**The right passages are not yet a right answer.** The citation checks (`grading.citations`) are plain
code, run on every answer but a survey:

- every quotation must occur word for word in a passage its sentence cites (or in any passage when
  the sentence cites none); case, spacing, quotation marks, apostrophes and Gutenberg's `_italics_`
  do not count, and an ellipsis or an editor's "[s]" marks a gap;
- every citation must have a passage behind it, ranges ("[5-6]") included;
- a sentence that cites passages and names corpus authors must name the author of one of them.

Each problem carries a note for the model and a label for the reader. With problems, the pipeline
writes the answer once more: the model gets its own draft back (the one place the application sends
an `assistant` message) and the notes; what still fails is shown with the answer, never hidden.

**What is not checked.** A claim its passage does not make, a view an author reports and then rejects
given as the author's own, a speaker confused inside a dialogue. A model check of every sentence
(groundedness) was built as a spike and dropped: the local model found 1 of 6 such errors read by
hand. These errors are measured by reading, in the evaluation.

## Evaluation

`aporia-rag evaluate` measures through protocols declared next to it (`Searcher`, `Judge`, `Run`),
which the retriever, the grader and the pipelines' `Outcome` satisfy, so `evaluation` imports none
of them.

**Retrieval on a gold set.** `evaluation/gold_retrieval.toml` holds 40 questions over all 12 works,
written in a reader's words rather than the source's, each with every passage known to answer it, as
a section and a short verbatim quote. Matching the quote rather than a chunk id keeps the set valid
when chunking changes. Metrics: Hit@1/5/10 and MRR@10, for the passage (the chunk holds the quote) and
the section. An answering passage missing from the set counts as a miss, so the set was completed by
pooling: every chunk the bake-off models retrieved was read and judged, and `--dump` saves what a
run retrieves for the next round.

**The grader against the gold set** (`--grade`): precision and recall of its verdicts on the top 5
and the top 10, the questions where the loop would start (rightly, needlessly, or not when it
should), and every disagreement, to be read. `evaluation/gold_loop.toml` adds 9 questions for the
loop: 5 answered in the corpus but asked in words far from the source's, and 4 the corpus cannot
answer.

**Answers, plain vs corrective** (`--answers PATH`): both pipelines answer all 49 questions with the
same models. Plain code counts the answers given a gold passage, the gold share of the passages
given, where the answers to outside questions came from, the check problems, the second drafts and
the median time; every answer is saved with its passages for reading. A blind reading of 15
questions' two answers (shuffled, without the systems' names) counts the errors plain code cannot
see.

## The web UI

`aporia-rag serve` runs a FastAPI app (`ui/app.py`) on 127.0.0.1:8000 with one static page
(`ui/static`: HTML, CSS and plain JavaScript, no build step).

- `POST /api/ask` takes `ask`'s options and answers with JSON lines: one per step as it finishes,
  then the outcome, or an error. The pipeline runs in a thread and hands each step to the response
  through a queue, so a step reaches the page when it finishes. A request that cannot be answered
  (an unknown author, an empty index) is refused with 400 before anything runs. One question runs
  at a time: on a laptop the model is the bottleneck.
- The page reads like a conversation: the question box (with every `ask` option: authors,
  number of passages, other authors) stays at the bottom, and each question gets a reply right
  under it. But each question is answered on its own: the page sends no earlier question or
  answer, so a follow-up such as "And Mill?" is searched as it is written.
- The reply draws the route as a line, the main stations (search, grade, write, check) on it and
  the detours (rewrite, search again, web, write again, other authors) below it, lit as the steps
  arrive; then the answer, the checks and the route taken, with the steps and their times folded
  away.
- The passages open in a panel beside the conversation (inside the reply on a narrow screen),
  from a reply's button or from a citation, which brings its passage forward. The panel stays
  closed until then, and follows the newest answer while open.
- The page scrolls as a whole, with the bar, the question box and the panel held in place, so a
  browser's full-size screenshot holds the whole conversation.
- Every change of state moves: the panel and the folded lists slide, the route draws on from
  station to station. With the system's "reduce motion" setting, everything changes at once.

## Quality gate

`ruff` (lint and format), `mypy --strict` over the source and the tests, and pytest with warnings
as errors and branch coverage of at least 90% (100% in practice): one `make check`, run locally, in
the pre-commit hook and in CI. Tests that need the real models (`make test-llm`) or the real corpus
and network (`make test-integration`) are opt-in and run by hand before a commit.

## Known limitations

- **Errors only a reader finds.** In the blind reading, 3 of 15 corrective answers still had one: a
  view an author reports and rejects given as his own (a chunk that starts mid-argument does not
  say who is speaking), a sentence of Cicero's misread, and a garbled self-correction added by the
  second draft.
- **The second draft can make an answer worse** while fixing its quotations; it fixed most problems
  in one run and half in another.
- **Not fully repeatable.** Temperature 0 makes a single call repeatable, but not a whole run on
  Ollama: the same code flagged 6 first drafts in one run and 12 in the next.
- **Comparisons can lose one side** when that author's best passage on the subject is elsewhere in
  the corpus (Mill on lying retrieves *On Liberty*, not *Utilitarianism*).
- **Questions about the details of a work outside the corpus** (Plato's *Lysis*) can end with
  nothing kept: the grader leaves out pages that say the dialogue reaches no conclusion.
- **No follow-up questions.** Each question is answered on its own; the page looks like a
  conversation, but a question that leans on the one before it ("Why does he say that?") is
  searched as written.
- **The model needs 22 GB of RAM;** `qwen3.5:9b` is the setting for 16 GB machines, and the
  measurements here are for the 35B model only.
