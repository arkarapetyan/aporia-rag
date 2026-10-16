# Architecture

## Layers and the dependency rule

```
src/aporia_rag/
├── domain/        Pure core: models, errors, ports (Protocols). Depends only on pydantic.
├── ingestion/     Raw text -> Documents -> Chunks. Depends on domain.
├── indexing/      Keeps the vector index in step with the corpus. Depends on domain ports.
├── retrieval/     Question -> nearest chunks. Depends on domain ports.
├── adapters/      Ollama, Chroma (later ddgs, Google...) implementations of the ports.
├── graph/         (planned) LangGraph state, nodes and wiring. Depends on domain ports only.
├── evaluation/    Gold questions and metrics: retrieval now (Hit@k, MRR), answers later.
├── config.py      Settings from the environment / .env.
├── bootstrap.py   Composition root: provider registries and the Container that builds everything.
└── cli.py         `aporia-rag fetch | index | search`.
```

**Rule:** dependencies point inward. `domain` imports nothing from the other packages;
`indexing`, `retrieval` and `graph` talk to models and stores only through the `Protocol`s in
`domain/ports.py`, never through Ollama/Chroma/ddgs types directly. Only `adapters/` imports
provider libraries, and only `bootstrap.py` knows which adapter is in use.

So `adapters/` importing `chromadb` and `ollama` is the design, not a leak: adapters are the one
place where a provider's API meets our interfaces. What must hold is that nothing *else* does.

**Enforced, not just documented.** The rules are [import-linter](https://import-linter.readthedocs.io)
contracts in `pyproject.toml`, checked by `tests/unit/test_architecture.py` on every test run:

- *layers*: `cli` > `bootstrap` > {`adapters`, `config`, `evaluation`, `indexing`, `ingestion`,
  `retrieval`} > `domain`. Nothing imports a layer above it, and the middle packages do not import each other:
  the indexer reaches Chroma only through the `VectorStore` port. `graph` gets its place in this
  contract when the package is created (it will sit above `retrieval`, which it calls).
- *provider-libraries*: `chromadb` and `ollama` are imported by `adapters` only, directly or
  indirectly.

A violation fails the build with the offending import chain, e.g.
`aporia_rag.indexing.indexer -> aporia_rag.adapters.chroma -> chromadb`.

This is the main thing that makes the project testable. Graph nodes are unit-tested against
small in-memory fakes that implement the same Protocols, instead of patching library internals
with mocks. The tests stay fast and deterministic, and swapping `qwen3:14b` for another model, or
Chroma for another store, means writing one adapter, not touching the graph.

Packages are created when the first component that needs them is built, not ahead of time.

## Key decisions

**`uv` with a `src/` layout.** One tool for Python versions, virtualenvs, locking and running.
The `src/` layout means tests import the installed package, not whatever happens to be in the
working directory.

**Runtime dependencies are added per component.** `pyproject.toml` lists only what is imported
today, so every lockfile change is traceable to the component that introduced it.

**Frozen, strict domain models.** `Document`, `Chunk` and `SourceMetadata` are immutable pydantic
models with `extra="forbid"`: no stage can mutate data another stage holds, and a misspelled
field fails immediately instead of disappearing.

**Chunks are exact spans of their source.** A chunk stores `start_char`/`end_char` and its text is
always `document.text[start_char:end_char]` (enforced by a model validator and a property-based
test). Citations can point to the exact passage.

**Recursive, structure-aware chunking.** Split at the coarsest boundary that fits: paragraph,
sentence, clause, word, then characters. Overlap repeats the longest tail of the previous chunk
that starts at a sentence boundary (falling back to a clause, then, for unpunctuated text, a
unit boundary), looking inside paragraph-sized units rather than only repeating whole units. The
first version only repeated whole units, which in practice meant almost no overlap: paragraphs are
nearly always longer than the budget (1.5% of boundaries overlapped; 89% after the change).
Overlap shrinks when needed to guarantee progress. Sizes are in characters, a proxy for
tokens, which is good enough for embedding models with large context windows; this can be
revisited if a small-context embedder is chosen.

**Local model: `qwen3:14b` via Ollama.** Fits a laptop and handles the structured grading prompts.
A larger local model is the fallback if grading quality turns out to be insufficient, which the
evaluation phase will measure rather than guess.

**Corpus from a Project Gutenberg mirror, never the website.** Gutenberg
[forbids automated access to gutenberg.org](https://www.gutenberg.org/policy/robot_access.html) and
asks scripts to use a mirror with a 2-second wait between requests. `GutenbergMirror` talks to a
mirror only and enforces the wait itself. Downloads are cached verbatim in `data/raw/` (written
atomically, so an interrupted run never leaves a truncated file), and cleaning runs on the cached
copy: changing the cleaning never means downloading again. The list of works is data
(`ingestion/corpus.toml`), validated on load.

**Per-edition layouts, shared clean-up rules.** Every Gutenberg file wraps the author's text in
someone else's: introductions, notes, indexes, a whole Latin original in the Loeb *De Officiis*.
What differs per edition (where the author's text starts and ends, how sections are headed) is
data in each work's `layout`; what recurs (footnote and apparatus conventions) is one set of
shared rules. The layouts were found with a throwaway spike over the real files and then built
test-first: unit tests use tiny synthetic editions, one convention each, and opt-in contract tests
pin the real corpus (section counts per work, Jowett excluded, no Latin left). An end-to-end scan
of the loaded corpus then caught 71 margin notes merged into Cicero's paragraphs that no test had
anticipated; that gap is now covered by both kinds of test.

**Web fallback: `ddgs`.** Free with no API key. (The older `duckduckgo-search` package was renamed
to `ddgs` upstream and is no longer maintained under its old name.)

**Latin detection by function words, smoothed by neighbours.** A paragraph is English or Latin by
majority vote of common function words (accents stripped, since the Loeb text marks verse
stress). A tie, typically a short verse line with no function words at all, takes the language
of the previous paragraph, because the bilingual edition alternates in page-sized blocks. On
*De Officiis* this keeps 522 English paragraphs and drops 511 Latin ones, with no misclassified
paragraph found. A language-identification library would add a dependency for one book.

## Provider independence: ports, adapters, registries

The application is written against three ports (`domain/ports.py`):

| Port | What it promises | Implementations |
|---|---|---|
| `Chunker` | `chunk(document) -> list[Chunk]` | `RecursiveChunker` |
| `Embedder` | `space`, `embed_documents(passages)`, `embed_query(text)` | `OllamaEmbedder` (tests: `HashingEmbedder`) |
| `VectorStore` | `space`, `upsert`, `query(vector, k, filters)`, `fingerprints`, `delete`, `clear`, `count` | `ChromaVectorStore` (tests: `InMemoryVectorStore`) |

They are `typing.Protocol`s: any class with these methods qualifies, without inheriting from
anything. The ports speak only in domain types (`Passage`, `Chunk`, plain `list[float]`
vectors), never in a provider's types, and adapters translate provider exceptions
into the domain's (`ProviderUnavailableError`, `ModelNotAvailableError`...), so nothing outside
`adapters/` can tell which provider is in use.

A port is added together with its first caller and its first adapter, never ahead of them: the
`ChatModel` port (plain and structured completion) arrives with the generation step, designed
test-first against what the generator and the graders actually need.

**Queries and passages are different calls.** Retrieval embedding models are trained
asymmetrically, and each provider marks the difference its own way: an instruction (Qwen3), a
prefix (Nomic, EmbeddingGemma), a parameter (`task_type` in Gemini, `input_type` in Voyage AI). So
`Embedder` states the intent and each adapter maps it. The text templates belong to the *model
family*, not to the server running it, so they are separate Strategy objects
(`adapters/prompt_formats.py`), chosen from the model name, which any adapter can reuse.

**The shared part of every embedder is written once.** `BatchingEmbedder` (Template Method)
batches, sends the right purpose and checks the answer: one vector per input, all of one size,
across calls too. A provider adapter implements one method, `_embed_batch`.

**An embedding space is a value.** `EmbeddingSpace(provider, model, dimensions, prompt_format)`
says what a vector means. Vectors from different spaces are never compared: the indexer and the
retriever refuse an embedder and a store from different spaces (`IndexMismatchError`), and Chroma
keeps one collection per space (name derived from the space, space recorded in its metadata and
checked on open). Switching the embedding model therefore builds a new index alongside the old one
instead of silently mixing vectors, which also lets the evaluation compare models side by side.

**Composition root, container and registries.** `bootstrap.py` maps provider names from the
settings to factories through a small `ProviderRegistry` per port kind, and its `Container` builds
each component once, on first use (`container.embedder`, `.vector_store`, `.indexer`,
`.retriever`). Entry points ask the container for what they need instead of wiring components
themselves, so the CLI and a future UI cannot wire them differently, and `fetch` never builds a
model client. The container also *owns* what it builds: it is a context manager, and on exit it
closes every component holding resources (the Ollama client's HTTP connection pool, a future SDK
client). `close()` is deliberately not part of the ports: the indexer and retriever only borrow
components, and only the owner should end their life. (This was found by `filterwarnings = error`:
the live tests failed on an unclosed socket that a CLI run silently drops at exit, but that a
long-running UI process would accumulate.)

Adding a provider is open/closed:

1. Write `adapters/<provider>.py`. For embeddings, extend `BatchingEmbedder` and implement
   `_embed_batch`, reusing a prompt format if the model family has one.
2. Register a factory in `bootstrap.py`: `EMBEDDERS.register("google", _google_embedder)`.
3. Add its contract tests: a new `VectorStore` gets the whole contract suite by adding one
   fixture parameter in `tests/unit/adapters/test_vector_store_contract.py`.
4. Configure it: `APORIA_EMBEDDING__PROVIDER=google`.

Nothing in `indexing/`, `retrieval/` or (later) `graph/` changes. Settings `provider` fields are
plain strings for the same reason: an enum of today's providers would have to be edited for each
new one. Factories receive an optional `httpx` transport, which every HTTP-based SDK can take; it
is how the CLI tests run whole commands with no network.

**One family of expected errors.** Every failure the application expects is an `AporiaError`
(`domain/errors.py`): configuration, corpus (`CorpusError`: a download missing, a file that is not
the expected edition), provider (unreachable, model not pulled) and index mismatches. Adapters
translate provider exceptions into these, so the CLI turns any of them, plus the environment's own
`OSError`s, into a one-line message; a traceback always means a bug.

**Fakes held to the same contract.** The in-memory fakes in `tests/support/fakes.py` are what the
rest of the suite runs on, so they must behave like the real thing: the vector-store contract
suite runs every test against both `InMemoryVectorStore` and real Chroma. `HashingEmbedder`
(feature hashing: texts sharing words get similar vectors) makes retrieval tests meaningful
without a model.

## Indexing

**Incremental and idempotent.** Each stored chunk carries a fingerprint (a hash of exactly what
was embedded: citation title and text). A run embeds only chunks that are new or changed, upserts
them in groups, and deletes stored chunks that no longer exist. A second run embeds nothing; an
interrupted run resumes where it stopped. Upsert, not add: Chroma's `add` silently ignores ids
that already exist, which would keep stale text forever.

**Embed with context, store the source.** A chunk is embedded as a passage titled with its
citation ("Immanuel Kant, Fundamental Principles..., First Section"), a contextual header that
tells the model what a mid-book paragraph is about. The store keeps only the exact source text,
so citations and offsets stay exact.

**Cosine, not L2.** Chroma's default distance is squared L2; collections are created with
cosine, and the store reports similarity (1 - distance), so every store returns the same scores.

**Fail loudly on over-long input.** Ollama silently truncates input longer than the model's
context unless told not to; the adapter sends `truncate: false`, so an over-long chunk is an error
instead of a vector of half its text.

## Evaluation

**Judge retrieval on a gold set, not on single queries.** `evaluation/gold_retrieval.toml` holds
40 hand-written questions over all 12 works, each with its evidence: every passage known to answer
it, as a section and a short verbatim quote. `aporia-rag evaluate --model A --model B` searches every
question in each model's own index and reports Hit@1/5/10 and MRR@10 at two levels: a *passage
hit* (the chunk contains the quote: what the generator needs) and a *section hit* (right section,
maybe the wrong paragraph). The gap between the two levels separates chunk dilution from not
understanding the question.

**Quotes, not chunk ids.** Matching the quote keeps the gold set valid when chunking changes, so
the same questions can later compare chunk sizes. An integration test checks that every quote
sits inside a chunk of its section, since an unfindable quote would silently lower every score.

**Every answering passage, not just the best one.** An answering passage missing from the gold
set counts as a miss when it is retrieved, so passage scores are only a lower bound until the
evidence is complete (in information retrieval: *incomplete relevance judgments*). The first
bake-off showed it: the right section in the top 5 for 88% of questions, the quoted passage for
40%. Reading the corpus for every other passage that answers each question found 39 more (84 in
all), including two errors: Hume poses the problem of induction in Section IV, and one Descartes
quote sat in a fragment that does not say what it is about. Passages a model finds in words no
search anticipated are found by reading its retrieved chunks (*pooling*, as in TREC):
`evaluate --dump data/eval/retrievals.jsonl` saves every retrieved chunk with its verdicts.

**Questions in a reader's words.** The questions avoid the source's wording, so they measure
meaning rather than word overlap: a bag-of-words baseline (feature hashing, the test embedder)
finds the passage in its top 5 for only 15% of them. With 40 questions, one question is 2.5
points, so smaller differences between models are noise.

**The evaluation depends on a `Searcher` protocol** (`search(question, *, k)`) declared next to
it. `Retriever` satisfies it structurally, so `evaluation` imports neither `retrieval` nor a
provider, and the corrective graph will be measured through the same interface.

## Quality gate

**One gate everywhere.** `ruff` (lint + format), `mypy --strict` over source *and* tests, pytest with
warnings treated as errors and branch coverage at 90% or above. The same `make check` runs
locally, in the pre-commit hook and in CI, so they can never disagree.
