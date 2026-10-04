# Architecture

## Layers and the dependency rule

```
src/aporia_rag/
├── domain/        Pure core: data models now, ports (Protocols) later. Depends only on pydantic.
├── ingestion/     Raw text -> Documents -> Chunks. Depends on domain.
├── adapters/      (planned) Ollama, Chroma, ddgs implementations of the domain ports.
├── graph/         (planned) LangGraph state, nodes and wiring. Depends on domain ports only.
└── evaluation/    (planned) Corrective vs. baseline comparison on a curated question set.
```

**Rule:** dependencies point inward. `domain` imports nothing from the other packages; `graph`
talks to the LLM, vector store and web search only through `Protocol`s defined in `domain`,
never through Ollama/Chroma/ddgs types directly.

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
sentence, clause, word, then characters. Overlap repeats whole units only (never half a
sentence) and shrinks when needed to guarantee progress. Sizes are in characters, a proxy for
tokens, which is good enough for embedding models with large context windows; this can be
revisited if a small-context embedder is chosen.

**Local model: `qwen3:14b` via Ollama.** Fits a laptop and handles the structured grading prompts.
A larger local model is the fallback if grading quality turns out to be insufficient, which the
evaluation phase will measure rather than guess.

**Web fallback: `ddgs`.** Free with no API key. (The older `duckduckgo-search` package was renamed
to `ddgs` upstream and is no longer maintained under its old name.)

**Quality gate.** `ruff` (lint + format), `mypy --strict` over source *and* tests, pytest with
warnings treated as errors and branch coverage at 90% or above. The same `make check` runs
locally, in the pre-commit hook and in CI, so they can never disagree.
