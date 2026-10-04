# AporiaRAG: Corrective RAG over Primary Philosophical Texts

A question-answering agent that answers from the primary texts of Western philosophy, and
**checks its own work** before answering: it grades what it retrieved, rewrites the query or
falls back to a web search when the corpus does not cover the question, and verifies that the
final answer is grounded in its sources. Everything runs locally on open models; nothing costs money.

> **Status:** early development. See [Roadmap](#roadmap).

## Why "corrective"?

Plain RAG is open-loop: embed the question, fetch the top-k passages, generate. When retrieval
misses, the model either hallucinates or gives up, and nothing notices.

Philosophy makes this failure common and subtle. Kant and Mill both write at length about
*duty*; Plato and Aristotle both write about *the good*, and mean different things by it. Cosine
similarity cannot tell which one the user is asking about, so retrieval regularly returns passages
that are on-topic but wrong. This project closes the loop with explicit grading and correction
steps, based on the Corrective RAG ([Yan et al., 2024](https://arxiv.org/abs/2401.15884)) and
Self-RAG ([Asai et al., 2023](https://arxiv.org/abs/2310.11511)) patterns.

## How it works

```mermaid
flowchart LR
    Q([Question]) --> R[Retrieve from corpus]
    R --> GD{Grade each passage<br/>for relevance}
    GD -- enough relevant --> GEN[Generate answer<br/>with citations]
    GD -- not enough --> RW[Rewrite query]
    RW --> WS[Web search fallback]
    WS --> GEN
    GEN --> V{Grounded in sources?<br/>Answers the question?}
    V -- yes --> A([Answer + citations])
    V -- not grounded --> GEN
    V -- off target --> RW
```

Every loop has a hard retry cap, so the agent always terminates, with an honest "I don't know"
if it has to.

## Tech stack

All free, all local by default.

| Concern | Choice |
|---|---|
| LLM | [Ollama](https://ollama.com) running `qwen3:14b` |
| Orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) state machine |
| Vector store | [Chroma](https://www.trychroma.com) (embedded, on disk) |
| Web fallback | [`ddgs`](https://pypi.org/project/ddgs/) (no API key) |
| Corpus | [Project Gutenberg](https://www.gutenberg.org) public-domain texts |
| Tooling | `uv`, `pytest` + Hypothesis, `ruff`, `mypy --strict`, GitHub Actions |

## Getting started

Prerequisites: [uv](https://docs.astral.sh/uv/) and, for the parts that call a model,
[Ollama](https://ollama.com) with `ollama pull qwen3:14b`.

```bash
make install   # create .venv and install from the lockfile
make check     # lint, strict type-check, tests with coverage: what CI runs
make help      # list all targets
```

## Development

The project is built test-first. For every component: design and list the tests, write them,
write the component, then run the full quality gate. Test lists live in
[`docs/test-plan.md`](docs/test-plan.md) and design decisions in
[`docs/architecture.md`](docs/architecture.md).

| Command | Runs |
|---|---|
| `make test` | Fast suite: unit + property-based tests. No network, no LLM. |
| `make test-integration` | Tests against real local infrastructure. |
| `make test-llm` | Tests against the real local model. Slow, run by hand. |

## Roadmap

- [x] Project scaffold, tooling, CI
- [x] Domain models
- [x] Structure-aware recursive chunker
- [ ] Gutenberg loader: fetch, strip boilerplate, split works into sections
- [ ] Embeddings + Chroma vector store
- [ ] Baseline RAG (retrieve then generate), end to end
- [ ] Relevance grader + query rewriter
- [ ] Web search fallback
- [ ] Groundedness and answer-relevance graders
- [ ] Full corrective graph with retry caps
- [ ] Evaluation: corrective vs. baseline RAG on a curated question set
- [ ] Demo UI
