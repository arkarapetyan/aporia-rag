# Test plan

## Workflow

Every component is built in four steps, in this order:

1. **Design and list the tests.** Write down what the component must do as a list of
   test cases in this file, before any test or production code exists.
2. **Write the tests.** Turn the list into pytest code.
3. **Write the component.** Implement until the tests pass, and no more.
4. **Run the tests.** Run the full quality gate (`make check`), not just the new tests.

A test is only useful if it can fail. Each boundary test below uses inputs where a naive
implementation would give a *different* answer than the intended one, so a passing test is
evidence that the behaviour exists rather than a coincidence.

## Test layers

| Layer | Marker | Runs by default | What it covers |
|---|---|---|---|
| Unit | none | yes | One component, no I/O. LLM, embedder, vector store and web search are replaced by in-memory fakes that implement the same `Protocol`. |
| Property-based | none | yes | Hypothesis generates thousands of inputs to check invariants that must hold for *any* input (e.g. "no text is lost"). |
| Integration | `integration` | no (`make test-integration`) | Real local infrastructure: on-disk Chroma, real embeddings, network. |
| LLM | `llm` | no (`make test-llm`) | Real `qwen3:14b` via Ollama. Slow and non-deterministic, so they assert on *structure* (valid JSON, a grade from the allowed set), not exact wording. |

---

## Component 1: Domain models (`aporia_rag.domain.models`)

`SourceMetadata` (provenance), `Document` (a whole source text) and `Chunk` (a retrievable span of
a document). Only *our* constraints are tested here, not pydantic itself.

| # | Test | Why it matters |
|---|---|---|
| 1.1 | `SourceMetadata.section` is optional and defaults to `None` | Not every work has sections. |
| 1.2 | `SourceMetadata` rejects unknown fields | A typo like `athor=` must fail loudly, not silently vanish. |
| 1.3 | `SourceMetadata` rejects a blank author or title | Every citation needs both. |
| 1.4 | `Document` rejects an empty id | Chunk ids are derived from it. |
| 1.5 | `Document` is immutable | Shared across pipeline stages; nobody may mutate it in place. |
| 1.6 | `Chunk` accepts a span whose length equals its text length | Happy path. |
| 1.7 | `Chunk` rejects a span whose length disagrees with its text | Offsets are what citations point to, so they must be right. |
| 1.8 | `Chunk` rejects empty text | An empty chunk would be embedded and retrieved as noise. |
| 1.9 | `Chunk` rejects negative `index` or `start_char` | Impossible positions. |

## Component 2: Recursive chunker (`aporia_rag.ingestion.chunker`)

Splits a `Document` into `Chunk`s no longer than `chunk_size` characters, preferring the
largest natural boundary that fits: **paragraph, then sentence, then clause (`;` `:`), then
word, then raw characters** as a last resort. Chunks are exact spans of the original text.
Consecutive chunks repeat up to `chunk_overlap` characters of whole trailing units.

**Configuration**

| # | Test |
|---|---|
| 2.1 | Rejects `chunk_size <= 0` |
| 2.2 | Rejects `chunk_overlap < 0` |
| 2.3 | Rejects `chunk_overlap >= chunk_size` |

**Degenerate and short input**

| # | Test |
|---|---|
| 2.4 | Empty or whitespace-only text produces no chunks |
| 2.5 | Text that fits produces exactly one chunk equal to the text |
| 2.6 | Leading/trailing whitespace is trimmed, offsets still point into the original text |
| 2.7 | Several paragraphs that fit together stay in one chunk, newlines preserved |

**Boundary preference** (sizes chosen so a naive word-packer would cut elsewhere)

| # | Input | `chunk_size` | Expected chunks |
|---|---|---|---|
| 2.8 | 3 paragraphs (30, 27, 16 chars) | 70 | `[p1 + p2, p3]`, not `p1 + p2 + "Glaucon"` |
| 2.9 | 1 paragraph of 4 sentences (15, 15, 15, 16) | 55 | `[s1 s2 s3, s4]`, not `"… third. Delta"` |
| 2.10 | Sentence ending in a closing quote: `…stronger.” Socrates…` | 55 | split after `.”`, not `"… Socrates was"` |
| 2.11 | 1 sentence of 4 `;`-clauses (25, 21, 19, 19) | 55 | `[c1 c2, c3 c4]`, not `"… rule; spirit"` |
| 2.12 | 10 words, no punctuation | 15 | `["one two three", "four five six", "seven eight", "nine ten"]` |
| 2.13 | A 20-char token between short words | 10 | token hard-split into two 10-char pieces, neighbours untouched |

**Overlap** (4 sentences of 15, 15, 15, 16 chars)

| # | `chunk_size` / `chunk_overlap` | Expected second chunk | Rule demonstrated |
|---|---|---|---|
| 2.14 | 55 / 0 | `"Delta is fourth."`, starting at or after the first chunk's end | No overlap means disjoint chunks |
| 2.15 | 55 / 25 | `"Gamma is third. Delta is fourth."` | Repeats the trailing sentence |
| 2.16 | 55 / 31 | `"Beta is second. Gamma is third. Delta is fourth."` | Repeats as many whole units as the budget allows |
| 2.17 | 55 / 10 | `"Delta is fourth."` | A unit larger than the budget is not repeated (never half a sentence) |
| 2.18 | 47 / 31 | `"Gamma is third. Delta is fourth."` | Overlap shrinks when the next new unit would not otherwise fit, guaranteeing progress |

**Provenance**

| # | Test |
|---|---|
| 2.19 | Every chunk carries its document's id and metadata |
| 2.20 | Chunk indices are `0..n-1` in order |
| 2.21 | Chunk ids are unique within a document and identical across repeated runs (idempotent re-indexing) |
| 2.22 | Chunk ids never collide across documents |

**Invariants** (property-based, over random prose-like and raw text and random valid sizes)

| # | Invariant |
|---|---|
| 2.23 | `document.text[chunk.start_char:chunk.end_char] == chunk.text` |
| 2.24 | `len(chunk.text) <= chunk_size` |
| 2.25 | Every non-whitespace character of the document lies inside at least one chunk |
| 2.26 | Chunk starts and ends both strictly increase |
| 2.27 | Characters shared by consecutive chunks never exceed `chunk_overlap` |
