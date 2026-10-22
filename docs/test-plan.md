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

Paths under `data/eval/` below name one-off measurement scripts (spikes) and their outputs, run on
Aren's laptop. That folder is not part of the repository; the scripts were deleted once their
results were recorded here and in `docs/experiments.md`.

## Test layers

| Layer | Marker | Runs by default | What it covers |
|---|---|---|---|
| Unit | none | yes | One component, no I/O. LLM, embedder, vector store and web search are replaced by in-memory fakes that implement the same `Protocol`. |
| Property-based | none | yes | Hypothesis generates thousands of inputs to check invariants that must hold for *any* input (e.g. "no text is lost"). |
| Integration | `integration` | no (`make test-integration`) | Real local infrastructure: on-disk Chroma, real embeddings, network. |
| LLM | `llm` | no (`make test-llm`) | The real configured models via Ollama (chat: `qwen3.5:35b-a3b` by default). Slow and non-deterministic, so they assert on *structure* (valid JSON, a grade from the allowed set), not exact wording. |

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
| 1.10 | `SourceMetadata.translator` is optional and defaults to `None` | Citations name the translator when there is one ("tr. Jowett"). |
| 1.11 | `Passage` (what an embedder receives) needs text; its title is optional but never empty | An empty passage would be embedded as noise. |
| 1.12 | `EmbeddingSpace.id` names provider, model, size (`native` when unset) and prompt format; the size must be positive | Vectors are only comparable within one space, so every part of it must be recorded. |
| 1.14 | `check_filters` accepts known metadata fields and rejects others, listing the known ones | A misspelled filter must fail, not silently match nothing. |
| 1.15 | `common_dimension` returns the shared vector size and raises `VectorDimensionError` (a `ValueError`) on mixed, unexpected or empty vectors | One rule for every vector store. |
| 1.16 | `SourceMetadata.citation` is "Author, Title, Section", or "Author, Title" without a section | One citation format for the index, the CLI and (later) generated answers. |
| 1.17 | `check_same_space` refuses an embedder and a store from different embedding spaces, naming both | Shared by the indexer and the retriever. |
| 1.18 | `ChatMessage` has the role `system`, `user` or `assistant` (the model's own earlier reply, sent back when it revises an answer: 19.5) and non-empty content | A role no adapter can map must fail here, not at the provider. |

## Component 2: Recursive chunker (`aporia_rag.ingestion.chunker`)

Splits a `Document` into `Chunk`s no longer than `chunk_size` characters, preferring the
largest natural boundary that fits: **paragraph, then sentence, then clause (`;` `:`), then
word, then raw characters** as a last resort. Chunks are exact spans of the original text.
Each chunk after the first repeats the longest tail of the previous chunk that fits in
`chunk_overlap` and starts at a whole-sentence boundary; failing that, a clause boundary; failing
that (text with no sentence or clause punctuation), a unit boundary. Never part of a word.

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
| 2.35 | `"Socrates asks.  \n\nGlaucon objects."` (spaces before the break) | 20 | `["Socrates asks.", "Glaucon objects."]`: the spaces belong to the break, not the chunk. Hypothesis did not find this case in 100 examples, so it is pinned here |

**Overlap** (4 sentences of 15, 15, 15, 16 chars)

| # | `chunk_size` / `chunk_overlap` | Expected second chunk | Rule demonstrated |
|---|---|---|---|
| 2.14 | 55 / 0 | `"Delta is fourth."`, starting at or after the first chunk's end | No overlap means disjoint chunks |
| 2.15 | 55 / 25 | `"Gamma is third. Delta is fourth."` | Repeats the trailing sentence |
| 2.16 | 55 / 31 | `"Beta is second. Gamma is third. Delta is fourth."` | Repeats as many whole units as the budget allows |
| 2.17 | 55 / 10 | `"Delta is fourth."` | A unit larger than the budget is not repeated (never half a sentence) |
| 2.18 | 47 / 31 | `"Gamma is third. Delta is fourth."` | Overlap shrinks when the next new unit would not otherwise fit, guaranteeing progress |

**Overlap inside paragraph-sized units.** Added after measuring that with paragraph units,
whole-unit overlap almost never applied (1.5% of boundaries at `chunk_size=1000`,
`chunk_overlap=150`), because a paragraph is nearly always longer than the overlap budget. Each
case is two paragraphs where the first paragraph fits in a chunk on its own; the old behaviour
would start the second chunk at paragraph 2 with no overlap.

| # | Paragraph 1 / paragraph 2 | `chunk_size` / `chunk_overlap` | Expected second chunk | Rule demonstrated |
|---|---|---|---|---|
| 2.28 | `"Alpha is first. Beta is second. Gamma is third."` / `"Delta is fourth. Epsilon is fifth."` | 55 / 25 | `"Gamma is third.\n\nDelta is fourth. Epsilon is fifth."` | Repeats the trailing whole sentence(s) of a paragraph unit |
| 2.29 | `"Reason rules; spirit helps. Appetite obeys."` / `"Plato agrees."` | 50 / 30 | `"Appetite obeys.\n\nPlato agrees."` | Prefers a whole sentence (15 chars) over a longer clause-started tail (29 chars) |
| 2.30 | `"Reason rules. The soul has three parts; appetite obeys."` / `"Plato agrees."` | 60 / 20 | `"appetite obeys.\n\nPlato agrees."` | Falls back to a clause when no whole sentence fits |
| 2.31 | `"Reason rules. The soul obeys its rational part."` / `"Plato agrees."` | 50 / 20 | `"Plato agrees."`, no overlap | Never descends below a clause, even when trailing words would fit |
| 2.32 | 10 words, no punctuation: `"one two … ten"` | 15 / 5 | chunks `["one two three", "three four five", "five six seven", "seven eight", "eight nine ten"]` | Text without punctuation overlaps on whole words (unit fallback) |

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
| 2.33 | Regression guard for the overlap gap: on prose where every sentence fits in the overlap budget (and paragraphs leave room for it), **every** chunk boundary has overlap |
| 2.34 | No chunk starts or ends with whitespace (overlap starts land on a real character) |

---

## Getting the corpus: components 3-6

Project Gutenberg [forbids automated access to its website](https://www.gutenberg.org/policy/robot_access.html)
(it blocks the IP) and directs scripts to its official mirrors, with a 2-second wait between
requests. Mirrors store ebook *N* under one folder per digit of *N* except the last
(`1497` → `1/4/9/1497/`). The UTF-8 text is `N-0.txt`, but older books only have `N-8.txt`
(Latin-1) and/or `N.txt` (ASCII): in our corpus, Hume (#9662) and Mill's *On Liberty* (#34901)
have no `-0.txt`. All layouts below were checked against the live mirror on 2026-10-04.

Network code is tested with `httpx.MockTransport`, an in-memory fake server, so unit tests never
touch the network. One opt-in integration test downloads a real (small) work.

## Component 3: Corpus manifest (`aporia_rag.ingestion.corpus`)

The list of works lives in a packaged TOML file, validated into `WorkSpec` / `CorpusManifest` models.

| # | Test |
|---|---|
| 3.1 | `WorkSpec.id` must be a lowercase slug (`plato-republic`); spaces, capitals, leading/trailing hyphens are rejected |
| 3.2 | `WorkSpec.gutenberg_id` must be positive |
| 3.3 | `WorkSpec.translator` is optional |
| 3.4 | `CorpusManifest` rejects duplicate work ids |
| 3.5 | `CorpusManifest` rejects duplicate Gutenberg ids |
| 3.6 | `load_manifest(path)` parses a TOML file into a `CorpusManifest` |
| 3.7 | The packaged manifest loads and contains exactly the 12 confirmed Gutenberg ids (contract test: an accidental edit fails loudly) |
| 3.8 | Every work has an edition `layout` (start marker, optional end marker, section heading patterns) |
| 3.9 | A layout rejects a pattern that is not a valid regular expression |
| 3.10 | A layout needs one or two section levels; zero or three are rejected |

## Component 4: Gutenberg mirror client and cache (`aporia_rag.ingestion.gutenberg`)

| # | Test |
|---|---|
| 4.1 | Mirror paths follow the digit layout: `1497` → `1/4/9/1497/`, `64576` → `6/4/5/7/64576/`, `59` → `5/59/`, and single-digit ids go under `0/` (`5` → `0/5/`) |
| 4.2 | Downloads the UTF-8 `N-0.txt` when present |
| 4.3 | Falls back to `N-8.txt`, decoded as Latin-1 (byte `0xE9` becomes `é`), when `N-0.txt` is missing |
| 4.4 | Falls back to plain `N.txt` when neither exists |
| 4.5 | Raises `EbookNotFoundError` naming the ebook when no candidate file exists |
| 4.6 | A server error (5xx) raises immediately instead of silently trying the next candidate |
| 4.7 | Waits `delay_seconds` before every request except the first (Gutenberg's 2-second rule), including 404 probes |
| 4.8 | Sends an identifying `User-Agent` |
| 4.9 | Cache hit: returns the cached file's path, marked as from the cache, without making any request |
| 4.10 | Cache miss: downloads, stores the text as UTF-8 at `cache_dir/pgN.txt`, returns that path |
| 4.11 | Creates the cache directory if it does not exist |
| 4.12 | A failed download writes nothing to the cache; a successful one leaves no temporary files |

## Component 5: Gutenberg text cleaning (`aporia_rag.ingestion.cleaning`)

`extract_body` removes Project Gutenberg's header and license; `normalize_text` turns hard-wrapped
plain text into one line per paragraph. Translator introductions, footnotes and section headings are
edition-specific and handled in a later component, after inspecting the real files.

| # | Test |
|---|---|
| 5.1 | `extract_body` returns only the text between the `*** START OF …` and `*** END OF …` markers |
| 5.2 | Marker variants are accepted: `THE`/`THIS`, `EBOOK`/`E-BOOK`, no space after `***`, any letter case |
| 5.3 | A missing START or END marker raises `ValueError` (never index the license as if it were Plato) |
| 5.4 | Older files' "End of the Project Gutenberg EBook of …" line is removed |
| 5.5 | `normalize_text` converts CRLF and lone CR line endings and drops a byte-order mark |
| 5.6 | Hard-wrapped lines inside a paragraph are joined with single spaces |
| 5.7 | Paragraphs are separated by exactly one blank line, however many there were |
| 5.8 | Underscore italics (`_Republic_`, `_res publica_`) lose their underscores; underscores inside words are kept |
| 5.9 | Runs of spaces and tabs collapse to one space |
| 5.10 | `clean_gutenberg_text` = `extract_body` then `normalize_text` |
| 5.11 | Property: `normalize_text` is idempotent |
| 5.12 | Property: output never has three consecutive newlines or a line with leading/trailing whitespace |
| 5.13 | Property: every character other than whitespace and `_` is preserved, in order |

## Component 6: `aporia-rag fetch` command (`aporia_rag.cli`)

| # | Test |
|---|---|
| 6.1 | Downloads every work in the manifest into `--cache-dir` and exits 0, reporting each work |
| 6.2 | A second run is served entirely from the cache: no requests |
| 6.3 | A work that cannot be found is reported, the remaining works are still fetched, and the exit code is 1 |

## Integration (opt-in: `make test-integration`, runs on a machine that can reach the mirror)

| # | Test |
|---|---|
| I.1 | Download Kant's *Metaphysic of Morals* (#5682, ~177 kB) from the real mirror; `clean_gutenberg_text` finds both markers and the body mentions "good will" |
| I.10 | The ddgs adapter searches the real web for "What did Wittgenstein mean by a language game?": at least one page, each with a web URL as its id and some text, and one of them mentions Wittgenstein |

---

## Structuring the editions: components 7-8

Designed from a spike over the 12 real downloaded files (2026-10-10). What the spike found:

* Every file is the author's text wrapped in material by someone else: Jowett's introduction is
  **46%** of the *Republic* file; J. A. Smith's introduction and the translator's notes wrap
  Aristotle; W. L. Courtney's introduction opens *On Liberty*; indexes close Hume, Seneca and Cicero;
  Nietzsche's file appends a separate fragment, *Peoples and Countries*.
* The Cicero edition is the bilingual Loeb: Latin paragraphs, Latin manuscript notes, Miller's
  margin summaries and the English translation alternate page by page.
* Footnote conventions vary: `[3]`, `[A]`, `[AB]`, `[Footnote A: ...]`, `* ...` (Kant), `word(3)`
  (Schopenhauer), `{12}` page numbers (Seneca), `*5*` section numbers and `#...#` margin notes (Cicero).
* One *Leviathan* heading lacks its full stop (`CHAPTER XL OF THE RIGHTS...`), which silently merged
  Chapter XL into XXXIX until the pattern allowed for it.

Each work's start marker, end marker and section heading patterns live in its `layout` in
`corpus.toml`; the clean-up rules below are shared by all works.

## Component 7: Edition structuring (`aporia_rag.ingestion.structure`)

**Boundaries**

| # | Test |
|---|---|
| 7.1 | Everything before the start paragraph (title page, contents, translator's introduction) is excluded |
| 7.2 | Everything from the end paragraph onward (notes, index, appended works) is excluded |
| 7.3 | A missing start or end paragraph raises `StructureError` |
| 7.4 | A start paragraph that is not itself a section heading raises `StructureError` (every section must have a label) |

**Sections**

| # | Test |
|---|---|
| 7.5 | Each level-1 heading starts a new section, labelled from the heading |
| 7.6 | With two levels, labels combine both (`Section IV, Part II`) and a new level-1 heading resets level 2 |
| 7.7 | Heading paragraphs are not part of any section's text |
| 7.8 | A heading followed directly by another heading yields no empty section (`PART I. OF MAN` then `CHAPTER I.`) |
| 7.9 | Explicit labels replace level-1 heading text, in order |
| 7.10 | A number of explicit labels different from the number of level-1 headings raises `StructureError` |

**Shared clean-up**

| # | Test |
|---|---|
| 7.11 | Footnote paragraphs are dropped: `[3] ...`, `[A] ...`, `[AB] ...`, `[Footnote A: ...]`, `* ...` |
| 7.12 | `FOOTNOTES:` headings, `#margin notes#`, `[Illustration...]`, `SEC_1` anchors and `{12}` page paragraphs are dropped |
| 7.13 | Inline references are removed: `word[3]`, `word[A]`, `word[AB]`, `word(3)`, a lone ` * `, `{12}`, `*5*`, and a `#margin note#` merged into a paragraph (71 such notes in Cicero, found by an end-to-end scan) |
| 7.14 | `[Greek: arete]` becomes `arete` |
| 7.15 | Paragraphs without a lowercase letter (leftover titles, bare numbers like `27.`) are dropped |
| 7.16 | Look-alikes are kept: an enumeration ` (1) first`, a bracketed aside `[Serenus.]` inside prose |

**English-only filter** (for the bilingual Cicero edition)

| # | Test |
|---|---|
| 7.17 | Latin paragraphs are dropped and English ones kept |
| 7.18 | Accents do not hide Latin (`ést` counts as `est`) |
| 7.19 | A paragraph with no evidence either way (Latin verse `Cedant arma togae`) takes the previous paragraph's language, in both directions |
| 7.20 | One-sided evidence beats inheritance: a Latin paragraph opening a new page after English (`Restat quarta pars, quae ...`) is dropped |
| 7.21 | Without `english_only`, Latin paragraphs are kept (e.g. Schopenhauer's Latin epigraphs) |

**Labels** (`format_label`)

| # | Input | Expected |
|---|---|---|
| 7.22 | `BOOK IV.` | `Book IV` |
| 7.23 | `CHAPTER XXVI. OF CIVILL LAWES` | `Chapter XXVI. Of Civill Lawes` (`CIVILL` is not a Roman numeral) |
| 7.24 | `A REVIEW, AND CONCLUSION` | `A Review, and Conclusion` (minor words lowercase except at a clause start) |
| 7.25 | `ARISTOTLE'S ETHICS` | `Aristotle's Ethics` |
| 7.26 | `Chapter III.` (already mixed case) | `Chapter III` |

## Component 8: Document loader (`aporia_rag.ingestion.loader`)

| # | Test |
|---|---|
| 8.1 | Yields one `Document` per section, in order, with author, title, translator, `source_id` (`gutenberg:N`) and section label |
| 8.2 | Document ids are `<work id>/<slug of label>`, e.g. `seneca-minor-dialogues/of-anger-book-i` |
| 8.3 | Two sections with the same label in one work raise `StructureError` instead of producing colliding ids |
| 8.4 | Lazy: documents of earlier works are yielded before a later work's missing file is even looked at |
| 8.5 | A missing cached file raises `FileNotFoundError` naming the work and pointing to `aporia-rag fetch` |
| 8.6 | A cached file that is not a Gutenberg text, or does not match its layout, raises `CorpusError` naming the work |

## Real-corpus contract tests (opt-in: `make test-integration`; skipped where `data/raw` is not populated)

| # | Test |
|---|---|
| I.2 | Every work structures without error, into the expected number of sections: Republic 10, Nicomachean Ethics 119, De Officiis 3, Seneca 14, Kant 4, Utilitarianism 5, Genealogy 4, Discourse 7, Hume 19, Schopenhauer 6, Leviathan 49, On Liberty 5 |
| I.3 | The *Republic* starts at Plato's text ("I went down yesterday to the Piraeus"), not Jowett's introduction |
| I.4 | *Leviathan* has its Chapter XL (regression: the heading without a full stop) |
| I.5 | *De Officiis* keeps no paragraph with two or more Latin function words, and starts with "My dear son Marcus" |
| I.6 | No document contains Gutenberg boilerplate, a footnote paragraph or an index |
| I.7 | Document ids are unique across the whole corpus |
| I.8 | No `#margin note#` survives anywhere in *De Officiis* (Miller's summaries are not Cicero's words) |
| I.9 | Every piece of evidence in the packaged gold sets (`gold_retrieval.toml`, `gold_loop.toml`) names an existing document, and its quote occurs in that document and inside at least one chunk at the configured chunk size and overlap |

Known residue, accepted as low-impact: `***` lacuna marks, and about 30 unbalanced italics underscores (`_any_thing`) across 6.4 million characters.

---

## Indexing and retrieval: components 9-15

**Design rule: provider independence.** Nothing outside `adapters/` knows which model provider or
vector database is in use. The application depends on interfaces in `domain/ports.py`:
`Embedder`, `VectorStore` and `Chunker` (and `ChatModel`, from components 18-20 on). Adapters implement
them; `bootstrap.py` chooses an adapter by name from the settings through a small registry, so
adding Google, Voyage or Anthropic later means one new adapter class and one registration, with no
change to the indexer, retriever or graph.

Providers express "this is a query" vs "this is a passage" differently: text prefixes (Qwen3,
EmbeddingGemma, Nomic, Google's `gemini-embedding-2`), an API parameter (`task_type` in
`gemini-embedding-001`, `input_type` in Voyage AI, the provider Anthropic recommends since it has no
embedding model of its own), and a title slot (EmbeddingGemma, Gemini). So the `Embedder` interface
states the *intent* (`embed_query` vs `embed_documents` of titled passages) and each adapter maps it
to its provider's mechanism. Prompt formats are per model family, not per provider, so a future
Gemini adapter can reuse the EmbeddingGemma format.

Test doubles live in `tests/support/fakes.py`: `HashingEmbedder` (feature hashing: texts sharing
words get similar vectors, so retrieval tests are meaningful without a model), `InMemoryVectorStore`,
and `FakeOllamaServer` (an `httpx` handler emulating Ollama's `/api/embed`). The vector store tests
are one contract suite run against both the in-memory fake and real Chroma, so the fake used by
later graph tests is proven to behave like the real store.

## Component 9: Settings (`aporia_rag.config`)

| # | Test |
|---|---|
| 9.1 | Defaults: Ollama at `http://localhost:11434`, embedding `ollama` / `qwen3-embedding:0.6b`, chat `ollama` / `qwen3.5:35b-a3b-q4_K_M` with a context of 8192 tokens, Chroma index in `data/index`, corpus in `data/raw`, chunks 1000/150 |
| 9.2 | Nested environment variables override defaults (`APORIA_EMBEDDING__MODEL=...`) |
| 9.3 | Values are read from a `.env` file, and real environment variables win over it |
| 9.4 | Invalid values are rejected: batch size 0, chunk overlap not smaller than chunk size |

## Component 10: Embedders (`aporia_rag.adapters.prompt_formats`, `.embedding`, `.ollama`)

**Prompt formats** (one strategy class per model family)

| # | Test |
|---|---|
| 10.1 | Qwen3: query is `Instruct: {instruction}\nQuery:{text}` with the model's documented default instruction |
| 10.2 | Qwen3: a custom instruction is used and changes the format's name (so it changes the embedding space) |
| 10.3 | Qwen3: a passage is `{title}\n\n{text}`, or just the text without a title |
| 10.4 | EmbeddingGemma: `task: search result \| query: {text}` and `title: {title or none} \| text: {text}` |
| 10.5 | Nomic: `search_query: ` and `search_document: ` prefixes; a custom instruction is rejected (fixed prefixes) |
| 10.6 | The format is found from the model name: `qwen3-embedding:8b`, `embeddinggemma-2:440m`, `nomic-embed-text:v1.5` |
| 10.7 | An unknown model family raises `ConfigurationError` listing the known families |

**Batching base class** (`BatchingEmbedder`, shared by every provider adapter)

| # | Test |
|---|---|
| 10.8 | Passages are sent in batches of `batch_size`, order preserved |
| 10.9 | Queries are sent with the QUERY purpose, passages with the DOCUMENT purpose |
| 10.10 | An empty list of passages returns `[]` without calling the provider (a blank query is the `Retriever`'s to reject: 13.5) |
| 10.11 | The wrong number of vectors back raises `ProviderError` |
| 10.12 | Vectors of inconsistent size, or not of the configured size, raise `ProviderError` |
| 10.13 | `batch_size` must be positive |

**Ollama adapter** (against `FakeOllamaServer` through `httpx.MockTransport`)

| # | Test |
|---|---|
| 10.14 | Posts to `/api/embed` with the model, the formatted inputs, `truncate: false` and `keep_alive`; `dimensions` only when configured |
| 10.15 | Returns the server's vectors in order |
| 10.16 | A model that is not pulled raises `ModelNotAvailableError` naming `ollama pull <model>` |
| 10.17 | Ollama not running raises `ProviderUnavailableError` |
| 10.18 | Any other failure (server error, timeout) raises `ProviderError` |
| 10.19 | Its embedding space is `ollama` / model / dimensions / prompt-format name |
| 10.20 | `close()` releases its HTTP connections |
| 10.21 | Queries are embedded on the CPU (`options: {num_gpu: 0}`), so a chat model keeps the GPU; passages (indexing) are not restricted |

## Component 11: Vector stores (`aporia_rag.adapters.chroma` and the in-memory fake)

**Contract** (every test runs against both `InMemoryVectorStore` and `ChromaVectorStore`)

| # | Test |
|---|---|
| 11.1 | An empty store has count 0, no fingerprints, and returns no results |
| 11.2 | A query returns the `k` nearest chunks, best first |
| 11.3 | Scores are cosine similarities: same direction 1.0, orthogonal 0.0, opposite -1.0 |
| 11.4 | Chunks come back intact: text, offsets, index, document id and all metadata (including a missing translator) |
| 11.5 | Upserting an existing id replaces it: no duplicate, new text and fingerprint |
| 11.6 | `k` larger than the store returns everything |
| 11.7 | A filter on `author` restricts results; several filter fields combine with AND |
| 11.8 | An unknown filter field raises `ValueError` (a typo must not silently return nothing) |
| 11.9 | `fingerprints()` maps every chunk id to its fingerprint |
| 11.10 | `delete` removes the given ids; unknown ids are ignored |
| 11.11 | `clear` empties the store |
| 11.12 | A vector of the wrong size raises `VectorDimensionError`, on upsert and on query, and within one batch |

**Chroma only**

| # | Test |
|---|---|
| 11.13 | Data survives closing and reopening the store at the same path |
| 11.14 | A collection built in another embedding space is refused with `IndexMismatchError` |
| 11.15 | Collection names satisfy Chroma's rules (3-63 characters, alphanumeric ends, no `..`) for any model name, and different spaces get different names |
| 11.16 | Upserts larger than the batch limit are split and all stored |
| 11.17 | The persistent client is created with telemetry disabled |

## Component 12: Indexer (`aporia_rag.indexing.indexer`)

| # | Test |
|---|---|
| 12.1 | The first run embeds and stores every chunk; the report counts them as added |
| 12.2 | A second run embeds nothing and reports everything unchanged |
| 12.3 | When a document changes, only its changed chunks are re-embedded (reported as updated) |
| 12.4 | Chunks that are no longer produced are deleted from the store |
| 12.5 | Each chunk is embedded as a passage titled with its citation (author, work, section); the store keeps the raw chunk text |
| 12.6 | Embedding happens in groups of `flush_size` |
| 12.7 | Progress is reported once per work |
| 12.8 | An embedder and a store from different embedding spaces are refused with `IndexMismatchError` |
| 12.9 | A passage's fingerprint changes when its title or text changes and is otherwise stable |
| 12.10 | Two chunks with the same id are an error (they would silently overwrite each other) |
| 12.11 | An interrupted run keeps what it stored; the next run embeds only the rest |

## Component 13: Retriever (`aporia_rag.retrieval.retriever`)

| # | Test |
|---|---|
| 13.1 | The question is embedded as a query, not as a passage |
| 13.2 | Returns the `k` best chunks, best first (a "good will" question finds the Kant passage) |
| 13.3 | Filters are passed through to the store |
| 13.4 | An embedder and a store from different embedding spaces are refused |
| 13.5 | `k` must be positive and the question non-blank |
| 13.6 | `best_per_author` embeds the question once and returns each author's best chunk, for the `limit` authors whose best chunk scores highest, best first |
| 13.7 | `best_per_author` leaves out an author without chunks; `limit` must be positive and the question non-blank |

## Component 14: Composition root (`aporia_rag.bootstrap`)

| # | Test |
|---|---|
| 14.1 | Builds an Ollama embedder from the settings, with the prompt format inferred from the model name |
| 14.2 | An unknown provider raises `ConfigurationError` listing the available ones |
| 14.3 | A newly registered provider is built without changing any builder (the extension point) |
| 14.4 | Registering the same provider name twice is an error |
| 14.5 | Builds a Chroma store in the configured index directory |
| 14.6 | The `Container` builds each component once, and only when first used (so an unusable provider fails only when it is needed) |
| 14.7 | An index built through the container's indexer is searchable through its retriever (they share the embedder and the store) |
| 14.8 | The container is a context manager: leaving it closes the components it built that hold connections, even when the block raises; a container that built nothing closes nothing |
| 14.9 | Builds an Ollama chat model from the settings (host, model, `keep_alive`, context length); an unknown chat provider raises `ConfigurationError` listing the available ones |
| 14.10 | The container's generator, relevance grader and query rewriter share its chat model, built once and only when first used (so `search` never needs one), and closed on leaving the container |
| 14.11 | The container's corrective graph is built once, from the container's retriever, grader, generator, rewriter and web search |
| 14.12 | The container builds the web search named in the settings (`web.provider`, `ddgs` by default) once, and only when first used; an unknown web search provider raises `ConfigurationError` listing the available ones; a web search given to the container replaces it (ddgs does not go through httpx, so this is how tests replace it) |

## Component 15: `index` and `search` commands (`aporia_rag.cli`)

| # | Test |
|---|---|
| 15.1 | `index` indexes the corpus and reports added, updated, unchanged and deleted counts |
| 15.2 | A second `index` embeds nothing |
| 15.3 | `index --rebuild` empties the index and embeds everything again |
| 15.4 | `search` prints ranked results with similarity, citation, chunk id and a snippet |
| 15.5 | `search --author` restricts results to that author (any part of the name, any case); an unknown author is an error listing the authors |
| 15.6 | `search` on an empty index says to run `aporia-rag index` and exits 1 |
| 15.7 | `search --all-authors` shows the best passage of each author, best first; it cannot be combined with `--author` |
| 15.8 | `search` keeps to the authors named in the question and says so on its first line: one named author as with `--author`, two or more as with `--all-authors` over them; `--author` and `--all-authors` take precedence over the names |
| 15.7 | Ollama not running gives a one-line error and exit code 1, no traceback |
| 15.8 | `index` before `fetch` says to run `aporia-rag fetch` |
| 15.9 | A corrupt corpus file is a one-line error naming the work, like every other expected failure |
| 15.10 | Commands release their connections before returning |

---

## Measuring retrieval: components 16-17

The embedding bake-off compares models on a **gold set**: questions written by hand, each with
the evidence a good retriever should find. Evidence is a section (document id) plus a short
verbatim quote from the passage that answers the question. The packaged set,
`evaluation/gold_retrieval.toml`, has 40 questions covering all 12 works, written in a reader's
words rather than the source's (a bag-of-words baseline finds only 15% of them in its top 5),
naming the author in some questions and not in others, with Hobbes in his 1651 spelling.

Relevance is judged at two levels, because they fail for different reasons:

- **Passage hit**: the chunk comes from the evidence section *and* contains the quote. This is
  what the generator needs to answer, so it is the headline metric.
- **Section hit**: the chunk only comes from the evidence section. A section hit without a
  passage hit means "right place, wrong paragraph", e.g. chunk dilution; a miss at both levels
  means retrieval did not understand the question.

Quotes are matched rather than chunk ids so that the gold set survives changes to chunking: a
chunk-size experiment re-chunks everything, but the quote is still in whichever chunk holds it.

Metrics come from one top-10 retrieval per question: **Hit@k** (share of questions with a hit in
the top k) and **MRR@10** (mean of 1/rank of the first hit, 0 for a miss). With 40 questions,
one question is 2.5 points: differences smaller than that are noise.

The evaluation depends on a small `Searcher` protocol (`search(question, *, k)`) defined next to
it, which `Retriever` satisfies structurally; so `evaluation` imports neither `retrieval` nor any
provider, and later the corrective graph can be measured the same way.

**Pooling.** An answering passage missing from the gold set counts as a miss, so the evidence must
list every passage that answers a question. Keyword searches of the corpus find the passages that
share words with what was searched for; the passages a model finds in other words are found by
reading what it retrieved. `evaluate --dump` writes exactly that: every retrieved chunk with its
verdicts, so the non-hits can be judged and the answering ones added as evidence (as in TREC, where
the union of all systems' top results is judged).

## Component 16: Retrieval evaluation (`aporia_rag.evaluation`)

| # | Test |
|---|---|
| 16.1 | `Evidence` needs a non-blank document id and quote; `GoldQuestion` needs a slug id, a non-blank question and its evidence, which must be given but may be empty (a question the corpus does not answer, component 23); unknown fields are rejected |
| 16.2 | `GoldSet` needs at least one question and rejects duplicate question ids |
| 16.3 | `load_gold_set` parses a TOML file; the packaged gold set loads and has evidence in every work of the corpus manifest |
| 16.4 | A chunk is a passage hit when it comes from the evidence's document and contains the quote, ignoring differences in whitespace; the same quote in another document is not a hit; a section hit only needs the document |
| 16.5 | A question with several pieces of evidence is hit by any one of them |
| 16.6 | `hit_rate(ranks, k)` is the share of ranks within k; `mean_reciprocal_rank(ranks)` averages 1/rank with misses counting 0 |
| 16.7 | `evaluate_retrieval` asks the searcher once per question for the top 10 and reports, in gold-set order, the rank of the first passage hit and of the first section hit (`None` for a miss) |
| 16.8 | The report keeps every question's retrieved chunks, in rank order, so what was judged can be inspected |
| 16.9 | `write_retrievals` writes one JSON line per model, question and retrieved chunk: the question, rank, score, chunk id, citation, chunk text, and whether the chunk is a passage hit and a section hit |
| 16.10 | A question without evidence is searched and its retrieved chunks are kept like any other, but it has no passage or section rank: the metrics leave it out instead of counting a miss |

## Component 17: `evaluate` command (`aporia_rag.cli`)

| # | Test |
|---|---|
| 17.1 | `evaluate` prints passage and section metrics for the configured embedding model and the ranks for every question |
| 17.2 | `evaluate --model A --model B` evaluates each model on its own index (built with that model) and prints one column per model |
| 17.3 | A model whose index is empty is a one-line error naming the model and the command that indexes it |
| 17.4 | `evaluate` releases the connections of every model it evaluated |
| 17.5 | `evaluate --dump PATH` also writes every model's retrieved chunks to PATH (16.9), creating its folder, and says where; the printed report is unchanged |
| 17.6 | `evaluate` says how many questions have no evidence and are not scored, and shows `no evidence` instead of their ranks |

---

## Answering questions: components 18-22

The baseline RAG pipeline: retrieve the `k` passages most similar to the question (component 13),
then have a chat model answer from them (the generation step: the "G" in RAG). Like the retriever it
is deliberately plain: no grading, no query rewriting, no web search. The corrective graph adds
those, and the evaluation will measure what each adds over this baseline.

**Grounding and citations.** The model receives the passages numbered from 1, each headed by its
citation, and the question last; the system prompt tells it to answer only from the passages that
bear on the question, to cite them by number (`[2]`), and to say so when they do not answer it. `Answer.sources[n -
1]` is the passage cited as `[n]`, so every citation leads back to an exact chunk. Whether the model
really keeps to the passages is for the graders to check later; the baseline only asks.

**Provider independence.** The `ChatModel` port is "messages in, reply out". Providers take the
system prompt differently (a message with the `system` role in Ollama and OpenAI, a separate
`system` parameter in Anthropic, a `system_instruction` in Gemini), so `ChatMessage` states the
role and each adapter maps it. The tests use `ScriptedChatModel`, a fake that records the messages
it receives and returns a fixed reply, and `FakeOllamaServer`, which now also answers `/api/chat`.

**Survey of all authors.** `--all-authors` replaces the top `k` with the best passage of each author
(`best_per_author` over the whole manifest), the grader keeps the relevant ones, and the generator's
survey is a structured reply with exactly one paragraph per passage, which the generator heads with
the author and the passage number. Asked in the prompt, numbered citations were a tendency: the
first live run of L.6 returned two correct paragraphs and no number. Eleven passages (the 12 works
have 11 authors), the instructions and eleven paragraphs come to up to about 4,500 tokens, more than
Ollama's default context of 4096 on an 8 GB GPU, so the chat model gets a configured context length
on every request.

## Component 18: Ollama chat model (`aporia_rag.adapters.ollama`)

| # | Test |
|---|---|
| 18.1 | Posts to `/api/chat` with the model and the messages as role/content pairs, in order, with thinking off; `keep_alive` only when configured |
| 18.2 | Returns the reply's text without surrounding whitespace |
| 18.3 | An empty reply raises `ProviderError` (there is nothing to show; a thinking model that only thought, say) |
| 18.4 | A model that is not pulled, Ollama not running and any other failure raise the same errors as for the embedder (10.16-10.18) |
| 18.5 | `close()` releases its HTTP connections |
| 18.6 | `chat_structured` sends the messages with the schema's JSON schema as `format`, thinking off and temperature 0 (a decision must be repeatable) |
| 18.7 | `chat_structured` returns the reply parsed into the schema |
| 18.8 | A structured reply that does not fit the schema raises `ProviderError` naming the schema; failures are translated as for `chat` |
| 18.9 | Every request sets the configured context length (`num_ctx`), the same for `chat` and `chat_structured`: Ollama reloads a model whose context length changes |
| 18.10 | A JSON schema that Ollama cannot turn into a grammar is a `ProviderError` with Ollama's reason. `FakeOllamaServer` rejects what Ollama rejects (a regex `pattern` not anchored with `^...$`): the first live run of the rewriter failed that way while every fake test passed (2026-10-20) |
| 18.11 | Every request sends the caller's reply cap (`max_tokens`) as Ollama's `num_predict`, for `chat` and `chat_structured` |
| 18.12 | A reply that Ollama cuts at the cap (`done_reason` "length") is a `ProviderError` that names the model and the cap, for `chat` and `chat_structured`: a reply that does not stop fails in seconds, not at the 120 s time-out (found 2026-10-21: a one-field rewrite ran on until the time-out). `FakeOllamaServer` cuts a reply longer than `num_predict`, a word for a token, and says so, as Ollama does |

## Component 19: Generator (`aporia_rag.generation.generator`)

| # | Test |
|---|---|
| 19.1 | The model receives the system prompt, then one user message: the passages numbered from 1 in the order given, each headed by its citation and followed by its exact text, then the numbers it may cite, and the question last |
| 19.2 | The answer is the model's reply, with the passages as its sources, in the order of their numbers |
| 19.3 | `survey` sends the same passages and question (without the numbers to cite) with the survey prompt as a structured request for exactly one paragraph per passage, and returns one section per passage, headed by the passage's author and number (`Immanuel Kant [2]`), with the passages as its sources |
| 19.4 | The generator caps an answer at 1,024 tokens (the longest so far: about 550) and a survey at 256 tokens per passage |
| 19.5 | `revise` sends the answer's messages, then the draft as the model's own reply, then one user message listing what to fix and asking for the whole answer again; it returns the new answer, with the same passages as its sources, capped like an answer |
| 19.6 | The numbers the model may cite are named for it: "There is one passage: cite it as [1]", "...cite them as [1] and [2]", "...as [1] to [5]"; the answer prompt gives no example number (2026-10-21: with one passage, 4 of 5 answers that made up citations cited the prompt's example, [2]) |

## Component 20: `ask` command (`aporia_rag.cli`)

| # | Test |
|---|---|
| 20.1 | `ask` prints the model's answer, then the passages it was given, numbered as the model saw them, with citation and chunk id |
| 20.2 | `ask -k N` gives the model N passages, and `ask --author` restricts them to one author, as in `search` |
| 20.3 | `ask` on an empty index says to run `aporia-rag index` and never calls the chat model |
| 20.4 | A chat model that is not pulled is a one-line error naming `ollama pull <model>` |
| 20.5 | `ask` releases its connections, the chat model's included |
| 20.6 | `ask` ends with how long the answer took, split into rewriting (only when a question was rewritten), retrieval, grading, web search (only when it ran), generation and the search for other authors (a scripted clock makes the numbers exact) |
| 20.7 | `ask` grades every retrieved passage: the model answers from the relevant ones only, numbered from 1, and the passages left out are listed as not relevant |
| 20.8 | When no passage is relevant, even after the rewritten search (25.2) and the web search (26.5), `ask` says so, counting every passage graded, lists them all as left out (as when some are relevant), never asks the model for an answer, and does not look for other authors |
| 20.9 | With `--others`, after the answer, `ask` grades the best passage of the other authors (those with no passage among the passages found, graded or not, at most 4) and lists the relevant ones as other authors who discuss the question |
| 20.10 | With `--author`, `ask` does not look for other authors |
| 20.11 | `ask --all-authors` grades the best passage of every author, prints the survey of the relevant ones, lists the others as not relevant, and does not look for other authors (the question is embedded once) |
| 20.12 | `ask` with one author named in the question answers from that author's passages only and does not look for other authors |
| 20.13 | `ask` with two or more authors named in the question compares them: it prints the question it asked about each author on its own (25.6), and surveys the best relevant passage of each, one section per author, in the order of the names |
| 20.14 | When the first search finds nothing relevant, `ask` says that it searched again and with which query, answers from what that search found, and lists the passages of both searches that it left out |
| 20.15 | When the rewriter names no author in the corpus, `ask` says so, quoting what it named, and that it searched the web, prints the answer with its sources under "Web sources" (citation and URL), lists what it left out, and does not look for other authors |
| 20.16 | When the web search fails, `ask` says why, still lists the passages it left out, and exits with 0 |
| 20.17 | Without `--others`, `ask` does not look for other authors: no verdict beyond those of the answer, and no "other authors" in the time report |
| 20.18 | `ask` lists the passages it did not grade, because enough relevant ones came before them (27.1), after those it left out |
| 20.19 | When the first draft had problems, `ask` says that it wrote the answer again, and lists under the answer the problems that remain |

## Component 21: Relevance grader (`aporia_rag.grading.relevance`)

Retrieval returns the passages *nearest* to a question, which is not the same as the passages that
*answer* it: a Hobbes chapter on honour is near a question about the dishonourable, and every
retrieved passage gets a paragraph in the answer. The grader asks the chat model about each passage
separately, with a structured yes/no reply, and keeps those judged relevant. One call per passage
keeps every prompt short and every verdict independent of the other passages.

| # | Test |
|---|---|
| 21.1 | Each passage is graded in its own call: the grading system prompt, then one user message with the passage's citation, its exact text and the question last, asking for a `RelevanceVerdict` |
| 21.2 | `relevant` returns the passages judged relevant, in their original order |
| 21.3 | `is_relevant` grades one passage in one call and returns the model's verdict (component 23 measures the grader through it) |
| 21.4 | The grader caps a verdict at 16 tokens (`{"relevant": false}` is about 6) |

**Questions with several parts (2026-10-20).** Component 23 found one systematic error: the grader
dropped passages that answer one part of a question with several parts, or one side of a
comparison (three music passages for "Why does music move us more deeply than painting or
poetry?", four for "Why do we expect the future to resemble the past?"). "How do Kant and Mill
differ on lying?" failed the same way: Kant's passage on truthfulness from duty was kept for "What
does Kant say about lying?" and dropped for the comparison. A prompt line saying that a passage
answering one part, or giving one author's view, helps was measured and reverted: on the same
chunks it kept 25 more (8 evidence, 17 not), so top-5 precision fell from 0.93 to 0.89, it
recovered only 2 of the 7 dropped passages, one more question no longer started the loop, and
L.7 still failed. The grader stays as it was. L.7 is kept as an expected failure (strict, so it
reports when the grader passes it) that records the limitation; the fix for comparisons belongs
to the caller, which can ask about one author at a time.

## Component 22: Authors named in the question (`aporia_rag.retrieval.names`)

A question that names an author ("How does Aristotle define moral virtue?") is about that author. In
the 2026-10-16 dump, the 15 gold questions that name one author already retrieve almost only that
author (1 of their 150 top-10 chunks is another author's), so the filter changes no result there.
What it changes is the rest of `ask`: no search for other authors (8-10 s and a list the question
did not ask for), and for two or more names, a section for each named author. Matching is plain
code over the manifest's author names: the surname, as a whole word, in any case.

| # | Test |
|---|---|
| 22.1 | A surname names its author in any case and in the possessive (`kant`, `Kant's`, `Hobbes'`), and so does the full name |
| 22.2 | Only whole words count: `Kantian` and `Platonic` name nobody, nor does a question without names |
| 22.3 | Every author named is returned, in the order the authors are given |

## Measuring the grader: component 23

The corrective loop (query rewriting, then the web) will start from the grader's verdicts; in the
simplest rule, when the grader keeps no retrieved passage. The grader is the switch of the loop,
and it can be wrong in two ways with different costs:

- **A false "not relevant"** drops a passage that answers. If it drops all of them, the loop
  rewrites the query or searches the web for nothing: time lost, and a weaker web answer can
  replace a good one from the corpus.
- **A false "relevant"** keeps a passage that does not answer. The loop does not start when it
  should, and the generator writes a paragraph about a passage that is beside the point (the
  failure the grader was built against).

The first live run (15 of 15 verdicts right) is too small to show either rate. The gold set already
holds the labels: the 2026-10-16 pooling judged every chunk that the three bake-off models ranked
in a top 10, so for `qwen3-embedding:0.6b` on its current index every retrieved chunk is either
evidence or was judged not to answer. `evaluate --grade` therefore retrieves the top 10 of every
gold question as usual, asks the grader about each chunk, and compares its verdicts with the gold
set, at the top 5 (the default of `ask`) and the top 10:

- **Agreement:** precision (the share of the kept passages that are evidence) and recall (the share
  of the evidence that is kept).
- **Loop starts:** the questions with no relevant passage in the top k, by the gold set and by the
  grader. Those of the grader only are needless loops; those of the gold set only are missed loops.
- **Time:** the mean grading time per question, which decides the parked options (one batched call;
  2-3 passages per author in a survey).
- **Every disagreement,** with its question, rank and chunk id, to be read: the gold set's rule
  ("the passage states or argues the author's answer to the question as asked") is stricter than the
  grader's prompt ("states, explains or argues something the question asks about, even in part"), so
  some disagreements are the grader being lenient by design, some are grader errors, and some may be
  missing evidence.

Limits: another embedding model, or a re-built index, retrieves chunks the pool never judged, so
the labels are only complete for the configuration that was pooled. Retrieval here is unrestricted,
while `ask` searches only the author a question names; for the 15 such gold questions this changes
1 of 150 top-10 chunks.

**Loop questions** (`evaluation/gold_loop.toml`, a separate file so the 40 retrieval questions
stay comparable with earlier runs): questions where plain retrieval is expected to fail. Some use
modern words for an old text (Hobbes's "Covenants extorted by feare", Aristotle's "incontinence"),
and some ask what the corpus does not contain (a work or an author outside it). The latter have
`evidence = []`: every passage retrieved for them is a correct "not relevant", the loop must start,
and later the web fallback is their only route. `evaluate --gold <file>` gives their retrieval
baseline and `--grade` shows whether the grader starts the loop for them.

The evaluation depends on a `Judge` protocol (`is_relevant(question, passage)`) declared next to
it, which `RelevanceGrader` satisfies structurally, as `Retriever` satisfies `Searcher`.

## Component 23: Grader evaluation (`aporia_rag.evaluation.grading`, `aporia_rag.cli`)

| # | Test |
|---|---|
| 23.1 | `evaluate_grading` asks the judge once about every retrieved chunk of every question, in gold-set and rank order, and records the question, rank, chunk id, whether the chunk is evidence (a passage hit) and the verdict |
| 23.2 | Each verdict records how long the judge took (a scripted clock makes it exact) |
| 23.3 | `agreement(report, k)` counts, over the top k of every question, the chunks kept and evidence, kept only, evidence only, and neither; precision is kept evidence over kept, recall is kept evidence over evidence; ranks after k do not count |
| 23.4 | Precision with nothing kept and recall with no evidence are `None`, not a division by zero |
| 23.5 | `loop_starts(report, k)` lists, in gold-set order, the questions with no relevant chunk in their top k by both the gold set and the grader, by the grader only (needless loops) and by the gold set only (missed loops) |
| 23.6 | `seconds_per_question(report, k)` is the mean, over questions, of the time spent on their top k verdicts |
| 23.7 | `evaluate --grade` grades every retrieved chunk with the chat model (one structured request each) and prints, after the retrieval report: the agreement and grading time at the top 5 and the top 10, the questions where the loop would start, and every disagreement with its question, rank and chunk id |
| 23.8 | Without `--grade`, `evaluate` sends no request to the chat model |
| 23.9 | The packaged loop question set loads, and has questions with and without evidence |

---

## The corrective loop: components 24-25

Measured before it was built (2026-10-20). The grader keeps no passage in the top 5 for 2 of the
40 retrieval questions whose answer is in the corpus, and for all 4 questions outside it. For the
in-corpus failures, a search query in the author's own vocabulary finds the evidence: the chat
model's rewrites put it in the top 5 for all 3 retrieval questions that fail (ranks 1-5, 5 and 4,
against none in the top 5), in 1.5-1.9 s each (`data/eval/spike_rewrites.py`). Comparisons fail
differently: for "How do Kant and Mill differ on lying?" the grader leaves out every passage, even
those on lying, but keeps each side when asked "What does Immanuel Kant say about lying?" and
"What does John Stuart Mill say about lying?" (3 of 3; two plain-code framings of the original
question scored 1 of 3; `data/eval/spike_comparison.py`).

So the loop adds model calls only where they are needed:

- **Normal path** (a passage is kept): unchanged, one verdict per passage and one answer.
- **Loop path** (nothing kept): one rewrite, one more search with the same `k` and scope, verdicts
  for the passages not graded yet, then an answer or "none". At most one rewrite: one was enough in
  the spike. Questions outside the corpus pay for it too (about 9 s) until the web fallback.
- **Comparison path** (two or more authors named): one call for the subject of the comparison;
  plain code then asks the same question about each author. A survey of all authors
  (`--all-authors`) is not a comparison and is never rewritten.

The rewriter's prompt is the one the spike measured. The search query it writes is used for the
search only: verdicts and the answer are always about the reader's question.

**Why LangGraph.** The loop is a small state machine: retrieve, grade, then generate, rewrite or
stop. LangGraph states it as nodes and conditional edges, which the web fallback (component 26)
extends with one node and one edge, and it draws the graph (Mermaid) for the README. It is the only
package that uses LangGraph (A.3); the graph sits above the components it calls and below the
composition root that builds it (A.1).

## Component 24: Query rewriter (`aporia_rag.retrieval.rewriter`)

| # | Test |
|---|---|
| 24.1 | `rewrite` makes one structured call: the rewriting prompt, which names every corpus author it is given, then the question; it returns the model's query and the author it names as most likely to answer (26.4), both without surrounding whitespace |
| 24.2 | `per_author` makes one structured call for the subject of a comparison (the question, then the authors compared) and returns, for each author in the order given, "What does <author> say about <subject>?"; a question mark or full stop that ends the subject is dropped |
| 24.3 | A blank query, author or subject does not fit the reply schema (so an adapter reports it as a `ProviderError`, as in 18.8), and the JSON schema sent to the model is a plain string, with no regex pattern (18.10) |
| 24.4 | The rewrite's reply schema asks for the author before the query, so the model chooses whose words to use and then writes the query in them (measured 2026-10-21: gold evidence in the top 5 for 7 of 8 questions, against 4 of 8 with the query first; see "The web fallback") |
| 24.5 | The rewriter caps a rewrite at 64 tokens (2026-10-21's: 10 to 25) and a comparison's subject at 32 |

## Component 25: Corrective graph (`aporia_rag.graph`)

The graph is tested with the real retriever, grader, generator and rewriter over in-memory fakes
(`HashingEmbedder`, `InMemoryVectorStore`, `ScriptedChatModel`), so each test states which
passages exist, which ones the scripted model keeps, and what it rewrites. `search` moved here from
the CLI with the scope rules, so the `search` command and the graph share them (covered by 15.x).

| # | Test |
|---|---|
| 25.1 | When the first search finds a relevant passage, the graph answers from the relevant passages and makes no other model call: one verdict per passage, then the answer |
| 25.2 | When no passage of the first search is relevant, the question is rewritten once (the rewriter gets the question and the corpus authors) and, when the rewriter names a corpus author, searched again with the same `k` and author filter; the new passages are graded against the original question, and the answer comes from the relevant ones |
| 25.3 | A passage already graded in the first search is not graded again |
| 25.4 | When the rewritten search finds nothing relevant either, there is no third corpus search (the web is searched instead: 26.5), and every graded passage is left out |
| 25.5 | A survey of all authors is never rewritten, and never searches the web: the best passage of each author, graded against the question, then a survey of the relevant ones or no answer |
| 25.6 | A comparison (two or more authors named) makes one model call for the per-author questions; each author's best passage is found with that author's question, among that author's works only, and graded against it; the survey answers the original question, in the order of the authors |
| 25.7 | The outcome records how long each stage took, summed over both searches: rewriting (only when it ran), retrieval, grading and generation (a scripted clock makes the times exact) |

## The web fallback: component 26

Decided with Aren on 2026-10-21. A question the corpus cannot answer (4 of the 9 loop questions)
used to end with "none" after about 22 s: a rewrite, a second corpus search and its verdicts
(about 6 s), and the search for other authors (about 6 s), all for nothing. Now:

- **When.** On the loop path only, never in a survey or a comparison (their form is one section
  per corpus author). After the rewrite: if the author most likely to answer is outside the
  corpus, the web is searched at once; if not, the corpus is searched again (25.2), and the web
  only when that finds nothing relevant either.
- **Which author, with no extra call.** The rewriter's reply names the author most likely to
  answer, in a field of its own. The measured prompt already asks it to name that author, and the
  Wittgenstein rewrite of 2026-10-20 wrote "author: Ludwig Wittgenstein" into its query. Plain
  code (`authors_named_in`) decides whether that author is in the corpus. A question about a work
  outside the corpus by an author in it (Plato's *Lysis*) still pays for the second corpus
  search: accepted.
- **Live, 2026-10-21.** The model reads the field as "which listed author, if any": "None of the
  listed authors" for Wittgenstein (to the web at once), John Stuart Mill for the veil of
  ignorance (one more corpus search, 2 verdicts, then the web; L.11 records it). That bias is the
  safe one: a wrong "none" would skip an answer the corpus may hold. The second field also changed
  the queries: told to "Name that author" with `query` as the only field, the model had kept
  writing the query (2026-10-20's Cicero query is 2026-10-21's, "Is it ever expedient to do
  injustice?", with more words after it). A reply schema steers the model like its prompt, so what
  the rewrites find was measured again (`data/eval/spike_rewrites_2.py`): 1 of the 3 loop
  questions found evidence in the top 5 (2026-10-20: 3 of 3), so five replies were compared on 8
  in-corpus questions with a rule fixed before the run (`data/eval/spike_rewrite_variants.py`).
  Author first, then query, with the measured prompt: 7 of 8 (3 of 3 loop questions); query
  first: 4 of 8 (1 of 3); "a query of 10 to 20 words": 5 and 6 of 8. Asked for after the query,
  the author also strayed outside the library (Machiavelli for Plato's royal lie), which would
  send a corpus question to the web; asked for first, it was Plato. The one-field reply of
  2026-10-20 ran on until Ollama's 120 s time-out for one question. Author first is built (24.4).
- **Web pages as passages.** Each page (its snippet or opening, at most 2,000 characters) becomes
  a chunk with the site as its author and the page title as its title, so the grader and the
  generator take it unchanged. The web is asked the reader's question, for `k` pages, each graded
  against it. `ask` lists the answer's sources as "Web sources", with their URLs.
- **Failure.** A web search that fails (no network, a rate limit) ends without an answer and says
  why; what was graded before is still listed. The default must keep working offline.
- **Other authors.** `ask` looks for other authors only when the corpus answered.
- **Provider.** `ddgs` (9.16), a free metasearch library that needs no API key (4 packages: ddgs,
  primp, lxml, click). It is the one part of the default setup that leaves the machine, and only
  for a question the corpus cannot answer. It does not go through httpx, so tests replace it
  through the container instead of the transport.

Cost of the outside path: the rewrite (1.3-2.3 s), the web search (to be measured), one verdict
per page (about 1.5 s each) and an answer. It saves the second corpus search's verdicts and the
search for other authors (about 12 s together on 2026-10-20's Wittgenstein question).

## Component 26: Web fallback (`aporia_rag.domain`, `aporia_rag.adapters.web`, `aporia_rag.graph`)

| # | Test |
|---|---|
| 26.1 | `web_passage` makes a chunk of a web page: the site, without "www.", as author; the page title as title; the URL as id, document and source; the page's text without surrounding whitespace, cut at the last word boundary before 2,000 characters when longer |
| 26.2 | The ddgs adapter asks ddgs for `limit` results to the query, from every text engine but Grokipedia (its articles are written by a language model, and an answer cites its pages as sources), and returns them as web passages, in ddgs's order; a result with no text or no web URL is skipped, and one with no title is titled by its URL |
| 26.3 | ddgs finding nothing is an empty list; any other ddgs failure (a rate limit, a time-out, no network) is a `ProviderError` naming ddgs and its reason |
| 26.4 | When the rewriter names an author outside the corpus, the corpus is not searched again: the web is searched with the reader's question (`k` pages), the pages are graded against the question, and the answer comes from the relevant ones; the outcome records the author and that the web was searched |
| 26.5 | When the rewriter names a corpus author (by any name `authors_named_in` recognizes, a surname say), the corpus is searched again (25.2); when that finds nothing relevant either, the web is searched with the reader's question |
| 26.6 | When the web finds nothing relevant, or nothing at all, there is no answer and no other search |
| 26.7 | A web search that fails ends without an answer: the outcome records why, and keeps every passage graded before |
| 26.8 | The outcome records how long the web search took, as a stage of its own |

## Faster answers: component 27

Decided with Aren on 2026-10-21 ("some quality drop is acceptable", keeping the grader and the
rewrites). Grading is the largest stage: 5 verdicts take about 7 s of an answer, and about half of
each verdict is the model reading its prompt (a repeated prompt, served from Ollama's cache, took
2.5-3.5 s for the same 5). Measured on Aren's laptop (`data/eval/spike_grading_speed.py`):

- **Concurrent verdicts: rejected.** 5 calls at once took as long as 5 in a row (7.2 s and 7.0 s,
  each on prompts not yet cached): with 73% of the model on the CPU, the laptop's compute is the
  limit. (That Ollama served all 5 repeated prompts from its cache suggests it already keeps
  several requests' prompts, so more `OLLAMA_NUM_PARALLEL` slots would not help either; they were
  not tried.) The spike's first summary said otherwise: its rotation ran the concurrent calls
  second, on cached prompts, in 2 of every 3 questions.
- **One call for all 5 passages: rejected by its rule.** 3.9 s, but recall 0.84 (rule: 0.85) and
  37 of 245 verdicts changed.
- **Early stop: chosen.** Grading goes in rank order and stops once 2 passages are kept: on the
  saved verdicts of 2026-10-20, 35% fewer verdicts on the 40 retrieval questions (200 to 129),
  no question left unanswered, and answers drawn from fewer passages (62 evidence passages kept
  instead of 108). It never changes whether the loop starts (it stops only after a kept passage).
  A survey or a comparison grades every author's passage: each one is a section.
- **Other authors on request: chosen.** The search for other authors cost 6.5 s (4 verdicts) after
  every answer from the corpus, and on 2026-10-21's Cicero question found nobody. `ask --others`
  asks for it.

## Component 27: Faster answers (`aporia_rag.graph`, `aporia_rag.cli`)

| # | Test |
|---|---|
| 27.1 | Grading follows the order found and stops once 2 passages are kept; the passages after them are not graded, and the outcome lists them as not graded, in the order found; the answer comes from the 2 kept |
| 27.2 | The early stop also holds for web pages |
| 27.3 | A survey grades every author's passage, however many are kept |

## Checking the answer: component 28

The corrective loop decides which passages the model gets; nothing yet checks what the model then
writes. Three errors were read in live answers, each with the right passages in hand:

- **A view given to the wrong author** (2026-10-20): "Plato argues that no ills affecting a
  person's body or property can be worse than those affecting the soul [1]", where [1] is
  Cicero's. Both passages said the soul matters more than the body; the paragraph was about Plato.
- **A reported view given as the author's own** (2026-10-21): "the passage concludes that…" for
  "As their concluding argument, they add…" (Cicero III#118, his opponents speaking; who "they"
  are is in the chunk before).
- **Quotes not word for word** (2026-10-21): "to get themselves out from that miserable condition
  of Warre" for "of getting themselves out…", and "natural Passions" for "naturall Passions".

Two layers, the free one first:

- **Layer 1, plain code, every answer:** every quotation must occur word for word in a passage
  its sentence cites, and a sentence that cites passages and names corpus authors must name the
  author of one of them. It catches the first and third errors, not the second.
- **Layer 2, the chat model, only if measured worth it:** "does passage [n] state this, as its
  author's own view?" for every cited sentence.

**Measured before either is wired into `ask`** (`data/eval/answers_c28.py`): the 49 gold questions
answered by the graph, saved with their passages, layer 1 run on every answer, and 15 answers read
claim by claim. Rules fixed before the run: if layer 1 flags at least 5 of the 49 answers, `ask`
writes the answer once more with the problems listed (one more call, only then); with fewer, it
prints them under the answer. Layer 2 is built only if the reading finds errors layer 1 cannot see
in at least 3 of the 15 answers.

**Measured (2026-10-21, `data/eval/answers-c28.jsonl`).** Layer 1 flagged 13 of 48 answers, and
every one of the 19 problems was real: 9 made-up citation numbers in 5 answers (plus a "[5-6]" it
did not read), 2 quotations from another passage than the one cited, 6 small changes inside
quotations ("use" for "using", "he" for "I"), 2 made-up quotations. So `ask` writes the answer
once more when layer 1 finds problems. Of the 15 answers read, 6 had errors layer 1 cannot see: a
view Aristotle reports and rejects given as his; Schopenhauer's "pendulum", from memory; Hobbes's
"although they love liberty" turned into "because"; a remark of Adeimantus given to Socrates; a
"test" the Kant passage never states; a misread closing sentence of Aristotle's. So layer 2 is
built, measured first on these 15 answers. The made-up numbers have a likely cause, fixed and
measured in the next run: 4 of the 5 answers cited the prompt's example, "[2]", with one passage.

**Measured again with both changes (2026-10-22, `data/eval/answers_c28b.py`).** Rules fixed before
the run: the numbers named in the prompt stay if first drafts have at most 2 made-up citation
numbers; writing again stays if at least half of the rewritten answers end with no problem. First
drafts had no made-up number (10 before), and layer 1 flagged 6 of 48 (13 before), all
quotations: the numbers stay. 3 of the 6 rewritten answers ended with no problem: writing again
stays, by the rule's narrowest margin. Read against their passages, 5 of the 6 flags were real
("shines" for "shine", "I did not even choose" made "not even to choose") and the rewrite fixed 2;
once it made a sentence worse. The sixth was a false alarm, fixed since (28.1): Nietzsche's
"switched off", in double quotation marks in the passage, quoted in single ones inside the
answer's quotation. A rewrite ran for 6 of 48 answers, 6.3 s each (median); the median question
took 11.3 s (10.7 before).

**Layer 2 measured and dropped (2026-10-22, `data/eval/spike_layer2.py`).** One call per answer
(the passages, the answer one numbered sentence per line, "reply with the numbers of the
unsupported sentences") on the 15 answers read. Rule fixed before the run: kept if it names the
wrong sentence in at least 4 of the 6 answers with an error, flags a sentence in at most 2 of the
9 without one, and takes at most 6 s (median). It found 1 of the 6 (the view Aristotle reports
and rejects), flagged 2 of the 9, in 2.5 s: dropped. The errors only reading finds stay a known
limitation, measured by the evaluation.

## Component 28: Citation checks (`aporia_rag.grading.citations`)

| # | Test |
|---|---|
| 28.1 | A quotation (in straight or curly double quotes) that occurs word for word in a passage its sentence cites passes; one that does not is a quote problem naming it. Case, spaces, line breaks, apostrophes and quotation marks inside the quotation (straight or curly, single or double: 'switched off' for "switched off.") and punctuation at the quote's ends do not count |
| 28.2 | A quotation in a sentence that cites no passage must occur in one of the answer's passages |
| 28.3 | A quotation with an ellipsis ("..." or "…") or an editor's change in brackets ("destroy[s] itself") passes when every part of it outside them occurs in the same cited passage |
| 28.4 | A sentence that cites passages and names corpus authors must name the author of a passage it cites: "Plato argues … [1]" with [1] Cicero's is an author problem; "Cicero answers Hobbes [1]" with [1] Cicero's is not |
| 28.5 | Citations are read per sentence, as "[1]", "[1][3]", "[1, 3]" or a range, "[5-6]"; a number with no passage is a citation problem |
| 28.6 | Every problem has a note that tells the model what to fix |
| 28.7 | After an answer (not a survey), the graph runs the checks: with no problem it ends; with problems it asks once for the answer again with the problems' notes (19.5), checks the new answer and ends; the outcome keeps the first draft's problems and those that remain |
| 28.8 | A survey is not checked |
| 28.9 | The revision is timed as its own stage |

## Comparing answers: component 29

The corrective graph adds model calls (a verdict per passage, sometimes a rewrite, a web search
and a second draft) to what plain RAG does: retrieve the top k, then answer from all of them. This
measures what the calls buy, on the same questions, with the same models and the same answer
prompt. Aren's decisions of 2026-10-22: the evaluation is part of the code base, so anyone can run
it again, and faithfulness is measured as well, by reading answers blind.

`evaluate --answers PATH` answers every gold question twice:

- **Plain RAG:** the top 5 passages in the question's scope (the authors it names, as `ask`
  searches), then one answer from all of them, checked by the same citation checks (layer 1) but
  never written again.
- **Corrective RAG:** the graph, as `ask` runs it with its defaults.

Everything is measured with plain code against the gold sets (`--gold` may be given twice, so the
40 retrieval questions and the 9 loop questions run as one set of 49):

- **Evidence**, for the questions the corpus answers: the answers given at least one gold passage,
  and the share of the passages given to the model that are gold. The gold labels are complete
  only for what the pooled model retrieved for the question as asked, so a rewritten search can
  find an answering passage that counts as not gold: the corrective share is a lower bound.
- **Outside questions** (no evidence): the answers from corpus passages, which cannot answer them,
  from web pages, and the questions left without an answer.
- **Checks:** the answers with a layer 1 problem in the answer given, by kind, and how many the
  graph wrote again.
- **Time:** the median seconds per question. Plain RAG runs first, then the graph, each over all
  questions, so neither benefits from the other's prompts in Ollama's cache within a question.

The answers, with their passages marked gold or not, go to PATH (JSON Lines), for the reading that
plain code cannot do. The reading is blind: a script outside the code base gives 15 questions'
two answers in a shuffled order, without the systems' names, and the key is opened after every
error is written down.

The evaluation depends on a `System`, a function from a question to a `Reply` (the answer, how it
was reached, whether its sources are web pages, the kinds of problem the checks found, whether it
was written again), declared next to it. The CLI, the composition root, builds both systems, so
`evaluation` imports neither the graph nor the generator.

## Component 29: Answer evaluation (`aporia_rag.evaluation.answers`, `aporia_rag.cli`)

| # | Test |
|---|---|
| 29.1 | `evaluate_answers` asks every system about every gold question once, system after system in the order given and questions in gold-set order, and keeps each reply with how long it took (a scripted clock makes it exact) |
| 29.2 | A result marks each of its answer's sources as gold when it is a passage hit of the question's evidence; a result without an answer has no sources |
| 29.3 | `summarize` counts the questions, the answers and the routes; for the questions with evidence, the answers given at least one gold passage, the passages given and the gold ones among them |
| 29.4 | `summarize` counts, for the questions without evidence, the answers from the corpus, the answers from the web, and the questions without an answer |
| 29.5 | `summarize` counts the answers with a problem, the problems by kind, the answers written again, and the median seconds per question |
| 29.6 | `write_answers` writes one JSON line per system and question: the system, question id and text, route, seconds, the answer (null when there is none), whether it was written again, its sources (id, citation, text, gold or not) and the kind of each problem; it creates the file's folder |
| 29.7 | `evaluate --answers PATH` prints, after the retrieval report, one column per system: answers, those with a gold passage, the gold share of the passages given, the outside questions answered from the corpus and from the web, the answers with a problem and written again, and the median seconds; then the corrective routes and where the answers went |
| 29.8 | Plain RAG sends one answer request per question, with every passage retrieved in the question's scope (k = 5), and no verdict request; its answer is checked: a citation with no passage is counted |
| 29.9 | The corrective system is the graph: its verdicts, rewrites and web searches happen, and its route, web flag, final problems and second draft are what the reply reports |
| 29.10 | `--gold` given twice evaluates both sets as one; without `--answers`, no answer is requested (23.8) |

## A second entry point: component 30

A web UI is next to the CLI, and it needs the same behaviour: which authors a question covers,
when other authors are looked for, the empty-index check. Those rules were in `cli.py`, where a
second entry point would have to copy them (Aren, 2026-10-22: "it is a behavioural thing, not
really connected to the CLI itself"). They move, unchanged, to the packages that own them, and the
CLI keeps only arguments and printing:

- **Scope** (`retrieval.scope`): `Scope`, `search` (from the graph), `scope_for` (from the CLI's
  `_scope` and `_named_scope`), `resolve_author` and `DEFAULT_K`; the manifest lists its authors.
- **Other authors** (`pipelines.corrective`): asked for with `others=True`; the rule of when they
  are looked for leaves the CLI, and the result is part of the `Outcome`.
- **Plain RAG** (`pipelines.plain`): the evaluation's baseline, a pipeline beside the corrective
  one, returning the same `Outcome`, which now says which route it took. The package `graph` is
  renamed `pipelines`, since it holds two pipelines and only one is built with LangGraph.
- **The empty-index check** (`bootstrap.Container.require_index`).

The evaluation reads a pipeline's `Outcome` through a protocol (`Run`) declared next to it, as it
reads the retriever through `Searcher`, so its own `Reply` type goes.

## Component 30: Shared rules (`aporia_rag.retrieval.scope`, `aporia_rag.pipelines`, `aporia_rag.bootstrap`)

| # | Test |
|---|---|
| 30.1 | `scope_for` with `author` gives that author's scope, the name matched in any case and in part ("hobbes"); a name that matches no author or several is an `UnknownAuthorError` that lists the authors |
| 30.2 | `scope_for` with `all_authors` gives a survey of every author |
| 30.3 | Without either, `scope_for` gives the authors the question names: none is unrestricted, one is that author's scope, several are a comparison; `named` records them |
| 30.4 | `search` takes the k best passages, of the scope's author when it has one, or each surveyed author's best passage |
| 30.5 | The manifest's `authors` are its works' authors, each once, sorted |
| 30.6 | `CorrectiveRAG.answer(..., others=True)` grades the best passage of up to 4 authors with no passage among those graded and keeps the relevant ones in `Outcome.others`, timed as "other authors"; not asked for, or in an author's scope, a survey, or after a web search, it looks for none |
| 30.7 | `Outcome.route` is "normal" when the first search answered, "loop" after a second corpus search, "web" after a web search |
| 30.8 | `PlainRAG.answer` answers from every passage `search` finds (one answer call, no verdict), checks the answer with the citation checks, never writes it again, and reports the route "plain" with the times of retrieval and generation |
| 30.9 | `Container.require_index` raises a `ConfigurationError` naming the command that builds the index when the index is empty, and passes when it is not |
| 30.10 | `evaluate --answers` also counts the outside questions left without an answer |

## The web UI: component 31

A demo page for the portfolio, next to the CLI: a question goes in, the reader watches the
corrective loop work, and the answer comes back with its citations linked to its passages.
Aren's decisions of 2026-10-22: plain HTML, CSS and JavaScript with a FastAPI backend, and the
steps shown live, as they finish.

- **Steps from the pipeline.** `CorrectiveRAG.answer(..., on_step=...)` calls back once per
  finished node, with the node's name, one sentence on what it did ("Graded 3 passages: 2
  relevant"), and its seconds. LangGraph streams each node's update; the pipeline turns it into a
  `Step`, so the CLI, the page and any later client get the same words.
- **The backend** (`aporia_rag.ui`, FastAPI): the page and its files, the corpus's authors, and
  `POST /api/ask`, which answers as a stream of JSON lines: a line per step, then the outcome (or
  an error). A request that cannot be answered (an unknown author, an empty index) is refused
  before anything runs. One question runs at a time: on a laptop the model is the bottleneck, and
  two at once would each take twice as long.
- **The page** (`ui/static`), a conversation (Aren, 2026-10-22: the answer right under the
  question, chat-like): the question box with `ask`'s options at the bottom; for each question, a
  map of the loop whose nodes light up as the steps arrive, the steps with their times, then the
  answer with each citation linked to its passage, the checks and the route; the passages, those
  left out and not graded, and the other authors in a panel beside the conversation, closed until
  a reply's button or a citation opens it. Each question is answered on its own: the page sends
  no earlier question or answer.
- `aporia-rag serve` starts it on this machine (127.0.0.1:8000 by default).

The page's JavaScript has no test runner of its own: it is checked by rendering it against the
backend with fake models and looking at it.

## Component 31: Web UI (`aporia_rag.pipelines`, `aporia_rag.ui`, `aporia_rag.cli`)

| # | Test |
|---|---|
| 31.1 | `answer(..., on_step=...)` reports every node as it finishes, in order, each with a sentence and its seconds: a first search answered gives retrieve, grade, generate, check |
| 31.2 | The loop's steps say what happened: the rewritten query, the second search, an author outside the corpus, the web pages found or the web search's failure, nothing relevant |
| 31.3 | Checks with problems, the answer written again, and other authors are reported as steps too; without `on_step`, nothing changes |
| 31.4 | `GET /` serves the page, and its script and stylesheet are served |
| 31.5 | `GET /api/authors` lists the corpus's authors and the default number of passages |
| 31.6 | `POST /api/ask` streams one JSON line per step, then the outcome: the answer, its sources numbered as cited (citation, text, and the URL of a web page), the route, the rewritten query, the checks, the passages left out and not graded, and the times |
| 31.7 | The request's options are `ask`'s: an author, all authors, the number of passages, and other authors |
| 31.8 | An unknown author or an empty index is refused with status 400 and the reason, before any step; a blank question or a number of passages below 1 is refused with 422 |
| 31.9 | A failure while answering (the model server down) ends the stream with an error line that says what failed; so does a bug, whose traceback is logged |
| 31.10 | `aporia-rag serve` runs the app with uvicorn on the host and port given |
| 31.11 | The checks' labels for each kind of problem are defined once (`grading.citations`) and used by the CLI and the page |

## Architecture rules (`tests/unit/test_architecture.py`, run in the default suite)

The dependency rules of [architecture.md](architecture.md) are checked by
[import-linter](https://import-linter.readthedocs.io), configured in `pyproject.toml`
(`[tool.importlinter]`). It reads every import in the package, so a violation fails the build
instead of waiting for a code review to notice it.

| # | Test |
|---|---|
| A.1 | Layers: `cli` > `ui` > `bootstrap` > `pipelines` > {`adapters`, `config`, `evaluation`, `generation`, `grading`, `indexing`, `ingestion`, `retrieval`} > `domain`; nothing imports a layer above it, and the middle packages do not import each other |
| A.3 | LangGraph is imported only by `graph` |
| A.2 | Provider libraries (`chromadb`, `ollama`, `ddgs`) are imported only by `adapters` (and so are reachable only through `bootstrap`) |
| A.4 | FastAPI, Starlette and uvicorn are imported only by `ui` and by the `cli` that serves it |

## Real-model tests (opt-in: `make test-llm`; need Ollama with the configured embedding and chat models)

| # | Test |
|---|---|
| L.1 | Vectors have the model's size, the same for every call |
| L.2 | The same text embedded as a query and as a passage gives different vectors (the prompt format is applied) |
| L.3 | A question about the good will ranks a Kant passage above a Hobbes passage |
| L.4 | The configured chat model answers a question from three passages: the answer cites at least one of them by a number in range, and contains no thinking (`<think>`) |
| L.5 | The relevance grader, with the configured chat model, keeps the Kant passage and leaves out the Hobbes passage for a question about what is good without qualification |
| L.6 | The configured chat model surveys a Kant and a Mill passage on what makes something morally good: Kant's section speaks of the good will, Mill's of happiness |
| L.7 | Expected to fail (known limitation, 2026-10-20): for "How do Kant and Mill differ on lying?", the relevance grader should keep Kant's passage on truthfulness from duty (one side of the comparison) and leave out Mill's passage on dissent; it leaves out both |
| L.8 | The configured chat model rewrites "Can a dishonourable act ever really be in our interest?" into one short search query: a single line of at most 30 words, not the question itself |
| L.9 | For "How do Kant and Mill differ on lying?", `per_author` returns one question per author, each naming its author and the subject (lying) |
| L.10 | The relevance grader keeps Kant's and Mill's passages on lying, each judged against its own author's question, and leaves out Mill's passage on dissent (the comparison spike of 2026-10-20, 3 of 3) |
| L.11 | The rewriter names no corpus author for the Wittgenstein and veil-of-ignorance loop questions, and a corpus author for the two in-corpus questions that needed the rewrite on 2026-10-20 (the Cicero and Hobbes questions). Expected to fail for the veil of ignorance (known limitation, 2026-10-21: the model names John Stuart Mill; the graph searches the corpus once more, then the web) |
