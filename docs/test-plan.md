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
| 1.18 | `ChatMessage` has the role `system` or `user` and non-empty content | A role no adapter can map must fail here, not at the provider. (`assistant` arrives with the first multi-turn caller.) |

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
| 14.10 | The container's generator and relevance grader share its chat model, built once and only when first used (so `search` never needs one), and closed on leaving the container |

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

## Component 19: Generator (`aporia_rag.generation.generator`)

| # | Test |
|---|---|
| 19.1 | The model receives the system prompt, then one user message: the passages numbered from 1 in the order given, each headed by its citation and followed by its exact text, and the question last |
| 19.2 | The answer is the model's reply, with the passages as its sources, in the order of their numbers |
| 19.3 | `survey` sends the same user message with the survey prompt as a structured request for exactly one paragraph per passage, and returns one section per passage, headed by the passage's author and number (`Immanuel Kant [2]`), with the passages as its sources |

## Component 20: `ask` command (`aporia_rag.cli`)

| # | Test |
|---|---|
| 20.1 | `ask` prints the model's answer, then the passages it was given, numbered as the model saw them, with citation and chunk id |
| 20.2 | `ask -k N` gives the model N passages, and `ask --author` restricts them to one author, as in `search` |
| 20.3 | `ask` on an empty index says to run `aporia-rag index` and never calls the chat model |
| 20.4 | A chat model that is not pulled is a one-line error naming `ollama pull <model>` |
| 20.5 | `ask` releases its connections, the chat model's included |
| 20.6 | `ask` ends with how long the answer took, split into retrieval, grading, generation and the search for other authors (a scripted clock makes the numbers exact) |
| 20.7 | `ask` grades every retrieved passage: the model answers from the relevant ones only, numbered from 1, and the passages left out are listed as not relevant |
| 20.8 | When no passage is relevant, `ask` says so and never asks the model for an answer |
| 20.9 | After the answer, `ask` grades the best passage of the other authors (those with no passage among the retrieved ones, at most 4) and lists the relevant ones as other authors who discuss the question |
| 20.10 | With `--author`, `ask` does not look for other authors |
| 20.11 | `ask --all-authors` grades the best passage of every author, prints the survey of the relevant ones, lists the others as not relevant, and does not look for other authors (the question is embedded once) |
| 20.12 | `ask` with one author named in the question answers from that author's passages only and does not look for other authors |
| 20.13 | `ask` with two or more authors named in the question surveys them: the best passage of each, one section per author |

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

## Architecture rules (`tests/unit/test_architecture.py`, run in the default suite)

The dependency rules of [architecture.md](architecture.md) are checked by
[import-linter](https://import-linter.readthedocs.io), configured in `pyproject.toml`
(`[tool.importlinter]`). It reads every import in the package, so a violation fails the build
instead of waiting for a code review to notice it.

| # | Test |
|---|---|
| A.1 | Layers: `cli` > `bootstrap` > {`adapters`, `config`, `evaluation`, `generation`, `grading`, `indexing`, `ingestion`, `retrieval`} > `domain`; nothing imports a layer above it, and the middle packages do not import each other |
| A.2 | Provider libraries (`chromadb`, `ollama`) are imported only by `adapters` (and so are reachable only through `bootstrap`) |

## Real-model tests (opt-in: `make test-llm`; need Ollama with the configured embedding and chat models)

| # | Test |
|---|---|
| L.1 | Vectors have the model's size, the same for every call |
| L.2 | The same text embedded as a query and as a passage gives different vectors (the prompt format is applied) |
| L.3 | A question about the good will ranks a Kant passage above a Hobbes passage |
| L.4 | The configured chat model answers a question from three passages: the answer cites at least one of them by a number in range, and contains no thinking (`<think>`) |
| L.5 | The relevance grader, with the configured chat model, keeps the Kant passage and leaves out the Hobbes passage for a question about what is good without qualification |
| L.6 | The configured chat model surveys a Kant and a Mill passage on what makes something morally good: Kant's section speaks of the good will, Mill's of happiness |
