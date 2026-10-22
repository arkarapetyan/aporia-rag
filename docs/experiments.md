# Experiments

What was measured while building AporiaRAG, and what each measurement decided. Most decisions were
taken with a rule written down before the run ("the batched grader is kept if it is 40% faster and
loses no more than 2% of the evidence"), and the rule was applied as written, including when the
result was a near miss. The scripts of one-off measurements ("spikes") are kept out of the code base;
the measurements that stay useful are commands: `aporia-rag evaluate` with `--grade` and `--answers`.

All runs are on one laptop: an 8 GB NVIDIA GPU, 32 GB of RAM, Ollama, the chat model
`qwen3.5:35b-a3b-q4_K_M` (about 73% of it in RAM) and the embedder `qwen3-embedding:0.6b`.

## The result: plain RAG against corrective RAG (2026-10-22)

`aporia-rag evaluate --answers`: both pipelines answer the 49 gold questions (40 retrieval
questions, 5 loop questions the corpus answers in other words, 4 it cannot answer) with the same
models, scope, retrieval and answer prompt. Plain RAG answers from the top 5 passages; the corrective
pipeline grades them, searches again or on the web when none answers, and writes the answer again
when the citation checks fail.

| | plain RAG | corrective RAG |
|---|---|---|
| questions answered | 49 | 48 |
| answers given at least one gold passage (of 45) | 42 | 42 |
| share of the passages given that are gold | 0.55 | 0.91 |
| outside questions answered from corpus passages (of 4) | 4 | 0 |
| outside questions answered from the web | 0 | 3 |
| answers with a citation-check problem | 15 | 1 |
| answers written a second time | 0 | 12 |
| median seconds per question | 9.5 | 11.9 |

- Almost half of what plain RAG gives the model does not answer the question; the grader removes
  most of it, for 2.4 s more per question (median).
- Plain RAG answers every question outside the corpus from passages that cannot answer it. The
  corrective pipeline answers 3 from the web and says the fourth (Plato's *Lysis*) has no answer.
- The 0.91 is a lower bound: the gold labels are complete only for what the pooled model retrieved
  for the question as asked. Of the 7 passages the pipeline used that the gold set did not mark, 2
  answer the question (Cicero, *De Officiis* III#116 and #127); counted, the share is 0.94. They are
  now in the gold set, with III#24 from an earlier answer.

**Run again** on the refactored code (component 30), with the same questions and gold set: the
first five rows came out the same; answers with a check problem 22 (plain) and 2 (corrective),
second drafts 8, median 9.9 s and 11.4 s. The gap between the two pipelines holds; the counts of
problems and second drafts vary from run to run.

**Blind reading.** What plain code cannot see (a claim its passage does not make) was counted by
reading. For 15 questions fixed before the run (every third of the 49), a script showed each
question's two answers as A and B in a seeded random order, without the systems' names, and only
the passages each answer cites, renumbered, so that the number of passages (plain RAG always 5)
would not name the system. Every error was written down before the key was opened.

| | plain RAG | corrective RAG |
|---|---|---|
| answers with an error | 10 of 15 | 3 of 15 |
| answers with a content error (claim not in its passage, wrong speaker, a reported view given as the author's) | 7 (9 errors) | 3 (3 errors) |
| answers with a quotation not word for word | 3 | 0 |
| answers the question: yes / partly / no | 12 / 3 / 0 | 11 / 3 / 1 |

Plain RAG errs most where it must use passages that do not answer: it put a Mill passage on
tranquillity into a Schopenhauer answer and misstated it, made a Kant passage reject "perfection"
when it rejects empirical principles only, and gave Hume's burden of proof ("it will then be
incumbent on us") to his objectors. The corrective pipeline's one "no" is honest: its passage gives
no test of a maxim, and it says so; plain RAG's answer invented one. Both made the same Aristotle
error, a view he reports and rejects given as his own: the chunk starts in the middle of the
argument. One of the corrective pipeline's errors was written by the second draft, a garbled
"correction" of a quotation that was right.

Limits: one reader (who built both systems), 15 questions, and not fully blind: the reader had seen
how many passages each system used per question. The same code flagged 6 first drafts in the
previous run and 12 in this one: temperature 0 makes each call repeatable, not a whole run.

## Retrieval

**A gold set, completed by pooling (2026-10-14 to 10-16).** 40 questions over all 12 works,
written in a reader's words: a bag-of-words baseline finds an answering passage in its top 5 for 25%
of them. The first bake-off found the right section in the top 5 for 88% of questions but the
quoted passage for 40%. Most of that gap was missing evidence: reading the corpus for every passage
that answers each question added 39 (84 in all, and corrected two), and judging the 808 chunks three
embedding models retrieved (`qwen3-embedding:0.6b`, `embeddinggemma`, `nomic-embed-text`) added 217
more (301). The best model's passage Hit@5 went from 57% to 93%. `qwen3-embedding:0.6b` is the
default.

| | passage Hit@1 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|
| 40 retrieval questions, 2026-10-20 | 0.70 | 0.93 | 0.95 | 0.78 |
| 45 retrieval and loop questions, 2026-10-22 | 0.69 | 0.93 | 0.96 | 0.79 |
| the same, run again | 0.67 | 0.93 | 0.96 | 0.78 |

**Chunk overlap.** The first chunker repeated only whole units of text, and paragraphs are nearly
always longer than the overlap: 1.5% of chunk boundaries overlapped. Overlap that starts at the
last sentence boundary that fits raised that to 89%.

**Queries on the CPU.** On the 8 GB GPU the embedder and the chat model did not both fit, so Ollama
swapped them on every question: retrieval took 7 s and every answer paid 8-11 s to reload the chat
model. Embedding the query on the CPU takes a fraction of a second, and the GPU stays with the chat
model.

## The chat model (2026-10-17)

`qwen3.5:35b-a3b`, a mixture of experts that uses about 3B of its 35B parameters per token, wrote
about 45 tokens/s with 72% of it in RAM: faster than `qwen3.5:9b`, which fits on the GPU but reads
all 9B weights per token, and four times faster than the dense `qwen3:14b`. On four test questions
it was also the only one of the three that kept apart an author's views from those he reports to
reject (Aristotle's "current views" on pleasure, the theory Nietzsche calls "wrong"). Four questions
choose a default, not a winner; the answer evaluation measured this model only.

The first answers set the answer prompt's rules: `qwen3:14b` gave every retrieved passage a
paragraph, presented the theory Nietzsche calls wrong as his, and cited a Hobbes passage for words
not in it.

## The relevance grader (2026-10-20)

`evaluate --grade` on the gold set: of 68 disagreements, read one by one, 21 were grader errors, 28
the grader's leniency by design (its prompt keeps a passage that answers "even in part"), 16
answering passages the gold set lacked (added) and 3 passing mentions. On the corrected labels, at
the top 5: **precision 0.93, recall 0.92**, and every passage retrieved for the 4 outside
questions was rejected.

- **A prompt line, measured and reverted.** The one systematic error was a passage that answers one
  side of a comparison. A prompt line naming that case made the grader more lenient everywhere (25
  more passages kept, 17 of them not evidence; precision 0.89) and still dropped Kant's passage for
  "How do Kant and Mill differ on lying?". Asking each author the question on their own kept both
  sides in 3 of 3 questions; two plain-code framings managed 1 of 3. Comparisons are now asked per
  author.
- **When the loop starts.** "No passage kept in the top 5" starts it for 6 of the 7 gold questions
  that need it, with no needless start; "fewer than 2 kept" would add 11 needless starts to catch
  the seventh.

## The query rewriter (2026-10-20 and 10-21)

Hand-written rewrites in each author's words brought evidence into the top 5 for all 3 retrieval
questions that fail, which bounds what a rewriter can gain; the model's rewrites did the same, at
1.5-1.9 s each. Adding a field for the author most likely to answer (so a question outside the
corpus can skip the second corpus search) changed the queries: a reply schema steers a model as its
prompt does. Re-measured, the rewriter kept evidence in the top 5 for only 1 of the 3 questions,
below its rule, so five reply shapes were measured on 8 questions the corpus answers:

| reply | gold evidence in the top 5 (of 8) |
|---|---|
| A: query, then author | 4 |
| **B: author, then query** | **7** |
| C: query, then author, longer query asked for | 5 |
| D: author, then query, longer query asked for | 6 |
| E: query only | ran to the 120 s time-out on the 8th question (6 of 7 before) |

B was built: the model chooses whose words to search in, then writes in them. A and C named
Machiavelli for Plato's "royal lie"; B named Plato. E's runaway reply led to a cap on every model
reply (a reply cut at its cap fails at once). The author field reads as "which listed author, if
any": the veil of ignorance gets John Stuart Mill (one more corpus search, about 3 s, then the web),
the safe direction for that error.

## Faster answers (2026-10-21)

Grading was most of the wait: 5 verdicts took about 7 s, half of it the model reading its prompt.
Rules fixed before the run: concurrent grading is kept if it takes at most 60% of the time with at
least 98% of the verdicts unchanged; one call for all passages is kept if it takes at most 60% of
the time, keeps precision and recall at 0.85 or above, keeps nothing for the outside questions,
and leaves at most 1 answerable question without a passage.

- Concurrent: 7.2 s against 7.0 s sequential. With most of the model on the CPU, compute is the
  limit. (The first summary had concurrency winning: its rotation ran the concurrent calls second,
  on cached prompts, for 2 of every 3 questions. Order effects need a balanced design.)
- One call: 3.9 s, but recall 0.84 and 37 of 245 verdicts changed.

Neither passed. Chosen instead: grading in the order found stops once 2 passages are kept (35% fewer
verdicts on the saved verdicts of the 40 questions, none left without an answer), and the search for
other authors runs only when asked for. The median normal-path question went to 10.6 s; a Cicero
question that took 32.5 s took 20.0 s.

## Checking the answer (2026-10-21 and 10-22)

**Plain-code checks, measured before they were wired in.** Rules fixed before the run: if the checks
flag at least 5 of the 49 answers, the pipeline writes a flagged answer again; a model check
(layer 2) is built only if reading 15 answers finds errors the checks cannot see in at least 3.
The checks flagged 13 of 48 answers, and all 19 problems were real: 9 made-up citation numbers, 2
quotations from another passage than the one cited, 6 quotations changed in a word, 2 made up.
Reading found 6 answers of 15 with errors plain code cannot see. Both rules fired.

**The made-up numbers had a cause.** 4 of the 5 answers that invented a citation cited [2] from a
single passage: the prompt's example, "like [2]". The prompt now names the numbers to cite. Rule:
the change stays if first drafts have at most 2 made-up numbers. They had 0 (10 before).

**The second draft.** Rule: it stays if at least half of the rewritten answers end with no problem.
3 of 6 did, the narrowest pass. Of the 6 flags, 5 were real and 1 a false alarm (Nietzsche's
"switched off" quoted in single quotation marks inside double ones), fixed since; in one case the
rewrite made the sentence worse. It costs 6.3 s, only for the answers flagged; the median question
went from 10.7 s to 11.3 s.

**The model check, dropped.** One call per answer with the passages and the answer one numbered
sentence per line: "which sentences do the passages not support?" Rule: kept if it finds at least 4
of the 6 known errors, flags at most 2 of 9 good answers, and takes at most 6 s. It found 1, flagged
2, in 2.5 s. The local model does not see what it got wrong in the first place.

## Smaller findings

- **Margin notes.** A scan of the loaded corpus caught 71 margin notes merged into Cicero's paragraphs
  that no test had anticipated; both the unit tests and the corpus contract tests cover them now.
- **Pointers to other authors.** On the first three questions (friendship, pleasure, justice) all 9
  pointers were relevant, and "What is justice?" got Plato back, whom the top 5 had missed.
- **Surveys need structure.** Asked in the prompt for one numbered paragraph per author, the first
  live survey test got two right paragraphs and no number; a structured reply with exactly one
  paragraph per passage fixed it for good.
- **Fakes must refuse what the real thing refuses.** The rewriter's first reply schemas held a regex;
  every test with fakes passed, and Ollama refused every live call. The fake server now refuses
  them too.
