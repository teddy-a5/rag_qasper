# RAG Optimization on QASPER

Retrieval-first RAG pipeline over the QASPER dataset: a baseline, a diagnosis of its failures, three independent
optimizations, and a final system. Retrieval is evaluated on a development split (80 papers) and run once on a
held-out split (50 papers); answer generation is evaluated on 20 fixed held-out questions.

## Setup

```bash
uv sync                    
```

The first run downloads the QASPER parquet and `BAAI/bge-small-en-v1.5` from Hugging Face. Embedding a corpus
takes about 6 to 9 minutes on a CPU the first time per chunking setup; later runs read `cache/`.

## Running

```bash
uv run scripts/run.py --split dev     --config baseline --cold-queries   # also: A, B, C, final, final_nohdr
uv run scripts/run.py --split heldout --config baseline --cold-queries
uv run scripts/run.py --split heldout --config final    --cold-queries
uv run scripts/generate.py --config baseline
uv run scripts/generate.py --config final
```

`--cold-queries` embeds each query fresh so the reported latency includes query embedding.
`run.py` rewrites a results file and resets its answers, so run `generate.py` again after re-running a held-out config
(responses are cached, so this costs nothing).

## Setup details and metric interpretation

- Embeddings: `BAAI/bge-small-en-v1.5`, normalized, brute-force cosine in NumPy. Queries carry the BGE retrieval instruction prefix; documents do not.
- Generator: `gemini-3.5-flash-lite`, temperature 0, one prompt shared by baseline and final.
- Search covers **all** papers of the split. `target_paper_id` and the target title are never used in retrieval.
- Gold evidence: QASPER text evidence matched to paragraph IDs by exact text, union over annotators. Figure/table-only evidence is not matched.
- A gold paragraph is found when the union of `source_spans` of retrieved chunks **from the target paper** covers at least 80% of its whitespace tokens
  (a token counts only if fully inside a span). Recall@k is found gold paragraphs over gold paragraphs.
  MRR@10 is 1/r for the smallest r where the top r chunks first find any gold paragraph, else 0.
- Scored questions: dev 261 of 295, held-out 155 of 182. The rest have no matched text evidence
  (dev: 19 unanswerable plus TODO: reason for the remaining ones, see the notebook or `check_unmatched`).
- Output `text` is always the original paragraph text; `source_spans` are zero-based offsets into the original paragraph.
  Headings in B are added only to the text that is embedded and indexed.

## Optimizations

| | Stage | What it does |
|---|---|---|
| Baseline | | 800-character windows, 200 overlap, over each paper's concatenated paragraphs; dense top 10 |
| **A** | Ingestion | One chunk per paragraph; paragraphs over 1,100 characters are split at sentence boundaries (no overlap) |
| **B** | Ingestion | Contextual headings: `Title:` and `Section:` prepended to the embedded and indexed text |
| **C** | Retrieval | Own BM25 (k1 = 1.5, b = 0.75, Lucene-style IDF, QASPER citation placeholders dropped) fused with dense by RRF (k = 60, pool 100) |
| **Final** | | A + B + C |

## Development results (80 papers, 261 scored questions)

| Configuration | R@5 | R@10 | MRR@10 | Retrieval latency (median ms) | LLM calls/q |
|---|---|---|---|---|---|
| Baseline | 0.142 | 0.172 | 0.112 | 28.9 | 0 |
| Baseline + A | 0.159 | 0.224 | 0.153 | 28.4 | 0 |
| Baseline + B | 0.143 | 0.207 | 0.109 | 27.2 | 0 |
| Baseline + C | 0.138 | 0.224 | 0.119 | 40.2 | 0 |
| A + C (extra row) | 0.178 | 0.246 | 0.175 | 42.9 | 0 |
| **Final (A + B + C)** | **0.206** | **0.286** | **0.196** | 48.0 | 0 |

Paired change in R@10 against baseline (bootstrap 95% CI, seed 42): A +0.051 [+0.007, +0.095]; B +0.035 [-0.001, +0.073] (not distinguishable from noise);
C +0.051 [+0.018, +0.084]; final +0.113 [+0.065, +0.162]. The final configuration was chosen between two dev candidates (with and without headings).

## Held-out results (50 papers, 155 scored questions; run once, nothing tuned on them)

| System | R@5 | R@10 | MRR@10 | Retrieval latency (median ms) |
|---|---|---|---|---|
| Baseline | 0.129 | 0.197 | 0.126 | 40.7 |
| Final | 0.177 | 0.284 | 0.182 | 52.7 |

### Answer generation (20 fixed held-out questions)

| System | Answer F1 | Abstained | False-answer rate | Median latency, retrieval + generation (ms) | LLM calls/q |
|---|---|---|---|---|---|
| Baseline | 0.276 | 10 / 20 | 0.0 | 1799 | 1 |
| Final | 0.177 | 11 / 20 | 0.0 | 1821 | 1 |

The sample contains **one** gold-unanswerable question, so the false-answer rate is exploratory only.
The other 162 questions are retrieval-only (latency in the table above). The generation prompt was not tuned on held-out results.

## Reflection (draft: rewrite in your own words)

- **What worked.** Paragraph chunks (A) fixed boundary splits. Hybrid retrieval (C) raised recall@10 but not the top ranks.
  Headings (B) were indistinguishable from noise alone, but added +0.04 R@10 on top of A + C (0.246 to 0.286); one possible reason is that
  the title also reaches BM25, which I have not isolated.
- **What did not.** B alone; C's MRR and R@5 were flat. On the held-out 20, Answer F1 fell from 0.276 to 0.177 while retrieval improved.
  That sample is too small to conclude anything (one gold-unanswerable question, about half of answers abstained in both systems).
- **Limits.** Many QASPER questions are generic, so many failures cannot be fixed by retrieval alone.
- **Cost.** No LLM calls in retrieval. Latency rose from about 29 ms to about 48 ms, and a new index takes minutes to embed on CPU.
  Generation costs one call (about 1.8 s) per selected question.

## Why each optimization works, and when it can hurt

- **A.** Gold evidence is paragraph-level, so a whole-paragraph chunk satisfies the 80% rule once retrieved. It can hurt for very long paragraphs
  (pieces compete) or very short ones (little context to embed).
- **B.** Chunk text alone often lacks the paper's identity; title and section make chunks distinguishable across 80 papers.
  It can hurt by adding shared boilerplate that dilutes each chunk's own content in a small embedding model.
- **C.** Dense vectors capture paraphrase; BM25 captures exact rare terms such as model and dataset names. RRF fuses ranks, so score scales do not matter.
  It can hurt when one retriever is much weaker, since fusion can pull good dense hits down.