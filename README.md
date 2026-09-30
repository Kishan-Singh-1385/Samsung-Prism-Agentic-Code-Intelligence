# Samsung PRISM - Agentic Code Intelligence

This project retrieves and ranks code snippets for natural-language programming
queries. It uses the CoIR AppsRetrieval benchmark through MTEB and runs on CPU.
It is a retrieval system; it does not generate code or train a model.

## Problem statement

Given a programming question and a large corpus of code snippets, return the
most relevant snippets in ranked order. The benchmark contains thousands of
queries and code documents, some of which are long, so the system embeds the
corpus once and compares each query embedding against the stored document
embeddings.

## Solution overview

The official control embeds the query and each document with the same frozen
Sentence Transformers model, then ranks documents by cosine similarity. The
final candidate adds a small, fixed language cue to the corpus text. It does
not change the model, query text, similarity calculation, or ranking method.

```text
Natural-language programming query
                |
                v
       MiniLM query embedding
                |
                v
       Cosine similarity search <----- cached corpus embeddings
                |                              ^
                v                              |
       Ranked top-K snippets       8,765 corpus documents
```

The AppsRetrieval corpus has blank titles in the observed data. The official
document input is therefore its `text` (Python solution code); in general the
official encoder joins nonempty `title` and `text` fields with a space. The
candidate corpus input is the deterministic string `python code:\n` followed
by that official representation. Queries remain the original query `text`.
URLs and other metadata are not used by either representation.

`src/prism/retrieval/encoder.py` implements MTEB's Sentence Transformers
encoder, `src/prism/retrieval/apps_index.py` loads the AppsRetrieval test data
and provides the local dense index, and scripts in `scripts/` expose evaluation,
prediction export, and demo commands.

## Final model

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Embedding dimension: 384
- Execution: CPU, batch size 16 for encoding
- Candidate: the same model and revision; only the fixed corpus marker differs

The baseline established a CPU-compatible control. In the measured
representation ablation, the `python code:` marker improved both reported
metrics on this benchmark's test split. This is an exploratory measurement,
not evidence of independently validated or general performance. See the
experiment summary below.

## Dataset and evaluation

The task is **CoIR AppsRetrieval**, dataset revision
`f22508f96b7a36c2415181ed8bb76f76e04ae2d5`:

- 8,765 corpus documents
- 3,765 queries
- The available evaluation split is `test`
- No train or development relevance labels are available
- Metrics are NDCG@10 and MRR@10, calculated with MTEB's retrieval metric
  implementation

The verified baseline evaluation ran on CPU. Since the task provides only test
relevance labels, all candidate scores on this task are exploratory test-set
measurements and have not been independently validated.

## Final results

The official baseline is preserved as the control. Variant B is prepared as the
candidate submission system; the original baseline artifacts and command are
unchanged.

| System | NDCG@10 | MRR@10 | Representation |
|---|---:|---:|---|
| Official MiniLM baseline | 0.06596 | 0.05581 | Official `title` + `text`; raw query |
| Final candidate, Variant B | 0.06922 | 0.05813855688357678 | `python code:\n` + official `title`/`text`; raw query |

Variant B's measured absolute change over the baseline is +0.00326 NDCG@10 and
+0.0023285568835767836 MRR@10. Its latest cached validation took 12.13 seconds
with a measured peak working set of 363.63 MiB. The corpus embeddings were
loaded from cache for that run; a prior first-time corpus encoding of this
representation took about 169 seconds on the same CPU environment. These
measurements describe this run only.

Machine-readable details are in
[`artifacts/submission/final_candidate_summary.json`](artifacts/submission/final_candidate_summary.json).

## Experiments

All comparisons below use AppsRetrieval's test split. They are not independent
validation, and no weights or prefixes were selected by per-query relevance
labels.

| Experiment | NDCG@10 | MRR@10 | Outcome |
|---|---:|---:|---|
| MiniLM official baseline | 0.06596 | 0.05581 | Preserved as control |
| MiniLM + `python code:` corpus marker (Variant B) | 0.06922 | 0.05813855688357678 | Candidate; both metrics increased in the exploratory measurement |
| MiniLM + `[LANGUAGE]` / `[CODE]` sections | 0.06917 | 0.057695777735618396 | Tested representation; not selected over the simpler marker |
| Conservative query whitespace normalization | 0.06596 | 0.055810409157022685 | No change from baseline |
| BGE code-search model, raw representation | 0.05659 | 0.048243 | Rejected; both below baseline |
| BGE code-search model, processed representation | 0.05469 | 0.046593 | Rejected; both below baseline |
| BM25 | 0.01528 | 0.012933978372225385 | Rejected; both below baseline |
| RRF, depth 200 / k=60 | 0.06696 | 0.05499040873542862 | Mixed; NDCG increased and MRR decreased |
| RRF, depth 100 / k=10 | 0.06435 | 0.05662450726195743 | Mixed; NDCG decreased and MRR increased |
| Fixed diagnostic scorer | 0.05814 | 0.047741626088239605 | Rejected; below baseline on both metrics; not learned |

No learned reranker was trained. AppsRetrieval lacks train/development qrels,
and test qrels were used only to calculate benchmark metrics and inspect
post-ranking diagnostics.

Experiment configurations and results are in
[`artifacts/experiments/results.json`](artifacts/experiments/results.json) and
[`artifacts/experiments/cheap_ablation_results.json`](artifacts/experiments/cheap_ablation_results.json).
The MTEB baseline score report is
[`artifacts/evaluation/sentence-transformers__all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/AppsRetrieval.json`](artifacts/evaluation/sentence-transformers__all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/AppsRetrieval.json).

## Installation

Use Python 3.10. From a terminal:

```bash
git clone https://github.com/Kishan-Singh-1385/Samsung-Prism-Agentic-Code-Intelligence.git
cd Samsung-Prism-Agentic-Code-Intelligence
python -m venv .venv
```

Windows PowerShell activation and installation:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

The verified environment used Python 3.10.11, MTEB 2.21.10, Sentence
Transformers 6.1.0, Datasets 5.0.1, and NumPy 1.26.4. The relevant versions
are recorded in `requirements.txt`. On first use, the model and dataset must be
downloaded from Hugging Face; network access may be needed. Runtime caches are
local and are not required to be committed.

## Evaluation

Reproduce the official MiniLM baseline from the repository root:

```bash
python scripts/evaluate.py --device cpu --batch-size 16
```

This uses `configs/config.yaml`, including the pinned model and dataset
revisions. MTEB writes the baseline score report under
`artifacts/evaluation/`.

To regenerate and evaluate Variant B's candidate predictions:

```bash
python scripts/validate_variant_b.py --device cpu --batch-size 16
```

The script reuses valid local embeddings when available and builds the frozen
MiniLM vectors when caches are absent. It writes the candidate prediction JSON
and summary described below. This command calculates test metrics; it does not
train or tune the retrieval system.

## Submission artifacts

- **Candidate predictions:**
  [`artifacts/submission/apps_retrieval_test_top10_variant_b.json`](artifacts/submission/apps_retrieval_test_top10_variant_b.json)
  contains 3,765 query IDs, each mapped to the top 10 document IDs and cosine
  similarity scores in the project's MTEB-compatible retrieval-results format.
- **Candidate summary:**
  [`artifacts/submission/final_candidate_summary.json`](artifacts/submission/final_candidate_summary.json)
  records the baseline, candidate, metrics, runtime, memory, and test-only
  limitation.
- **Official baseline predictions:**
  [`artifacts/submission/apps_retrieval_test_top10.json`](artifacts/submission/apps_retrieval_test_top10.json)
  remain available as the control output.
- **Dataset schema:** [`artifacts/dataset_schema.json`](artifacts/dataset_schema.json).

The MTEB JSON under `artifacts/evaluation/` contains evaluation metrics; the
separate submission JSON contains query-level predictions.

## Demo

Run one of the five real benchmark-corpus demo queries:

```bash
python scripts/demo_retrieve.py --demo-query 1
```

Queries 1-5 cover graph traversal/BFS, sorting and binary search, dynamic
programming, input preprocessing, and data-structure manipulation. Run an
arbitrary programming query with:

```bash
python scripts/demo_retrieve.py --query "Use breadth-first search to find shortest path distances in an unweighted graph."
```

The demo uses the official MiniLM index, returns real corpus document IDs and
similarity scores, and displays a short code preview and retrieval latency.
Use `--top-k N` to change the number of displayed results.

## Demo video

Demo video link: TO BE ADDED AFTER UPLOAD

No demo video file or uploaded link was present in the repository when this
README was prepared.

## Presentation

The final project presentation is available here:

[SAMSUNG-PRISM-AGENTIC-CODE-INTELLIGENCE.pptx](presentation/SAMSUNG-PRISM-AGENTIC-CODE-INTELLIGENCE.pptx)

## Team

Team member information was not present in the existing repository.

## Reproducibility and limitations

- Python 3.10.11; CPU execution; batch size 16.
- Frozen model revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Dataset revision: `f22508f96b7a36c2415181ed8bb76f76e04ae2d5`.
- Official baseline command: `python scripts/evaluate.py --device cpu --batch-size 16`.
- Candidate validation command: `python scripts/validate_variant_b.py --device cpu --batch-size 16`.
- Demo command: `python scripts/demo_retrieve.py --demo-query 1`.
- Candidate predictions: `artifacts/submission/apps_retrieval_test_top10_variant_b.json`.
- AppsRetrieval exposes only test relevance labels. The reported candidate
  improvement is exploratory and has not been independently validated.
- Cross-version and evolutionary retrieval have not been implemented.
- The reported metrics are retrieval metrics; the system does not generate
  explanations or code.
