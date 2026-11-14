# Starter RAG benchmark

This small labeled file is a **schema/example dataset**, not a measured benchmark result.
Replace the example chunk IDs with IDs from a controlled fixture corpus before reporting
scores. Each case contains:

- `query_id`: stable case identifier
- `question`: user query
- `relevant_chunk_ids`: one or more gold relevant chunk IDs
- `reference_answer`: concise reference answer for lexical evidence coverage
- `k`: ranking cutoff

The deterministic evaluator reports Recall@K, MRR, nDCG@K, and lexical evidence-term
coverage. It does not claim semantic correctness or answer faithfulness.

## Building a benchmark from your own uploaded documents

1. Upload a small, representative set of documents through the dashboard.
2. Call authenticated `GET /api/benchmark/corpus` to retrieve actual indexed chunk IDs and their text.
3. Write questions whose answers are supported by the corpus. For each question, manually identify the relevant chunk IDs and write a reference answer.
4. Replace the placeholder IDs in your dataset with the actual integer IDs returned by the corpus endpoint.
5. Run `POST /api/benchmark/run` with the labeled cases and compare the strategies.

The corpus endpoint returns at most 500 chunks to keep responses bounded. Chunk IDs can change when documents are re-uploaded or the database is recreated, so create fresh labels after rebuilding the knowledge base. Do not use the starter example IDs as if they were real corpus IDs.

## Validate a labeled dataset before running it

After saving the response from `GET /api/benchmark/corpus` as `corpus.json`, validate your
hand-labeled dataset before submitting it to the benchmark API:

```bash
cd backend
python scripts/validate_benchmark_dataset.py --corpus corpus.json --dataset my_dataset.json
```

The validator checks required text fields, unique query IDs, non-empty integer gold
chunk IDs, duplicate labels, cutoff values, and whether every gold chunk ID exists in
the supplied corpus export. It exits with a non-zero status when it finds an issue.
This catches stale IDs after a corpus rebuild, but it cannot decide whether a human
relevance label or reference answer is factually correct.

## Retrieval strategy comparison

The API also exposes `POST /api/benchmark/compare`. Submit the same labeled case
to compare `tfidf`, `hybrid`, and `hybrid_mmr` on the currently indexed corpus.
The response includes per-strategy rankings and metrics plus a best-for-this-query
indicator (nDCG@K, then MRR, then Recall@K). This is a per-query result, not a
global claim about which retriever is universally best.

MMR uses cosine similarity between stored embeddings when all indexed chunks have
compatible vectors. If embeddings are unavailable, it falls back to TF-IDF chunk
similarity, so the comparison remains usable without a Gemini API key. The default
diversity weight is 0.35; larger values penalize redundancy more strongly.

## Answer grounding evaluation

The API exposes `POST /api/quality/evaluate` for model-assisted answer evaluation.
Send `question`, `answer`, and `retrieved_chunk_ids` (real IDs from the current
knowledge base). Optionally include `relevant_chunk_ids` to calculate context
precision and context recall from manually labeled evidence.

Gemini inspects factual claims against only the retrieved chunks and returns a
faithfulness estimate plus supported and unsupported claims. This is an
LLM-as-a-judge signal: it can make mistakes and must not be presented as a
human-verified hallucination verdict. The context precision/recall values are
only as reliable as the supplied relevance labels.


## Batch dataset benchmarking

Use `POST /api/benchmark/run` to execute 1–100 labeled cases against selected
strategies (`tfidf`, `hybrid`, `hybrid_mmr`). The request contains a
`dataset_name`, a `cases` array using the schema above, and an optional
`strategies` array and `mmr_diversity` value. Each strategy returns per-query
retrieved chunk IDs, Recall@K, MRR, nDCG@K, lexical evidence coverage, and measured
embedding/retrieval/total latency. Aggregate results include mean ranking metrics
and average, p50, and p95 latency. One experiment-history record is saved per
strategy.

All cases must use real integer chunk IDs from the currently indexed corpus.
The benchmark does not create an isolated corpus snapshot; re-running after
changing indexed documents may produce different results. Compare runs only when
the corpus and labels are held constant. Timing is an in-process measurement,
not a production load test or provider-cost estimate.


## Benchmark regression gates

Use `POST /api/benchmark/regression` with `baseline_run_id`,
`current_run_id`, and optional `max_metric_drop` (default 0.05). The endpoint
compares Recall@K, MRR, nDCG@K, and lexical evidence-term coverage. It returns
`passed` or `regression`, per-metric deltas, and the metrics that exceeded the
allowed absolute drop.

Only saved batch runs with the same dataset name, strategy, and case count are
comparable. Keep the indexed corpus and relevance labels unchanged as well; the
API cannot independently verify corpus identity. This is a quality regression
gate, not a claim that the benchmark labels are correct or that semantic answer
faithfulness has been proven. The endpoint looks up runs among the latest 100
saved experiments.


## Historical reports and CSV export

Use `GET /api/benchmark/report?dataset_name=...&strategy=...` to retrieve
the saved batch-run history and first-to-latest trends for ranking quality and
latency metrics. Download the same history with
`GET /api/benchmark/report.csv?dataset_name=...&strategy=...`.

Reports include up to the latest 100 saved experiment records available from
the experiment store. Runs are filtered to the exact dataset and strategy.
Interpret trends only when the indexed corpus, relevance labels, and benchmark
settings were held constant; corpus identity is not currently fingerprinted.
