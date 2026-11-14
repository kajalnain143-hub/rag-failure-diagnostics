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
