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
