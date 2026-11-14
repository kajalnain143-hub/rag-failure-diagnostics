# Metric definitions

- **Recall@K**: fraction of labeled relevant chunks found in the top K retrieved chunks.
- **MRR**: reciprocal rank of the first relevant chunk; zero when none is retrieved.
- **nDCG@K**: ranking quality with logarithmic position discount for binary relevance labels.
- **Evidence term coverage**: fraction of unique reference-answer terms appearing in retrieved context.

The first three require human-labeled relevant chunk IDs. Evidence term coverage is lexical
and can be inflated by common words or paraphrase mismatch. None of these metrics alone
measures answer faithfulness, truth, or user satisfaction. Report corpus, query set, K,
retrieval configuration, and date alongside any benchmark numbers.
