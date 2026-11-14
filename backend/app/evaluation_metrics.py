"""Deterministic retrieval and answer-evidence metrics for repeatable RAG evaluation.

These metrics measure ranking and lexical evidence overlap. They are not substitutes
for human review or a calibrated semantic/faithfulness judge.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable


def _terms(text: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z0-9]{2,}\b", (text or "").lower()))


def recall_at_k(retrieved_ids: list[int | str], relevant_ids: Iterable[int | str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(retrieved_ids[:max(0, k)]) & relevant) / len(relevant)


def reciprocal_rank(retrieved_ids: list[int | str], relevant_ids: Iterable[int | str]) -> float:
    relevant = set(relevant_ids)
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[int | str], relevant_ids: Iterable[int | str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    dcg = sum((1.0 / math.log2(rank + 1)) for rank, item_id in
              enumerate(retrieved_ids[:max(0, k)], start=1) if item_id in relevant)
    ideal_count = min(len(relevant), max(0, k))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0


def evidence_term_coverage(reference_answer: str, retrieved_contexts: Iterable[str]) -> float:
    expected = _terms(reference_answer)
    if not expected:
        return 0.0
    observed = set()
    for context in retrieved_contexts:
        observed.update(_terms(context))
    return len(expected & observed) / len(expected)


def evaluate_case(case: dict, retrieved: list[dict], k: int | None = None) -> dict:
    """Evaluate one labeled query against retrieved chunks.

    Each case requires query_id, question, relevant_chunk_ids, and reference_answer.
    Retrieved chunks must expose chunk_id and content.
    """
    cutoff = k or int(case.get("k", 5))
    ids = [item["chunk_id"] for item in retrieved]
    relevant = case.get("relevant_chunk_ids", [])
    coverage = evidence_term_coverage(
        case.get("reference_answer", ""), [item.get("content", "") for item in retrieved[:cutoff]]
    )
    return {
        "query_id": case["query_id"],
        "recall_at_k": round(recall_at_k(ids, relevant, cutoff), 4),
        "mrr": round(reciprocal_rank(ids, relevant), 4),
        "ndcg_at_k": round(ndcg_at_k(ids, relevant, cutoff), 4),
        "evidence_term_coverage": round(coverage, 4),
        "retrieved_count": min(len(retrieved), cutoff),
        "metric_note": "Evidence term coverage is lexical overlap, not semantic faithfulness.",
    }


def aggregate_metrics(results: list[dict]) -> dict:
    keys = ("recall_at_k", "mrr", "ndcg_at_k", "evidence_term_coverage")
    if not results:
        return {"case_count": 0, **{key: 0.0 for key in keys}}
    return {
        "case_count": len(results),
        **{key: round(sum(row[key] for row in results) / len(results), 4) for key in keys},
    }
