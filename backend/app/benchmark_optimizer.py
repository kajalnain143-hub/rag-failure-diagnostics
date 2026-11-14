"""Small, bounded MMR diversity sweep helpers."""
from __future__ import annotations

QUALITY_KEYS = ("recall_at_k", "mrr", "ndcg_at_k", "evidence_term_coverage")


def rank_mmr_sweep(results: list[dict]) -> dict:
    """Rank MMR sweep runs by unweighted retrieval quality, then p95 latency."""
    rows = []
    for result in results:
        metrics = result.get("metrics", {})
        if any(key not in metrics for key in (*QUALITY_KEYS, "latency_p95_ms")):
            continue
        score = sum(float(metrics[key]) for key in QUALITY_KEYS) / len(QUALITY_KEYS)
        rows.append({
            "mmr_diversity": result["mmr_diversity"],
            "experiment_id": result["experiment_id"],
            "snapshot_id": result.get("snapshot_id"),
            "quality_score": round(score, 4),
            "metrics": {key: round(float(metrics[key]), 4)
                        for key in (*QUALITY_KEYS, "latency_p95_ms")},
        })
    rows.sort(key=lambda row: (-row["quality_score"], row["metrics"]["latency_p95_ms"]))
    return {
        "runs": rows,
        "best_mmr_diversity": rows[0]["mmr_diversity"] if rows else None,
        "best_experiment_id": rows[0]["experiment_id"] if rows else None,
        "selection_rule": "Highest unweighted mean of Recall@K, MRR, nDCG@K and lexical evidence coverage; p95 latency breaks ties.",
        "note": "This is a bounded offline sweep on manually labeled data, not a globally optimal setting. Use identical corpus and labels; lexical evidence coverage is not semantic faithfulness.",
    }
