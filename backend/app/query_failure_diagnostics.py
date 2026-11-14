"""Explainable, metric-driven diagnostics for one labeled benchmark query."""
from __future__ import annotations


def diagnose_query_failure(metrics: dict) -> dict:
    """Classify retrieval symptoms from one manually labeled query result.

    These are deterministic hypotheses from ranking/lexical metrics, not proof
    of a root cause. The caller must supply metrics computed against gold labels.
    """
    recall = float(metrics.get("recall_at_k", 0.0))
    mrr = float(metrics.get("mrr", 0.0))
    ndcg = float(metrics.get("ndcg_at_k", 0.0))
    coverage = float(metrics.get("evidence_term_coverage", 0.0))
    signals = []

    if recall == 0:
        signals.append({
            "pattern_id": "P03",
            "severity": "high",
            "classification": "retrieval_miss",
            "evidence": "No manually labeled relevant chunk appeared in the retrieved top-K results.",
            "recommendations": [
                "Inspect query wording, document extraction, chunk boundaries, and embedding availability.",
                "Compare TF-IDF, hybrid, and hybrid+MMR on the same corpus snapshot.",
                "Verify that the gold chunk labels are correct and current.",
            ],
        })
    elif mrr < 0.5 or ndcg < 0.5:
        signals.append({
            "pattern_id": "P03",
            "severity": "medium",
            "classification": "ranking_quality",
            "evidence": f"Relevant evidence was retrieved, but ranking is weak (MRR={mrr:.3f}, nDCG@K={ndcg:.3f}).",
            "recommendations": [
                "Inspect the rank of each relevant chunk and compare retrieval strategies.",
                "Tune top-K and MMR diversity using the same labeled dataset.",
                "Check whether broad or repetitive chunks outrank precise evidence.",
            ],
        })

    if coverage < 0.4:
        signals.append({
            "pattern_id": "P02",
            "severity": "medium",
            "classification": "evidence_coverage",
            "evidence": f"Lexical evidence-term coverage is low ({coverage:.3f}).",
            "recommendations": [
                "Inspect PDF/text extraction for missing or malformed content.",
                "Review chunk size and overlap; relevant facts may be split across chunks.",
                "Check whether the reference answer uses terms absent from the source wording.",
            ],
        })

    if not signals:
        signals.append({
            "pattern_id": None,
            "severity": "info",
            "classification": "no_threshold_failure",
            "evidence": "This query did not cross the current retrieval or lexical-coverage alert thresholds.",
            "recommendations": [
                "Review the retrieved text manually; good ranking metrics do not prove answer faithfulness.",
            ],
        })

    return {
        "classification": "multiple_signals" if len(signals) > 1 else signals[0]["classification"],
        "signals": signals,
        "metrics": {
            "recall_at_k": round(recall, 4),
            "mrr": round(mrr, 4),
            "ndcg_at_k": round(ndcg, 4),
            "evidence_term_coverage": round(coverage, 4),
        },
        "note": "Rule-based hypotheses derived from manually labeled ranking metrics; verify source content and labels before changing the retrieval pipeline.",
    }
