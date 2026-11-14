"""Historical batch benchmark report generation."""
from __future__ import annotations

REPORT_METRICS = (
    "recall_at_k", "mrr", "ndcg_at_k", "evidence_term_coverage",
    "latency_avg_ms", "latency_p50_ms", "latency_p95_ms",
)


def build_benchmark_report(runs: list[dict], dataset_name: str, strategy: str) -> dict:
    """Build ordered metric history and first-to-latest trend deltas."""
    selected = [run for run in runs
                if run.get("config", {}).get("dataset_name") == dataset_name
                and run.get("config", {}).get("strategy") == strategy
                and all(metric in run.get("metrics", {}) for metric in REPORT_METRICS)]
    selected.sort(key=lambda run: (str(run.get("created_at", "")), int(run["id"])))
    history = []
    for run in selected:
        history.append({
            "run_id": run["id"], "created_at": run.get("created_at"), "name": run.get("name"),
            "metrics": {key: round(float(run["metrics"][key]), 4) for key in REPORT_METRICS},
            "case_count": run["config"].get("case_count"),
            "mmr_diversity": run["config"].get("mmr_diversity"),
            "snapshot_id": run["config"].get("snapshot_id"),
            "corpus_fingerprint": run["config"].get("corpus_fingerprint"),
            "dataset_fingerprint": run["config"].get("dataset_fingerprint"),
        })
    trends = {}
    for metric in REPORT_METRICS:
        values = [item["metrics"][metric] for item in history]
        trends[metric] = {"first": values[0] if values else None,
                          "latest": values[-1] if values else None,
                          "delta": round(values[-1] - values[0], 4) if len(values) > 1 else 0.0}
    return {"dataset_name": dataset_name, "strategy": strategy, "run_count": len(history),
            "history": history, "trends": trends,
            "note": "Each run references a stored corpus and labeled-dataset snapshot. Replay a snapshot ID to reproduce benchmark inputs; provider versions and timings may still vary."}


INSIGHT_STRATEGIES = ("tfidf", "hybrid", "hybrid_mmr")
QUALITY_KEYS = ("recall_at_k", "mrr", "ndcg_at_k", "evidence_term_coverage")


def build_benchmark_insights(runs: list[dict], dataset_name: str) -> dict:
    """Summarize the latest comparable run for each strategy and make metric-based suggestions."""
    selected = [run for run in runs
                if run.get("config", {}).get("dataset_name") == dataset_name
                and run.get("config", {}).get("strategy") in INSIGHT_STRATEGIES
                and all(key in run.get("metrics", {}) for key in (*QUALITY_KEYS, "latency_p95_ms"))
                and run.get("config", {}).get("corpus_fingerprint")
                and run.get("config", {}).get("dataset_fingerprint")]
    selected.sort(key=lambda run: (str(run.get("created_at", "")), int(run["id"])))
    latest = {}
    for run in selected:
        latest[run["config"]["strategy"]] = run
    if not latest:
        return {"dataset_name": dataset_name, "strategy_count": 0, "strategies": [],
                "best_strategy": None, "recommendations": [],
                "note": "No fingerprinted batch benchmark runs with complete quality and latency metrics were found."}

    # Only compare strategies evaluated against the same immutable benchmark inputs.
    fingerprints = {(run["config"]["corpus_fingerprint"], run["config"]["dataset_fingerprint"],
                     run["config"].get("case_count")) for run in latest.values()}
    comparable = len(fingerprints) == 1
    rows = []
    for strategy, run in latest.items():
        metrics = {key: round(float(run["metrics"][key]), 4)
                   for key in (*QUALITY_KEYS, "latency_p95_ms")}
        rows.append({
            "strategy": strategy, "run_id": run["id"],
            "snapshot_id": run["config"].get("snapshot_id"),
            "case_count": run["config"].get("case_count"),
            "metrics": metrics,
            "quality_score": round(sum(metrics[key] for key in QUALITY_KEYS) / len(QUALITY_KEYS), 4),
        })
    rows.sort(key=lambda row: row["quality_score"], reverse=True)
    best = rows[0]["strategy"] if comparable else None
    recommendations = []
    if comparable and len(rows) > 1:
        recommendations.append(
            f"{best.replace('_', ' ').upper()} has the highest mean of Recall@K, MRR, nDCG@K, and lexical evidence coverage on this snapshot."
        )
        fastest = min(rows, key=lambda row: row["metrics"]["latency_p95_ms"])
        if fastest["strategy"] != best:
            recommendations.append(
                f"{fastest['strategy'].replace('_', ' ').upper()} has the lowest p95 latency; choose it when response speed matters more than the quality-score lead."
            )
        weakest_metric = min(QUALITY_KEYS, key=lambda key: sum(row["metrics"][key] for row in rows) / len(rows))
        recommendations.append(
            f"Review {weakest_metric.replace('_', ' ')} across strategies; it has the lowest cross-strategy average."
        )
    elif not comparable:
        recommendations.append("Latest strategy runs use different snapshots or case counts; rerun all strategies on one saved snapshot before comparing them.")
    else:
        recommendations.append("Run the same dataset snapshot with at least two retrieval strategies to enable a meaningful comparison.")
    return {
        "dataset_name": dataset_name, "strategy_count": len(rows),
        "comparable": comparable, "best_strategy": best,
        "strategies": rows, "recommendations": recommendations,
        "note": "Quality score is the unweighted mean of four retrieval metrics. Evidence coverage is lexical, not semantic faithfulness. Comparisons require matching corpus and dataset fingerprints plus case count.",
    }
