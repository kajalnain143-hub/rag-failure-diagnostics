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
