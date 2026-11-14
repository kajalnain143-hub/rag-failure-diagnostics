"""Quality regression checks for comparable saved benchmark runs."""
from __future__ import annotations

QUALITY_METRICS = ("recall_at_k", "mrr", "ndcg_at_k", "evidence_term_coverage")


def compare_benchmark_runs(baseline: dict, current: dict, max_metric_drop: float = 0.05) -> dict:
    """Compare matching dataset/strategy runs and flag metric drops beyond tolerance."""
    baseline_config = baseline.get("config", {})
    current_config = current.get("config", {})
    for key in ("dataset_name", "strategy", "case_count"):
        if baseline_config.get(key) != current_config.get(key):
            raise ValueError(f"Runs are not comparable: config field '{key}' differs.")
    baseline_metrics = baseline.get("metrics", {})
    current_metrics = current.get("metrics", {})
    comparisons = {}
    regressions = []
    for metric in QUALITY_METRICS:
        if metric not in baseline_metrics or metric not in current_metrics:
            raise ValueError(f"Both runs must contain '{metric}'.")
        before, after = float(baseline_metrics[metric]), float(current_metrics[metric])
        delta = round(after - before, 4)
        regressed = delta < -max_metric_drop
        comparisons[metric] = {
            "baseline": round(before, 4),
            "current": round(after, 4),
            "delta": delta,
            "allowed_drop": max_metric_drop,
            "regression": regressed,
        }
        if regressed:
            regressions.append(metric)
    return {
        "status": "regression" if regressions else "passed",
        "baseline_run_id": baseline["id"],
        "current_run_id": current["id"],
        "dataset_name": baseline_config["dataset_name"],
        "strategy": baseline_config["strategy"],
        "case_count": baseline_config["case_count"],
        "max_metric_drop": max_metric_drop,
        "regressed_metrics": regressions,
        "comparisons": comparisons,
        "note": "A metric is flagged only when its absolute drop is greater than the configured tolerance. This check assumes identical dataset labels and corpus.",
    }
