"""Quality regression checks for comparable saved benchmark runs."""
from __future__ import annotations

import math

QUALITY_METRICS = ("recall_at_k", "mrr", "ndcg_at_k", "evidence_term_coverage")


def compare_benchmark_runs(baseline: dict, current: dict, max_metric_drop: float = 0.05) -> dict:
    """Compare matching dataset/strategy runs and flag metric drops beyond tolerance."""
    try:
        tolerance = float(max_metric_drop)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_metric_drop must be a finite number between 0 and 1.") from exc
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("max_metric_drop must be a finite number between 0 and 1.")

    baseline_config = baseline.get("config", {})
    current_config = current.get("config", {})
    for key in ("dataset_name", "strategy", "case_count"):
        if baseline_config.get(key) != current_config.get(key):
            raise ValueError(f"Runs are not comparable: config field '{key}' differs.")
    for key in ("corpus_fingerprint", "dataset_fingerprint"):
        if baseline_config.get(key) != current_config.get(key):
            raise ValueError(f"Runs are not comparable: {key} differs or is missing.")

    baseline_metrics = baseline.get("metrics", {})
    current_metrics = current.get("metrics", {})
    comparisons = {}
    regressions = []
    for metric in QUALITY_METRICS:
        if metric not in baseline_metrics or metric not in current_metrics:
            raise ValueError(f"Both runs must contain '{metric}'.")
        try:
            before, after = float(baseline_metrics[metric]), float(current_metrics[metric])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Metric '{metric}' must contain finite values between 0 and 1.") from exc
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (before, after)):
            raise ValueError(f"Metric '{metric}' must contain finite values between 0 and 1.")
        delta = round(after - before, 4)
        regressed = delta < -tolerance
        comparisons[metric] = {
            "baseline": round(before, 4),
            "current": round(after, 4),
            "delta": delta,
            "allowed_drop": tolerance,
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
        "max_metric_drop": tolerance,
        "regressed_metrics": regressions,
        "comparisons": comparisons,
        "corpus_fingerprint": baseline_config["corpus_fingerprint"],
        "dataset_fingerprint": baseline_config["dataset_fingerprint"],
        "note": "A metric is flagged only when its absolute drop is greater than the configured tolerance. Matching fingerprints verify the stored corpus and labeled cases.",
    }
