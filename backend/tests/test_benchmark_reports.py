"""Tests for historical benchmark report aggregation."""
from app.benchmark_reports import build_benchmark_report


def make_run(run_id, recall, created_at):
    return {"id": run_id, "name": f"batch-{run_id}", "created_at": created_at,
            "config": {"dataset_name": "suite", "strategy": "hybrid", "case_count": 4},
            "metrics": {"recall_at_k": recall, "mrr": 0.7, "ndcg_at_k": 0.8,
                        "evidence_term_coverage": 0.6, "latency_avg_ms": 12.0,
                        "latency_p50_ms": 10.0, "latency_p95_ms": 20.0}}


def test_report_orders_runs_and_calculates_metric_trends():
    report = build_benchmark_report([make_run(2, 0.7, "2026-11-02"),
                                    make_run(1, 0.8, "2026-11-01")], "suite", "hybrid")
    assert report["run_count"] == 2
    assert [row["run_id"] for row in report["history"]] == [1, 2]
    assert report["trends"]["recall_at_k"]["delta"] == -0.1


def test_report_excludes_other_datasets_and_strategies():
    runs = [make_run(1, 0.8, "2026-11-01")]
    runs.append({**make_run(2, 0.9, "2026-11-02"),
                 "config": {"dataset_name": "different", "strategy": "hybrid", "case_count": 4}})
    assert build_benchmark_report(runs, "suite", "hybrid")["run_count"] == 1
