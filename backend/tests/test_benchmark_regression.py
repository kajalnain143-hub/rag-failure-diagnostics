"""Tests for benchmark regression gate behavior."""
import pytest
from app.benchmark_regression import compare_benchmark_runs


def run(run_id, metrics, dataset="support-v1", strategy="hybrid", count=20):
    return {"id": run_id, "config": {"dataset_name": dataset, "strategy": strategy, "case_count": count, "corpus_fingerprint": "corpus-sha", "dataset_fingerprint": "dataset-sha"},
            "metrics": metrics}


def metrics(recall=0.8, mrr=0.7, ndcg=0.75, coverage=0.6):
    return {"recall_at_k": recall, "mrr": mrr, "ndcg_at_k": ndcg, "evidence_term_coverage": coverage}


def test_regression_gate_passes_when_metric_drops_are_within_tolerance():
    result = compare_benchmark_runs(run(1, metrics()), run(2, metrics(recall=0.76, mrr=0.68)))
    assert result["status"] == "passed"
    assert result["regressed_metrics"] == []


def test_regression_gate_flags_metric_drop_beyond_tolerance():
    result = compare_benchmark_runs(run(1, metrics()), run(2, metrics(recall=0.70)))
    assert result["status"] == "regression"
    assert result["regressed_metrics"] == ["recall_at_k"]
    assert result["comparisons"]["recall_at_k"]["delta"] == -0.1


def test_regression_gate_rejects_different_dataset_or_strategy():
    with pytest.raises(ValueError, match="not comparable"):
        compare_benchmark_runs(run(1, metrics()), run(2, metrics(), dataset="other"))


def test_regression_gate_rejects_missing_metrics():
    with pytest.raises(ValueError, match="must contain"):
        compare_benchmark_runs(run(1, metrics()), run(2, {"recall_at_k": 0.8}))
