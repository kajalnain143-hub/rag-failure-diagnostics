"""Boundary and malformed-value tests for the CI benchmark quality gate."""
import pytest

from app.benchmark_regression import compare_benchmark_runs


def _run(run_id, recall=0.8):
    return {
        "id": run_id,
        "config": {
            "dataset_name": "ci-smoke",
            "strategy": "tfidf",
            "case_count": 2,
            "corpus_fingerprint": "corpus-sha",
            "dataset_fingerprint": "dataset-sha",
        },
        "metrics": {
            "recall_at_k": recall,
            "mrr": 0.7,
            "ndcg_at_k": 0.75,
            "evidence_term_coverage": 0.6,
        },
    }


@pytest.mark.parametrize("tolerance", [-0.01, 1.01, float("nan"), float("inf"), "bad"])
def test_quality_gate_rejects_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match="max_metric_drop"):
        compare_benchmark_runs(_run(1), _run(2), max_metric_drop=tolerance)


@pytest.mark.parametrize("metric_value", [-0.1, 1.1, float("nan"), float("inf")])
def test_quality_gate_rejects_invalid_metric_values(metric_value):
    current = _run(2)
    current["metrics"]["recall_at_k"] = metric_value
    with pytest.raises(ValueError, match="finite values between 0 and 1"):
        compare_benchmark_runs(_run(1), current)


def test_quality_gate_accepts_drop_exactly_at_tolerance():
    result = compare_benchmark_runs(_run(1, recall=0.8), _run(2, recall=0.75))
    assert result["status"] == "passed"
    assert result["comparisons"]["recall_at_k"]["delta"] == -0.05
