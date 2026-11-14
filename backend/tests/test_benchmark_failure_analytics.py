from app.benchmark_failure_analytics import analyze_benchmark_failures


def _run(run_id, strategy, *, recall=0.8, mrr=0.8, ndcg=0.8, coverage=0.8, cases=20):
    return {
        "id": run_id, "created_at": f"2026-11-{run_id:02d}",
        "config": {"dataset_name": "suite", "strategy": strategy, "case_count": cases, "snapshot_id": 3},
        "metrics": {"recall_at_k": recall, "mrr": mrr, "ndcg_at_k": ndcg, "evidence_term_coverage": coverage},
    }


def test_failure_analytics_flags_retrieval_and_chunk_signals():
    result = analyze_benchmark_failures([_run(1, "tfidf", recall=0.2, mrr=0.3, ndcg=0.4, coverage=0.2)], "suite")
    assert {row["pattern_id"] for row in result["signals"]} == {"P02", "P03"}
    assert result["signal_count"] == 2


def test_failure_analytics_flags_small_evaluation_sample():
    result = analyze_benchmark_failures([_run(1, "hybrid", cases=3)], "suite")
    assert result["distribution"][0]["pattern_id"] == "P09"


def test_failure_analytics_uses_latest_run_per_strategy():
    result = analyze_benchmark_failures([_run(1, "hybrid", recall=0.2), _run(2, "hybrid", recall=0.9)], "suite")
    assert result["analyzed_strategy_count"] == 1
    assert result["signal_count"] == 0


def test_failure_analytics_ignores_other_datasets():
    result = analyze_benchmark_failures([_run(1, "hybrid")], "other-suite")
    assert result["analyzed_strategy_count"] == 0
    assert result["distribution"] == []
