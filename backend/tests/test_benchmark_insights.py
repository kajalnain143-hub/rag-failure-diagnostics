"""Tests for deterministic benchmark intelligence summaries."""
from app.benchmark_reports import build_benchmark_insights


def _run(run_id, strategy, recall, corpus="c1", dataset="d1", created="2026-11-14"):
    return {
        "id": run_id, "created_at": created,
        "config": {"dataset_name": "suite", "strategy": strategy, "case_count": 4,
                   "snapshot_id": 7, "corpus_fingerprint": corpus,
                   "dataset_fingerprint": dataset},
        "metrics": {"recall_at_k": recall, "mrr": 0.7, "ndcg_at_k": 0.75,
                    "evidence_term_coverage": 0.6, "latency_p95_ms": 40.0},
    }


def test_insights_rank_strategies_and_recommend_lowest_latency():
    runs = [_run(1, "tfidf", 0.6), _run(2, "hybrid", 0.9),
            _run(3, "hybrid_mmr", 0.8)]
    result = build_benchmark_insights(runs, "suite")
    assert result["comparable"] is True
    assert result["best_strategy"] == "hybrid"
    assert result["strategy_count"] == 3
    assert any("p95 latency" in item for item in result["recommendations"])


def test_insights_do_not_compare_different_snapshots():
    runs = [_run(1, "tfidf", 0.6), _run(2, "hybrid", 0.9, corpus="other")]
    result = build_benchmark_insights(runs, "suite")
    assert result["comparable"] is False
    assert result["best_strategy"] is None
    assert any("different snapshots" in item for item in result["recommendations"])


def test_insights_select_latest_run_per_strategy():
    runs = [_run(1, "hybrid", 0.5, created="2026-11-13"),
            _run(2, "hybrid", 0.9, created="2026-11-14")]
    result = build_benchmark_insights(runs, "suite")
    assert result["strategies"][0]["run_id"] == 2


def test_insights_return_empty_guidance_without_runs():
    result = build_benchmark_insights([], "suite")
    assert result["strategy_count"] == 0
    assert result["best_strategy"] is None
