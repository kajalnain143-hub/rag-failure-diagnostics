from app.benchmark_optimizer import rank_mmr_sweep


def test_sweep_ranks_quality_first_and_latency_as_tiebreaker():
    result = rank_mmr_sweep([
        {"mmr_diversity": 0.2, "experiment_id": 1, "snapshot_id": 10,
         "metrics": {"recall_at_k": .8, "mrr": .8, "ndcg_at_k": .8,
                     "evidence_term_coverage": .8, "latency_p95_ms": 100}},
        {"mmr_diversity": 0.5, "experiment_id": 2, "snapshot_id": 11,
         "metrics": {"recall_at_k": .9, "mrr": .9, "ndcg_at_k": .9,
                     "evidence_term_coverage": .9, "latency_p95_ms": 150}},
    ])
    assert result["best_mmr_diversity"] == 0.5
    assert result["best_experiment_id"] == 2


def test_latency_breaks_equal_quality_tie():
    result = rank_mmr_sweep([
        {"mmr_diversity": .7, "experiment_id": 7,
         "metrics": {"recall_at_k": .8, "mrr": .8, "ndcg_at_k": .8,
                     "evidence_term_coverage": .8, "latency_p95_ms": 90}},
        {"mmr_diversity": .3, "experiment_id": 3,
         "metrics": {"recall_at_k": .8, "mrr": .8, "ndcg_at_k": .8,
                     "evidence_term_coverage": .8, "latency_p95_ms": 40}},
    ])
    assert result["best_mmr_diversity"] == .3


def test_empty_or_incomplete_sweep_is_safe():
    assert rank_mmr_sweep([])["best_experiment_id"] is None
    assert rank_mmr_sweep([{"mmr_diversity": .3, "experiment_id": 1,
                            "metrics": {"mrr": .9}}])["runs"] == []
