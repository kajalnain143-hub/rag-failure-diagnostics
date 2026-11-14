from app.query_failure_diagnostics import diagnose_query_failure


def test_zero_recall_is_classified_as_retrieval_miss():
    result = diagnose_query_failure({
        "recall_at_k": 0, "mrr": 0, "ndcg_at_k": 0,
        "evidence_term_coverage": 0.8,
    })
    assert result["classification"] == "retrieval_miss"
    assert result["signals"][0]["pattern_id"] == "P03"
    assert result["signals"][0]["severity"] == "high"


def test_relevant_but_poorly_ranked_evidence_is_flagged():
    result = diagnose_query_failure({
        "recall_at_k": 1, "mrr": 0.2, "ndcg_at_k": 0.3,
        "evidence_term_coverage": 0.8,
    })
    assert result["classification"] == "ranking_quality"


def test_low_lexical_coverage_adds_separate_signal():
    result = diagnose_query_failure({
        "recall_at_k": 1, "mrr": 1, "ndcg_at_k": 1,
        "evidence_term_coverage": 0.1,
    })
    assert result["classification"] == "evidence_coverage"
    assert result["signals"][0]["pattern_id"] == "P02"


def test_multiple_failure_signals_are_preserved():
    result = diagnose_query_failure({
        "recall_at_k": 0, "mrr": 0, "ndcg_at_k": 0,
        "evidence_term_coverage": 0.1,
    })
    assert result["classification"] == "multiple_signals"
    assert {signal["pattern_id"] for signal in result["signals"]} == {"P02", "P03"}


def test_healthy_metrics_do_not_claim_semantic_faithfulness():
    result = diagnose_query_failure({
        "recall_at_k": 1, "mrr": 1, "ndcg_at_k": 1,
        "evidence_term_coverage": 0.9,
    })
    assert result["classification"] == "no_threshold_failure"
    assert "do not prove answer faithfulness" in result["note"]
