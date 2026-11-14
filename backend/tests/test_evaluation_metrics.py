from app.evaluation_metrics import (
    aggregate_metrics, evaluate_case, evidence_term_coverage,
    ndcg_at_k, recall_at_k, reciprocal_rank,
)


def test_ranking_metrics_for_relevant_chunk():
    ranked = [10, 20, 30]
    relevant = [20, 30]
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert reciprocal_rank(ranked, relevant) == 0.5
    assert 0 < ndcg_at_k(ranked, relevant, 3) <= 1


def test_evidence_coverage_is_bounded_and_lexical():
    score = evidence_term_coverage("Python supports typed hints", ["Python has typed hints"])
    assert score == 1.0
    assert evidence_term_coverage("", ["anything"]) == 0.0


def test_evaluate_case_and_aggregate():
    case = {
        "query_id": "q1", "question": "What is Python?",
        "relevant_chunk_ids": [2], "reference_answer": "Python is a language.", "k": 2,
    }
    rows = [
        {"chunk_id": 2, "content": "Python is a programming language."},
        {"chunk_id": 1, "content": "Other context."},
    ]
    result = evaluate_case(case, rows)
    assert result["recall_at_k"] == 1.0
    assert result["mrr"] == 1.0
    assert aggregate_metrics([result])["case_count"] == 1
    assert aggregate_metrics([])["case_count"] == 0
