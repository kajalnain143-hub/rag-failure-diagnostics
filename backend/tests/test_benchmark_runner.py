"""Focused tests for batch benchmark aggregation and timing output."""
from app.benchmark_runner import run_dataset


def test_runner_aggregates_cases_and_persists_one_run_per_strategy():
    chunks = [{"id": 7, "content": "Python is a programming language."}]
    saved = []
    def embed(question):
        return [0.2, 0.8]
    def retrieve(question, stored, k, **kwargs):
        return [{"chunk_id": 7, "content": stored[0]["content"]}]
    def save(name, config, metrics, notes):
        saved.append((name, config, metrics, notes))
        return len(saved)
    cases = [
        {"query_id": "a", "question": "What is Python?", "relevant_chunk_ids": [7], "reference_answer": "Python language", "k": 3},
        {"query_id": "b", "question": "Define Python", "relevant_chunk_ids": [7], "reference_answer": "Python programming language", "k": 2},
    ]
    result = run_dataset("unit-suite", cases, ["tfidf", "hybrid"], chunks, embed, retrieve, save)
    assert result["case_count"] == 2
    assert result["strategies"]["tfidf"]["metrics"]["recall_at_k"] == 1.0
    assert result["strategies"]["hybrid"]["metrics"]["case_count"] == 2
    assert result["strategies"]["hybrid"]["metrics"]["latency_avg_ms"] >= 0
    assert result["strategies"]["hybrid"]["metrics"]["latency_p50_ms"] >= 0
    assert result["strategies"]["hybrid"]["metrics"]["latency_p95_ms"] >= 0
    assert len(result["strategies"]["hybrid"]["cases"]) == 2
    assert len(saved) == 2
    assert saved[0][1]["dataset_name"] == "unit-suite"


def test_runner_skips_embedding_for_tfidf_strategy():
    calls = []
    def embed(question):
        calls.append(question)
        return None
    def retrieve(question, stored, k, **kwargs):
        return [{"chunk_id": 1, "content": "Gold answer."}]
    case = {"query_id": "only", "question": "Q", "relevant_chunk_ids": [1], "reference_answer": "Gold answer", "k": 1}
    run_dataset("offline", [case], ["tfidf"], [{"id": 1, "content": "Gold answer."}], embed, retrieve, lambda *args: 1)
    assert calls == []
