import json

from app import trace_store


def test_trace_store_round_trip_and_limit(monkeypatch, tmp_path):
    from app import database

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "traces.db")
    trace_store.init_trace_store()
    trace_store.save_trace({
        "trace_id": "trace-1",
        "question": "How does retrieval work?",
        "answer_mode": "gemini",
        "retrieval_mode": "hybrid",
        "top_k": 4,
        "timings": {"embedding_ms": 1.2, "retrieval_ms": 2.3, "generation_ms": 4.5, "total_ms": 8.0},
        "token_usage": None,
        "retrieved_count": 4,
        "error": None,
    })
    trace = trace_store.get_trace("trace-1")
    assert trace["trace_id"] == "trace-1"
    assert trace["timings"]["total_ms"] == 8.0
    assert trace["token_usage"] is None
    assert trace_store.list_traces(1)[0]["question"] == "How does retrieval work?"
    assert trace_store.get_trace("missing") is None


def test_trace_store_keeps_error_details(monkeypatch, tmp_path):
    from app import database

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "errors.db")
    trace_store.init_trace_store()
    trace_store.save_trace({
        "trace_id": "trace-error", "question": "failed query", "answer_mode": "error",
        "retrieval_mode": "none", "top_k": 3, "timings": {"total_ms": 5},
        "retrieved_count": 0, "error": "embedding unavailable",
    })
    assert trace_store.get_trace("trace-error")["error"] == "embedding unavailable"
