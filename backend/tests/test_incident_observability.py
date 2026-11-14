from app import database, incident_store, trace_store


def test_incident_trace_link_and_observability(monkeypatch, tmp_path):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "linked.db")
    trace_store.init_trace_store()
    incident_store.init_incident_store()
    trace_store.save_trace({
        "trace_id": "trace-linked", "question": "Why did retrieval fail?",
        "answer_mode": "llm_error", "retrieval_mode": "hybrid", "top_k": 4,
        "timings": {"total_ms": 120.0, "retrieval_ms": 15.0},
        "retrieved_count": 2, "error": "generation timeout",
    })
    incident_store.save_incident(
        {"description": "The generated answer timed out", "trace_id": "trace-linked"},
        {"primary_pattern_id": "P03", "secondary_pattern_ids": [], "severity": "high",
         "confidence": 0.8},
    )
    report = incident_store.incident_trace_analytics()
    assert report["linked_count"] == 1
    assert report["unlinked_count"] == 0
    assert report["avg_total_ms"] == 120.0
    assert report["by_answer_mode"][0]["value"] == "llm_error"
    assert report["by_answer_mode"][0]["error_count"] == 1


def test_incident_trace_analytics_handles_unlinked_incidents(monkeypatch, tmp_path):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "unlinked.db")
    incident_store.init_incident_store()
    incident_store.save_incident(
        {"description": "An incident without a trace"},
        {"primary_pattern_id": "P02", "secondary_pattern_ids": [], "severity": "medium",
         "confidence": 0.6},
    )
    report = incident_store.incident_trace_analytics()
    assert report["incident_count"] == 1
    assert report["linked_count"] == 0
    assert report["unlinked_count"] == 1
    assert report["p95_total_ms"] is None
