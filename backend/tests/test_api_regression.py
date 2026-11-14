"""API-level regression tests for the document and query workflows."""
from fastapi.testclient import TestClient

from app import main


def client(monkeypatch):
    # Keep API tests isolated from the developer's local SQLite database.
    monkeypatch.setattr(main, "init_db", lambda: None)
    monkeypatch.setattr(main, "init_experiment_store", lambda: None)
    monkeypatch.setattr(main, "init_trace_store", lambda: None)
    return TestClient(main.app)


def test_health_endpoint_returns_request_and_security_headers(monkeypatch):
    import sqlite3

    monkeypatch.setattr(main, "get_connection", lambda: sqlite3.connect(":memory:"))
    with client(monkeypatch) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json()["database"] == "ready"
    assert response.headers.get("x-request-id")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


def test_upload_rejects_empty_file(monkeypatch):
    with client(monkeypatch) as test_client:
        response = test_client.post(
            "/api/documents/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded file is empty."


def test_upload_rejects_unsupported_file_type(monkeypatch):
    with client(monkeypatch) as test_client:
        response = test_client.post(
            "/api/documents/upload",
            files={"file": ("payload.exe", b"not executable", "application/octet-stream")},
        )

    assert response.status_code == 400
    assert "Only PDF, TXT, and Markdown" in response.json()["detail"]


def test_upload_indexes_utf8_text_document(monkeypatch):
    monkeypatch.setattr(main, "embed_text", lambda text: None)
    monkeypatch.setattr(main, "save_document", lambda filename, chunks, embeddings, model: 42)
    with client(monkeypatch) as test_client:
        response = test_client.post(
            "/api/documents/upload",
            files={"file": ("notes.txt", "Regression testing works.".encode(), "text/plain")},
        )

    assert response.status_code == 200
    assert response.json()["document_id"] == 42
    assert response.json()["chunk_count"] == 1


def test_query_returns_trace_and_retrieved_evidence(monkeypatch):
    chunks = [{"id": 11, "document_id": 3, "filename": "notes.txt", "content": "Python is a language."}]
    results = [{
        "chunk_id": 11, "document_id": 3, "filename": "notes.txt",
        "content": "Python is a language.", "score": 0.8, "retrieval_mode": "tfidf",
    }]
    saved = []
    monkeypatch.setattr(main, "get_all_chunks", lambda: chunks)
    monkeypatch.setattr(main, "embed_query", lambda question: None)
    monkeypatch.setattr(main, "retrieve", lambda *args, **kwargs: results)
    monkeypatch.setattr(main, "diagnose", lambda question, retrieved: {"status": "ok"})
    monkeypatch.setattr(main, "generate_grounded_answer", lambda question, retrieved: "Python is a language. [S1]")
    monkeypatch.setattr(main, "save_trace", lambda trace: saved.append(trace))
    with client(monkeypatch) as test_client:
        response = test_client.post("/api/query", json={"question": "What is Python?", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["trace_id"]
    assert body["answer"].startswith("Python")
    assert body["retrieved_chunks"][0]["chunk_id"] == 11
    assert set(body["timings"]) == {"embedding_ms", "retrieval_ms", "generation_ms", "total_ms"}
    assert saved[0]["trace_id"] == body["trace_id"]
    assert saved[0]["retrieved_count"] == 1


def test_query_without_documents_returns_400_and_persists_error_trace(monkeypatch):
    saved = []
    monkeypatch.setattr(main, "get_all_chunks", lambda: [])
    monkeypatch.setattr(main, "save_trace", lambda trace: saved.append(trace))
    with client(monkeypatch) as test_client:
        response = test_client.post("/api/query", json={"question": "What is Python?"})

    assert response.status_code == 400
    assert saved[0]["error"] == "Upload a document before asking a question."
    assert saved[0]["answer_mode"] == "error"


def test_batch_benchmark_runs_cases_for_selected_strategies_and_saves(monkeypatch):
    chunks = [{"id": 11, "document_id": 3, "filename": "notes.txt", "content": "Python is a programming language."}]
    monkeypatch.setattr(main, "get_all_chunks", lambda: chunks)
    monkeypatch.setattr(main, "embed_query", lambda question: [0.1, 0.2])
    def fake_retrieve(question, stored, k, **kwargs):
        return [{"chunk_id": 11, "content": stored[0]["content"]}]
    monkeypatch.setattr(main, "retrieve", fake_retrieve)
    saved = []
    monkeypatch.setattr(main, "save_experiment", lambda name, config, metrics, notes: saved.append((name, metrics)) or len(saved) + 1)
    payload = {
        "dataset_name": "smoke-suite",
        "cases": [
            {"query_id": "q1", "question": "What is Python?", "relevant_chunk_ids": [11],
             "reference_answer": "Python programming language", "k": 3},
            {"query_id": "q2", "question": "Define Python", "relevant_chunk_ids": [11],
             "reference_answer": "Python language", "k": 1},
        ],
        "strategies": ["tfidf", "hybrid"],
    }
    with client(monkeypatch) as test_client:
        response = test_client.post("/api/benchmark/run", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["case_count"] == 2
    assert set(body["strategies"]) == {"tfidf", "hybrid"}
    assert body["strategies"]["hybrid"]["metrics"]["recall_at_k"] == 1.0
    assert body["strategies"]["hybrid"]["metrics"]["latency_p95_ms"] >= 0
    assert len(body["strategies"]["tfidf"]["cases"]) == 2
    assert len(saved) == 2


def test_batch_benchmark_rejects_duplicate_query_ids(monkeypatch):
    payload = {"cases": [
        {"query_id": "same", "question": "Q1", "relevant_chunk_ids": [1], "reference_answer": "A"},
        {"query_id": "same", "question": "Q2", "relevant_chunk_ids": [1], "reference_answer": "A"},
    ]}
    with client(monkeypatch) as test_client:
        response = test_client.post("/api/benchmark/run", json=payload)
    assert response.status_code == 400
    assert "unique" in response.json()["detail"]


def test_batch_benchmark_rejects_non_integer_k(monkeypatch):
    payload = {"cases": [
        {"query_id": "q1", "question": "Question", "relevant_chunk_ids": [1],
         "reference_answer": "Answer", "k": 2.5}
    ]}
    with client(monkeypatch) as test_client:
        response = test_client.post("/api/benchmark/run", json=payload)
    assert response.status_code == 400
    assert "k must be an integer" in response.json()["detail"]


def test_benchmark_regression_endpoint_compares_saved_runs(monkeypatch):
    baseline = {"id": 10, "config": {"dataset_name": "suite", "strategy": "hybrid", "case_count": 2},
                "metrics": {"recall_at_k": 0.9, "mrr": 0.8, "ndcg_at_k": 0.85, "evidence_term_coverage": 0.7}}
    current = {"id": 11, "config": {"dataset_name": "suite", "strategy": "hybrid", "case_count": 2},
               "metrics": {"recall_at_k": 0.7, "mrr": 0.8, "ndcg_at_k": 0.85, "evidence_term_coverage": 0.7}}
    monkeypatch.setattr(main, "list_experiments", lambda limit: [current, baseline])
    with client(monkeypatch) as test_client:
        response = test_client.post("/api/benchmark/regression", json={
            "baseline_run_id": 10, "current_run_id": 11, "max_metric_drop": 0.05})
    assert response.status_code == 200
    assert response.json()["status"] == "regression"
    assert response.json()["regressed_metrics"] == ["recall_at_k"]


def test_benchmark_regression_rejects_unmatched_runs(monkeypatch):
    runs = [
        {"id": 10, "config": {"dataset_name": "suite-a", "strategy": "hybrid", "case_count": 2},
         "metrics": {"recall_at_k": 0.9, "mrr": 0.8, "ndcg_at_k": 0.85, "evidence_term_coverage": 0.7}},
        {"id": 11, "config": {"dataset_name": "suite-b", "strategy": "hybrid", "case_count": 2},
         "metrics": {"recall_at_k": 0.8, "mrr": 0.8, "ndcg_at_k": 0.85, "evidence_term_coverage": 0.7}},
    ]
    monkeypatch.setattr(main, "list_experiments", lambda limit: runs)
    with client(monkeypatch) as test_client:
        response = test_client.post("/api/benchmark/regression", json={
            "baseline_run_id": 10, "current_run_id": 11})
    assert response.status_code == 400
    assert "not comparable" in response.json()["detail"]
