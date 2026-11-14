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
