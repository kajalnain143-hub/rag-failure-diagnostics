"""Tests for immutable benchmark snapshot hashing and storage."""
import sqlite3
from app import benchmark_snapshots


def test_snapshot_fingerprints_are_stable_and_change_with_inputs(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE benchmark_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_name TEXT NOT NULL,
        corpus_fingerprint TEXT NOT NULL, dataset_fingerprint TEXT NOT NULL,
        mmr_diversity REAL NOT NULL, snapshot_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    monkeypatch.setattr(benchmark_snapshots, "get_connection", lambda: conn)
    cases = [{"query_id": "q1", "question": "What?", "relevant_chunk_ids": [1],
              "reference_answer": "Answer", "k": 2}]
    chunks = [{"id": 1, "content": "Evidence", "embedding_json": None}]
    first = benchmark_snapshots.create_snapshot("suite", cases, chunks, 0.35)
    second = benchmark_snapshots.create_snapshot("suite", cases, chunks, 0.35)
    changed = benchmark_snapshots.create_snapshot(
        "suite", cases, [{**chunks[0], "content": "Changed"}], 0.35)
    assert first["corpus_fingerprint"] == second["corpus_fingerprint"]
    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]
    assert changed["corpus_fingerprint"] != first["corpus_fingerprint"]
    assert benchmark_snapshots.get_snapshot(first["snapshot_id"])["chunks"] == chunks
