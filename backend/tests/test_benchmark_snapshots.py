"""Integrity verification tests for persisted benchmark snapshots."""
import json
import sqlite3

import pytest

from app import benchmark_snapshots as snapshots


def _store(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE benchmark_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_name TEXT NOT NULL,
        corpus_fingerprint TEXT NOT NULL, dataset_fingerprint TEXT NOT NULL,
        mmr_diversity REAL NOT NULL, snapshot_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    monkeypatch.setattr(snapshots, "get_connection", lambda: conn)
    return conn


def _inputs():
    cases = [{"query_id": "q1", "question": "What?", "relevant_chunk_ids": [1],
              "reference_answer": "Answer", "k": 2}]
    chunks = [{"id": 1, "content": "Evidence", "embedding_json": None}]
    return cases, chunks


def test_snapshot_fingerprints_are_stable_and_change_with_inputs(monkeypatch):
    _store(monkeypatch)
    cases, chunks = _inputs()
    first = snapshots.create_snapshot("suite", cases, chunks, 0.35)
    second = snapshots.create_snapshot("suite", cases, chunks, 0.35)
    changed = snapshots.create_snapshot("suite", cases, [{**chunks[0], "content": "Changed"}], 0.35)
    assert first["corpus_fingerprint"] == second["corpus_fingerprint"]
    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]
    assert changed["corpus_fingerprint"] != first["corpus_fingerprint"]
    assert snapshots.get_snapshot(first["snapshot_id"])["chunks"] == chunks


def test_snapshot_read_detects_tampered_corpus(monkeypatch):
    conn = _store(monkeypatch)
    cases, chunks = _inputs()
    saved = snapshots.create_snapshot("suite", cases, chunks, 0.35)
    with conn:
        row = conn.execute("SELECT snapshot_json FROM benchmark_snapshots WHERE id=?",
                           (saved["snapshot_id"],)).fetchone()
        data = json.loads(row["snapshot_json"])
        data["chunks"][0]["content"] = "Tampered evidence"
        conn.execute("UPDATE benchmark_snapshots SET snapshot_json=? WHERE id=?",
                     (json.dumps(data), saved["snapshot_id"]))
    with pytest.raises(snapshots.SnapshotIntegrityError, match="corpus fingerprint mismatch"):
        snapshots.get_snapshot(saved["snapshot_id"])


def test_snapshot_read_detects_tampered_dataset_labels(monkeypatch):
    conn = _store(monkeypatch)
    cases, chunks = _inputs()
    saved = snapshots.create_snapshot("suite", cases, chunks, 0.35)
    with conn:
        row = conn.execute("SELECT snapshot_json FROM benchmark_snapshots WHERE id=?",
                           (saved["snapshot_id"],)).fetchone()
        data = json.loads(row["snapshot_json"])
        data["cases"][0]["reference_answer"] = "Changed label"
        conn.execute("UPDATE benchmark_snapshots SET snapshot_json=? WHERE id=?",
                     (json.dumps(data), saved["snapshot_id"]))
    with pytest.raises(snapshots.SnapshotIntegrityError, match="dataset fingerprint mismatch"):
        snapshots.get_snapshot(saved["snapshot_id"])


def test_snapshot_read_detects_database_fingerprint_metadata_tampering(monkeypatch):
    conn = _store(monkeypatch)
    cases, chunks = _inputs()
    saved = snapshots.create_snapshot("suite", cases, chunks, 0.35)
    with conn:
        conn.execute("UPDATE benchmark_snapshots SET corpus_fingerprint=? WHERE id=?",
                     ("wrong-hash", saved["snapshot_id"]))
    with pytest.raises(snapshots.SnapshotIntegrityError, match="database metadata"):
        snapshots.get_snapshot(saved["snapshot_id"])


def test_snapshot_read_rejects_corrupt_json(monkeypatch):
    conn = _store(monkeypatch)
    cases, chunks = _inputs()
    saved = snapshots.create_snapshot("suite", cases, chunks, 0.35)
    with conn:
        conn.execute("UPDATE benchmark_snapshots SET snapshot_json=? WHERE id=?",
                     ("{not-json", saved["snapshot_id"]))
    with pytest.raises(snapshots.SnapshotIntegrityError, match="cannot be decoded"):
        snapshots.get_snapshot(saved["snapshot_id"])
