"""Immutable corpus and labeled dataset snapshots for reproducible benchmarks."""
from __future__ import annotations
import hashlib
import json
from .database import get_connection


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _fingerprint(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def init_snapshot_store():
    with get_connection() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS benchmark_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_name TEXT NOT NULL,
            corpus_fingerprint TEXT NOT NULL, dataset_fingerprint TEXT NOT NULL,
            mmr_diversity REAL NOT NULL, snapshot_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        conn.commit()


def create_snapshot(dataset_name, cases, chunks, mmr_diversity):
    corpus, dataset = json.loads(_canonical(chunks)), json.loads(_canonical(cases))
    snapshot = {"dataset_name": dataset_name, "cases": dataset, "chunks": corpus,
                "mmr_diversity": mmr_diversity,
                "corpus_fingerprint": _fingerprint(corpus),
                "dataset_fingerprint": _fingerprint(dataset)}
    with get_connection() as conn:
        cursor = conn.execute("""INSERT INTO benchmark_snapshots
            (dataset_name, corpus_fingerprint, dataset_fingerprint, mmr_diversity, snapshot_json)
            VALUES (?, ?, ?, ?, ?)""",
            (dataset_name, snapshot["corpus_fingerprint"], snapshot["dataset_fingerprint"],
             mmr_diversity, _canonical(snapshot)))
        conn.commit()
        snapshot["snapshot_id"] = int(cursor.lastrowid)
    return snapshot


def get_snapshot(snapshot_id):
    with get_connection() as conn:
        row = conn.execute("SELECT id, created_at, snapshot_json FROM benchmark_snapshots WHERE id = ?",
                           (snapshot_id,)).fetchone()
    if row is None:
        return None
    snapshot = json.loads(row["snapshot_json"])
    snapshot["snapshot_id"] = int(row["id"])
    snapshot["created_at"] = row["created_at"]
    return snapshot
