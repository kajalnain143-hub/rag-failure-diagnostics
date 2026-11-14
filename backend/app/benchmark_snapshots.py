"""Immutable corpus and labeled dataset snapshots for reproducible benchmarks."""
from __future__ import annotations
import hashlib
import json
from .database import get_connection, USING_POSTGRES


class SnapshotIntegrityError(ValueError):
    """Raised when persisted snapshot content no longer matches its fingerprints."""


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
                "mmr_diversity": mmr_diversity, "corpus_fingerprint": _fingerprint(corpus),
                "dataset_fingerprint": _fingerprint(dataset)}
    with get_connection() as conn:
        sql = """INSERT INTO benchmark_snapshots
            (dataset_name, corpus_fingerprint, dataset_fingerprint, mmr_diversity, snapshot_json)
            VALUES (?, ?, ?, ?, ?)"""
        values = (dataset_name, snapshot["corpus_fingerprint"], snapshot["dataset_fingerprint"],
                  mmr_diversity, _canonical(snapshot))
        if USING_POSTGRES:
            snapshot_id = conn.execute(sql + " RETURNING id", values).fetchone()["id"]
        else:
            snapshot_id = conn.execute(sql, values).lastrowid
        conn.commit()
        snapshot["snapshot_id"] = int(snapshot_id)
    return snapshot


def verify_snapshot(snapshot, stored_corpus_fingerprint=None, stored_dataset_fingerprint=None):
    if not isinstance(snapshot, dict):
        raise SnapshotIntegrityError("Snapshot payload is not a valid object.")
    try:
        corpus_fingerprint, dataset_fingerprint = _fingerprint(snapshot["chunks"]), _fingerprint(snapshot["cases"])
        if corpus_fingerprint != snapshot["corpus_fingerprint"]:
            raise SnapshotIntegrityError("Snapshot corpus fingerprint mismatch; stored content may be corrupted.")
        if dataset_fingerprint != snapshot["dataset_fingerprint"]:
            raise SnapshotIntegrityError("Snapshot dataset fingerprint mismatch; stored labels may be corrupted.")
        if stored_corpus_fingerprint is not None and corpus_fingerprint != stored_corpus_fingerprint:
            raise SnapshotIntegrityError("Snapshot corpus fingerprint does not match its database metadata.")
        if stored_dataset_fingerprint is not None and dataset_fingerprint != stored_dataset_fingerprint:
            raise SnapshotIntegrityError("Snapshot dataset fingerprint does not match its database metadata.")
        if not isinstance(snapshot["dataset_name"], str) or not snapshot["dataset_name"]:
            raise SnapshotIntegrityError("Snapshot dataset name is invalid.")
        diversity = snapshot["mmr_diversity"]
        if isinstance(diversity, bool) or not isinstance(diversity, (int, float)) or not 0 <= diversity <= 1:
            raise SnapshotIntegrityError("Snapshot MMR diversity is invalid.")
        if not isinstance(snapshot["cases"], list) or not isinstance(snapshot["chunks"], list):
            raise SnapshotIntegrityError("Snapshot cases and chunks must be arrays.")
    except SnapshotIntegrityError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotIntegrityError("Snapshot is missing required fields or contains invalid data.") from exc
    return True


def get_snapshot(snapshot_id):
    with get_connection() as conn:
        row = conn.execute("""SELECT id, created_at, snapshot_json,
            corpus_fingerprint, dataset_fingerprint FROM benchmark_snapshots WHERE id = ?""",
            (snapshot_id,)).fetchone()
    if row is None:
        return None
    try:
        snapshot = json.loads(row["snapshot_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise SnapshotIntegrityError("Snapshot JSON is corrupted and cannot be decoded.") from exc
    verify_snapshot(snapshot, row["corpus_fingerprint"], row["dataset_fingerprint"])
    snapshot["snapshot_id"], snapshot["created_at"] = int(row["id"]), row["created_at"]
    return snapshot
