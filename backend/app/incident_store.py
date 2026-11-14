"""Persistent incident diagnosis history and pattern summaries in SQLite."""
import json
from uuid import uuid4

from .database import get_connection
from .failure_clinic import PATTERN_IDS


def init_incident_store() -> None:
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS incident_reports (
                incident_id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                expected_behavior TEXT NOT NULL DEFAULT '',
                logs TEXT NOT NULL DEFAULT '',
                primary_pattern_id TEXT NOT NULL,
                secondary_pattern_ids_json TEXT NOT NULL,
                severity TEXT NOT NULL,
                confidence REAL NOT NULL,
                diagnosis_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_incidents_created ON incident_reports(created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_incidents_pattern ON incident_reports(primary_pattern_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incident_reports(severity)")
        conn.commit()


def save_incident(incident: dict, diagnosis: dict) -> dict:
    primary = diagnosis.get("primary_pattern_id")
    severity = diagnosis.get("severity")
    if primary not in PATTERN_IDS or severity not in {"low", "medium", "high", "critical"}:
        raise ValueError("Incident diagnosis contains unsupported taxonomy or severity values.")
    incident_id = str(uuid4())
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO incident_reports
            (incident_id, description, expected_behavior, logs, primary_pattern_id,
             secondary_pattern_ids_json, severity, confidence, diagnosis_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (incident_id, incident.get("description", "")[:12000],
             incident.get("expected_behavior", "")[:5000], incident.get("logs", "")[:20000],
             primary, json.dumps(diagnosis.get("secondary_pattern_ids", [])),
             severity, float(diagnosis.get("confidence", 0)),
             json.dumps(diagnosis, sort_keys=True)),
        )
        conn.commit()
    return get_incident(incident_id)


def _decode(row):
    if row is None:
        return None
    item = dict(row)
    item["secondary_pattern_ids"] = json.loads(item.pop("secondary_pattern_ids_json"))
    item["diagnosis"] = json.loads(item.pop("diagnosis_json"))
    return item


def get_incident(incident_id: str):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM incident_reports WHERE incident_id = ?", (incident_id,)).fetchone()
    return _decode(row)


def list_incidents(limit: int = 50, pattern_id: str | None = None,
                   severity: str | None = None) -> list[dict]:
    safe_limit = max(1, min(int(limit), 200))
    clauses, values = [], []
    if pattern_id:
        if pattern_id not in PATTERN_IDS:
            raise ValueError("Unsupported failure pattern ID.")
        clauses.append("primary_pattern_id = ?")
        values.append(pattern_id)
    if severity:
        if severity not in {"low", "medium", "high", "critical"}:
            raise ValueError("Unsupported incident severity.")
        clauses.append("severity = ?")
        values.append(severity)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM incident_reports" + where + " ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (*values, safe_limit),
        ).fetchall()
    return [_decode(row) for row in rows]


def incident_pattern_summary() -> dict:
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) AS count FROM incident_reports").fetchone()["count"]
        pattern_rows = conn.execute(
            "SELECT primary_pattern_id, COUNT(*) AS count FROM incident_reports "
            "GROUP BY primary_pattern_id ORDER BY count DESC, primary_pattern_id"
        ).fetchall()
        severity_rows = conn.execute(
            "SELECT severity, COUNT(*) AS count FROM incident_reports GROUP BY severity"
        ).fetchall()
    return {
        "total_incidents": total,
        "patterns": [{"pattern_id": row["primary_pattern_id"], "count": row["count"]} for row in pattern_rows],
        "severity": {row["severity"]: row["count"] for row in severity_rows},
    }
