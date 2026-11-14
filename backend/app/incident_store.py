"""Persistent incident diagnosis history and pattern summaries."""
import json
from uuid import uuid4

from .database import get_connection, column_exists
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
                trace_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    if not column_exists("incident_reports", "trace_id"):
        with get_connection() as migration_conn:
            migration_conn.execute("ALTER TABLE incident_reports ADD COLUMN trace_id TEXT")
            migration_conn.commit()
    with get_connection() as conn:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_incidents_trace ON incident_reports(trace_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_incidents_created ON incident_reports(created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_incidents_pattern ON incident_reports(primary_pattern_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incident_reports(severity)")
        conn.commit()


def save_incident(incident: dict, diagnosis: dict) -> dict:
    primary, severity = diagnosis.get("primary_pattern_id"), diagnosis.get("severity")
    if primary not in PATTERN_IDS or severity not in {"low", "medium", "high", "critical"}:
        raise ValueError("Incident diagnosis contains unsupported taxonomy or severity values.")
    incident_id = str(uuid4())
    with get_connection() as conn:
        conn.execute("""INSERT INTO incident_reports
            (incident_id, description, expected_behavior, logs, primary_pattern_id,
             secondary_pattern_ids_json, severity, confidence, diagnosis_json, trace_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (incident_id, incident.get("description", "")[:12000],
             incident.get("expected_behavior", "")[:5000], incident.get("logs", "")[:20000],
             primary, json.dumps(diagnosis.get("secondary_pattern_ids", [])), severity,
             float(diagnosis.get("confidence", 0)), json.dumps(diagnosis, sort_keys=True),
             incident.get("trace_id")))
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
            "SELECT * FROM incident_reports" + where + " ORDER BY created_at DESC, incident_id DESC LIMIT ?",
            (*values, safe_limit)).fetchall()
    return [_decode(row) for row in rows]


def incident_pattern_summary() -> dict:
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) AS count FROM incident_reports").fetchone()["count"]
        pattern_rows = conn.execute(
            "SELECT primary_pattern_id, COUNT(*) AS count FROM incident_reports "
            "GROUP BY primary_pattern_id ORDER BY count DESC, primary_pattern_id").fetchall()
        severity_rows = conn.execute(
            "SELECT severity, COUNT(*) AS count FROM incident_reports GROUP BY severity").fetchall()
    return {"total_incidents": total,
            "patterns": [{"pattern_id": row["primary_pattern_id"], "count": row["count"]} for row in pattern_rows],
            "severity": {row["severity"]: row["count"] for row in severity_rows}}


def incident_trace_analytics() -> dict:
    """Summarize latency and mode signals only for incidents explicitly linked to traces."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT i.incident_id, i.primary_pattern_id, i.severity, i.confidence,
                   t.trace_id, t.answer_mode, t.retrieval_mode, t.error,
                   t.timings_json, t.retrieved_count, t.created_at AS trace_created_at
            FROM incident_reports i LEFT JOIN query_traces t ON t.trace_id = i.trace_id
            ORDER BY i.created_at DESC
        """).fetchall()
    linked, unlinked = [], 0
    for row in rows:
        item = dict(row)
        if not item.get("trace_id") or not item.get("timings_json"):
            unlinked += 1
            continue
        timings = json.loads(item.pop("timings_json"))
        item["timings"] = timings
        item["total_ms"] = float(timings.get("total_ms", 0))
        linked.append(item)
    if not linked:
        return {"incident_count": len(rows), "linked_count": 0, "unlinked_count": unlinked,
                "avg_total_ms": None, "p95_total_ms": None, "by_pattern": [],
                "by_answer_mode": [], "by_retrieval_mode": [],
                "note": "No incidents are linked to query traces yet. Link a trace when submitting a diagnosis to enable correlation."}
    import math
    durations = sorted(item["total_ms"] for item in linked)
    p95_index = max(0, min(len(durations) - 1, math.ceil(0.95 * len(durations)) - 1))
    def grouped(field):
        buckets = {}
        for item in linked:
            key = item.get(field) or "unknown"
            buckets.setdefault(key, []).append(item)
        return [{"value": key, "incident_count": len(items),
                 "avg_total_ms": round(sum(x["total_ms"] for x in items) / len(items), 2),
                 "error_count": sum(bool(x.get("error")) for x in items)}
                for key, items in sorted(buckets.items(), key=lambda pair: (-len(pair[1]), pair[0]))]
    return {"incident_count": len(rows), "linked_count": len(linked), "unlinked_count": unlinked,
            "avg_total_ms": round(sum(durations) / len(durations), 2),
            "p95_total_ms": round(durations[p95_index], 2),
            "by_pattern": grouped("primary_pattern_id"), "by_answer_mode": grouped("answer_mode"),
            "by_retrieval_mode": grouped("retrieval_mode"),
            "note": "Correlation is descriptive, not causal. Only explicitly linked incident/trace pairs are included."}
