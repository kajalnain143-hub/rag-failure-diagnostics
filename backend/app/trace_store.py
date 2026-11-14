"""Persistent query-level observability traces."""
import json
from .database import get_connection


def init_trace_store() -> None:
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_traces (
                trace_id TEXT PRIMARY KEY, question TEXT NOT NULL, answer_mode TEXT NOT NULL,
                retrieval_mode TEXT NOT NULL, top_k INTEGER NOT NULL, timings_json TEXT NOT NULL,
                token_usage_json TEXT, retrieved_count INTEGER NOT NULL DEFAULT 0,
                error TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_query_traces_created ON query_traces(created_at DESC)")
        conn.commit()


def save_trace(trace: dict) -> None:
    with get_connection() as conn:
        conn.execute("""INSERT INTO query_traces
            (trace_id, question, answer_mode, retrieval_mode, top_k, timings_json,
             token_usage_json, retrieved_count, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trace_id) DO UPDATE SET
              question=excluded.question, answer_mode=excluded.answer_mode,
              retrieval_mode=excluded.retrieval_mode, top_k=excluded.top_k,
              timings_json=excluded.timings_json, token_usage_json=excluded.token_usage_json,
              retrieved_count=excluded.retrieved_count, error=excluded.error""",
            (trace["trace_id"], str(trace.get("question", ""))[:2000],
             str(trace.get("answer_mode", "unknown")), str(trace.get("retrieval_mode", "none")),
             int(trace.get("top_k", 0)), json.dumps(trace.get("timings", {}), sort_keys=True),
             json.dumps(trace["token_usage"], sort_keys=True) if trace.get("token_usage") is not None else None,
             int(trace.get("retrieved_count", 0)), trace.get("error")))
        conn.commit()


def _decode(row):
    if row is None:
        return None
    item = dict(row)
    item["timings"] = json.loads(item.pop("timings_json"))
    raw_usage = item.pop("token_usage_json")
    item["token_usage"] = json.loads(raw_usage) if raw_usage else None
    return item


def list_traces(limit: int = 30) -> list[dict]:
    safe_limit = max(1, min(int(limit), 100))
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM query_traces ORDER BY created_at DESC, trace_id DESC LIMIT ?", (safe_limit,)
        ).fetchall()
    return [_decode(row) for row in rows]


def get_trace(trace_id: str):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM query_traces WHERE trace_id = ?", (trace_id,)).fetchone()
    return _decode(row)
