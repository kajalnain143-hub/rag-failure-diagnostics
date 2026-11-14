"""SQLite-backed history for reproducible retrieval/evaluation experiments."""
import json
from .database import get_connection


def init_experiment_store() -> None:
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiment_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                config_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def save_experiment(name: str, config: dict, metrics: dict, notes: str = "") -> int:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Experiment name cannot be empty.")
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO experiment_runs (name, config_json, metrics_json, notes) VALUES (?, ?, ?, ?)",
            (clean_name[:120], json.dumps(config, sort_keys=True), json.dumps(metrics, sort_keys=True), notes[:2000]),
        )
        conn.commit()
        return int(cursor.lastrowid)


def list_experiments(limit: int = 50) -> list[dict]:
    safe_limit = max(1, min(int(limit), 100))
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, name, config_json, metrics_json, notes, created_at "
            "FROM experiment_runs ORDER BY id DESC LIMIT ?", (safe_limit,)
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["config"] = json.loads(item.pop("config_json"))
        item["metrics"] = json.loads(item.pop("metrics_json"))
        result.append(item)
    return result
