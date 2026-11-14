"""One-time, idempotent copy of application data from SQLite into PostgreSQL.

Usage:
  DATABASE_URL=... python migrate_sqlite_to_postgres.py --source /path/to/rag_diagnostics.db
The script never deletes or overwrites destination rows. Review row counts before switching production traffic.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import psycopg
from psycopg import sql

from app.database import DATABASE_URL, init_db
from app.experiment_store import init_experiment_store
from app.trace_store import init_trace_store
from app.benchmark_snapshots import init_snapshot_store
from app.incident_store import init_incident_store


TABLE_ORDER = (
    "documents",
    "chunks",
    "experiment_runs",
    "query_traces",
    "benchmark_snapshots",
    "incident_reports",
)
SEQUENCE_TABLES = ("documents", "chunks", "experiment_runs", "benchmark_snapshots")


def migrate(source_path: str, database_url: str) -> dict[str, int]:
    if not database_url.startswith(("postgres://", "postgresql://")):
        raise ValueError("DATABASE_URL must be a PostgreSQL connection URL.")
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"SQLite source database not found: {source_path}")

    # Initialize every application table before copying any rows.
    init_db()
    init_experiment_store()
    init_trace_store()
    init_snapshot_store()
    init_incident_store()

    counts: dict[str, int] = {}
    with sqlite3.connect(source_path) as source, psycopg.connect(database_url) as target:
        source.row_factory = sqlite3.Row
        existing = {row["name"] for row in source.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        for table in TABLE_ORDER:
            if table not in existing:
                counts[table] = 0
                continue
            columns = [row["name"] for row in source.execute(f"PRAGMA table_info({table})")]
            if not columns:
                counts[table] = 0
                continue
            rows = source.execute(f"SELECT * FROM {table}").fetchall()
            if rows:
                insert = sql.SQL("INSERT INTO {} ({}) VALUES ({}) ON CONFLICT DO NOTHING").format(
                    sql.Identifier(table),
                    sql.SQL(", ").join(map(sql.Identifier, columns)),
                    sql.SQL(", ").join(sql.Placeholder() for _ in columns),
                )
                with target.cursor() as cursor:
                    cursor.executemany(insert, [tuple(row[column] for column in columns) for row in rows])
            counts[table] = len(rows)

        # Keep generated IDs above the imported maximum for subsequent inserts.
        with target.cursor() as cursor:
            for table in SEQUENCE_TABLES:
                cursor.execute(
                    "SELECT pg_get_serial_sequence(%s, 'id')",
                    (table,),
                )
                sequence = cursor.fetchone()[0]
                if sequence:
                    cursor.execute(
                        sql.SQL("SELECT setval(%s, GREATEST(COALESCE((SELECT MAX(id) FROM {}), 1), 1), true)").format(
                            sql.Identifier(table)
                        ),
                        (sequence,),
                    )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Path to the source SQLite database file")
    args = parser.parse_args()
    if not DATABASE_URL:
        parser.error("Set DATABASE_URL to the destination PostgreSQL connection string.")
    counts = migrate(args.source, DATABASE_URL)
    print("SQLite to PostgreSQL copy finished (existing destination rows were preserved):")
    for table, count in counts.items():
        print(f"  {table}: {count} source rows scanned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
