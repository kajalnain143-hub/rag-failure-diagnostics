"""Database access for local SQLite and production PostgreSQL deployments."""
import os
import re
import sqlite3
from pathlib import Path
from urllib.parse import urlparse

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DB_PATH = Path(os.getenv("RAG_DB_PATH", str(Path(__file__).resolve().parents[1] / "rag_diagnostics.db")))
USING_POSTGRES = DATABASE_URL.startswith(("postgres://", "postgresql://"))


class PostgresConnection:
    """Small DB-API adapter preserving the existing qmark-based store interface."""
    def __init__(self, connection):
        self._connection = connection

    def __enter__(self):
        self._connection.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._connection.__exit__(exc_type, exc, tb)

    def execute(self, sql, params=()):
        sql = sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY")
        sql = re.sub(r"\?", "%s", sql)
        return self._connection.execute(sql, params)

    def executemany(self, sql, params):
        sql = sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY")
        sql = re.sub(r"\?", "%s", sql)
        return self._connection.cursor().executemany(sql, params)

    def commit(self):
        self._connection.commit()

    def rollback(self):
        self._connection.rollback()

    def close(self):
        self._connection.close()


def get_connection():
    if USING_POSTGRES:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("PostgreSQL support requires psycopg[binary].") from exc
        return PostgresConnection(psycopg.connect(DATABASE_URL, row_factory=dict_row, connect_timeout=10))
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def column_exists(table: str, column: str) -> bool:
    if USING_POSTGRES:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = ? AND column_name = ?",
                (table, column),
            ).fetchone()
            return row is not None
    with get_connection() as conn:
        return any(row["name"] == column for row in conn.execute(f"PRAGMA table_info({table})"))


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding_json TEXT,
                embedding_model TEXT,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)
        if not column_exists("chunks", "embedding_json"):
            conn.execute("ALTER TABLE chunks ADD COLUMN embedding_json TEXT")
        if not column_exists("chunks", "embedding_model"):
            conn.execute("ALTER TABLE chunks ADD COLUMN embedding_model TEXT")
        conn.commit()


def save_document(filename: str, chunks: list[str], embeddings: list[list[float] | None] | None = None,
                  embedding_model: str | None = None) -> int:
    embeddings = embeddings or [None] * len(chunks)
    if len(embeddings) != len(chunks):
        raise ValueError("Each chunk must have exactly one embedding slot.")
    with get_connection() as conn:
        if USING_POSTGRES:
            document_id = conn.execute(
                "INSERT INTO documents (filename) VALUES (?) RETURNING id", (filename,)
            ).fetchone()["id"]
        else:
            document_id = conn.execute("INSERT INTO documents (filename) VALUES (?)", (filename,)).lastrowid
        conn.executemany(
            """INSERT INTO chunks
               (document_id, chunk_index, content, embedding_json, embedding_model)
               VALUES (?, ?, ?, ?, ?)""",
            [(document_id, index, chunk,
              __import__("json").dumps(embeddings[index]) if embeddings[index] else None,
              embedding_model if embeddings[index] else None)
             for index, chunk in enumerate(chunks)],
        )
        conn.commit()
        return int(document_id)


def get_all_chunks():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT chunks.id, chunks.document_id, documents.filename,
                   chunks.chunk_index, chunks.content, chunks.embedding_json,
                   chunks.embedding_model
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            ORDER BY chunks.id
        """).fetchall()
        return [dict(row) for row in rows]


def get_documents():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT documents.id, documents.filename, documents.uploaded_at,
                   COUNT(chunks.id) AS chunk_count
            FROM documents
            LEFT JOIN chunks ON chunks.document_id = documents.id
            GROUP BY documents.id
            ORDER BY documents.id DESC
        """).fetchall()
        return [dict(row) for row in rows]


def create_database_backup() -> bytes:
    """Create a SQLite backup; PostgreSQL backups must use pg_dump."""
    if USING_POSTGRES:
        raise RuntimeError("For PostgreSQL, create backups with pg_dump or the managed provider backup.")
    with get_connection() as source:
        target = sqlite3.connect(":memory:")
        try:
            source.backup(target)
            return target.serialize()
        finally:
            target.close()
