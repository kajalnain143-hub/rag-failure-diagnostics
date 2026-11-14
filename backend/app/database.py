import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "rag_diagnostics.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


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
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)")}
        if "embedding_json" not in columns:
            conn.execute("ALTER TABLE chunks ADD COLUMN embedding_json TEXT")
        if "embedding_model" not in columns:
            conn.execute("ALTER TABLE chunks ADD COLUMN embedding_model TEXT")
        conn.commit()


def save_document(filename: str, chunks: list[str], embeddings: list[list[float] | None] | None = None,
                  embedding_model: str | None = None) -> int:
    embeddings = embeddings or [None] * len(chunks)
    if len(embeddings) != len(chunks):
        raise ValueError("Each chunk must have exactly one embedding slot.")
    with get_connection() as conn:
        cursor = conn.execute("INSERT INTO documents (filename) VALUES (?)", (filename,))
        document_id = cursor.lastrowid
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
