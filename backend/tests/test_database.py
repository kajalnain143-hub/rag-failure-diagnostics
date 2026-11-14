import sqlite3

from app import database


def test_database_persists_embeddings_and_migrates(tmp_path, monkeypatch):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    database.init_db()
    document_id = database.save_document("sample.txt", ["hello world"], [[0.1, 0.2]], "test-model")
    rows = database.get_all_chunks()
    assert rows[0]["document_id"] == document_id
    assert rows[0]["embedding_model"] == "test-model"
    assert rows[0]["embedding_json"] == "[0.1, 0.2]"
    assert database.get_documents()[0]["chunk_count"] == 1


def test_database_backup_is_consistent_and_readable(tmp_path, monkeypatch):
    db_path = tmp_path / "backup-source.sqlite3"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    database.init_db()
    database.save_document("backup.txt", ["backup content"])
    payload = database.create_database_backup()
    assert payload.startswith(b"SQLite format 3")
    restored = sqlite3.connect(":memory:")
    try:
        restored.deserialize(payload)
        row = restored.execute(
            "SELECT documents.filename, chunks.content FROM documents "
            "JOIN chunks ON chunks.document_id = documents.id"
        ).fetchone()
        assert row == ("backup.txt", "backup content")
    finally:
        restored.close()
