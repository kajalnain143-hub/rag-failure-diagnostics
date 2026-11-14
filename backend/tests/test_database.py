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
