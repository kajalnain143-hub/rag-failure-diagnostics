"""Regression coverage for SQLite/PostgreSQL database compatibility."""
import sqlite3
from unittest.mock import MagicMock

import pytest

from app import database


def test_postgres_adapter_converts_placeholders_and_identity_columns():
    raw = MagicMock()
    adapter = database.PostgresConnection(raw)
    adapter.execute(
        "CREATE TABLE things (id INTEGER PRIMARY KEY AUTOINCREMENT)"
    )
    raw.execute.assert_called_once_with(
        "CREATE TABLE things (id BIGSERIAL PRIMARY KEY)", ()
    )
    adapter.execute("SELECT * FROM things WHERE id = ?", (3,))
    raw.execute.assert_called_with("SELECT * FROM things WHERE id = %s", (3,))


def test_postgres_adapter_closes_connection_after_context():
    raw = MagicMock()
    adapter = database.PostgresConnection(raw)
    with adapter as connection:
        assert connection is adapter
    raw.__exit__.assert_called_once()
    raw.close.assert_called_once()


def test_postgres_adapter_closes_on_error():
    raw = MagicMock()
    adapter = database.PostgresConnection(raw)
    with pytest.raises(ValueError):
        with adapter:
            raise ValueError("rollback")
    raw.__exit__.assert_called_once()
    raw.close.assert_called_once()


def test_postgres_column_exists_uses_information_schema(monkeypatch):
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.execute.return_value.fetchone.return_value = {"?column?": 1}
    monkeypatch.setattr(database, "USING_POSTGRES", True)
    monkeypatch.setattr(database, "get_connection", lambda: connection)
    assert database.column_exists("chunks", "embedding_json")
    query, params = connection.execute.call_args.args
    assert "information_schema.columns" in query
    assert params == ("chunks", "embedding_json")


def test_local_sqlite_persists_after_reopening(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "USING_POSTGRES", False)
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "persistent.sqlite")
    database.init_db()
    document_id = database.save_document("persist.txt", ["persistent evidence"])
    assert document_id > 0
    with sqlite3.connect(database.DB_PATH) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM documents WHERE id = ?", (document_id,)
        ).fetchone()[0] == 1
    assert database.get_all_chunks()[0]["content"] == "persistent evidence"
