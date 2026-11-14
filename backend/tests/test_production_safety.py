import pytest

from app import main


def test_cors_origins_parse_explicit_trusted_origins(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://app.example.com, https://admin.example.com/")
    assert main.configured_cors_origins() == [
        "https://app.example.com",
        "https://admin.example.com",
    ]


@pytest.mark.parametrize("value", ["", "*", "https://app.example.com,*"])
def test_cors_origins_reject_empty_or_wildcard(monkeypatch, value):
    monkeypatch.setenv("CORS_ORIGINS", value)
    with pytest.raises(RuntimeError):
        main.configured_cors_origins()


def test_health_check_reports_database_readiness(monkeypatch):
    import sqlite3

    monkeypatch.setattr(main, "get_connection", lambda: sqlite3.connect(":memory:"))
    result = main.health_check()
    assert result["status"] == "healthy"
    assert result["database"] == "ready"


def test_health_check_returns_service_unavailable_when_database_fails(monkeypatch):
    from fastapi import HTTPException

    def unavailable():
        raise OSError("database offline")

    monkeypatch.setattr(main, "get_connection", unavailable)
    with pytest.raises(HTTPException) as error:
        main.health_check()
    assert error.value.status_code == 503
