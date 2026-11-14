"""Gemini live-check tests never make network calls or require API credentials."""
from types import SimpleNamespace

from app import llm, main


class FakeModels:
    def generate_content(self, **kwargs):
        return SimpleNamespace(text="OK")

    def embed_content(self, **kwargs):
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2])])


def test_gemini_verification_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert llm.verify_live_api()["status"] == "not_configured"


def test_gemini_verification_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(llm, "_client", lambda: SimpleNamespace(models=FakeModels()))
    result = llm.verify_live_api()
    assert result["status"] == "passed"
    assert result["generation_status"] == "passed"
    assert result["embedding_status"] == "passed"
    assert "fake-key" not in str(result)


def test_gemini_verification_reports_failures_without_secrets(monkeypatch):
    class BrokenModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("secret should not appear in response")

        def embed_content(self, **kwargs):
            raise RuntimeError("secret should not appear in response")

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(llm, "_client", lambda: SimpleNamespace(models=BrokenModels()))
    result = llm.verify_live_api()
    assert result["status"] == "failed"
    assert result["generation_status"] == "failed"
    assert result["embedding_status"] == "failed"
    assert "secret" not in str(result)
    assert "fake-key" not in str(result)


def test_gemini_health_returns_cached_verification(monkeypatch):
    monkeypatch.setattr(main, "GEMINI_LIVE_STATUS", {"status": "passed"})
    assert main.gemini_health_check() == {"status": "passed"}
